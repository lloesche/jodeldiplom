//! macOS: what Core Audio reports about each output device, to find out
//! why a stream sounds wrong (too quiet, say).
//!
//! `cargo run --release -p jodel-output --example coreaudio_info -- [--tone <device>]`
//!
//! For every output device: hog mode and mixing, and per output stream its
//! current physical format, its virtual format (what an IOProc exchanges)
//! and every available physical format, each as the raw description (bits,
//! bytes per frame, flags spelled out); then the device's volume and mute
//! controls (master and per channel, with whether they can be set). This
//! reads only and changes nothing.
//!
//! `--tone <device>` (`default`, an id or part of a name) then opens that
//! device shared as the player does (stereo, at the device's nominal
//! rate), plays a 1 kHz tone at −20 dBFS for three seconds, and prints the
//! opened format, the stream's formats and the volume controls while it
//! plays.

#[cfg(target_os = "macos")]
fn main() {
    if let Err(e) = mac::run() {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("coreaudio_info is for macOS");
}

#[cfg(target_os = "macos")]
mod mac {
    use jodel_output::{find_device, mix_format, open, OpenRequest, Output};
    use objc2_core_audio::{
        kAudioDevicePropertyDeviceUID, kAudioDevicePropertyHogMode, kAudioDevicePropertyStreams,
        kAudioDevicePropertySupportsMixing, kAudioHardwarePropertyDevices,
        kAudioObjectPropertyElementMain, kAudioObjectPropertyName, kAudioObjectPropertyScopeGlobal,
        kAudioObjectPropertyScopeOutput, kAudioObjectSystemObject,
        kAudioStreamPropertyAvailablePhysicalFormats, kAudioStreamPropertyPhysicalFormat,
        kAudioStreamPropertyVirtualFormat, AudioObjectGetPropertyData,
        AudioObjectGetPropertyDataSize, AudioObjectHasProperty, AudioObjectID,
        AudioObjectIsPropertySettable, AudioObjectPropertyAddress, AudioStreamRangedDescription,
    };
    use objc2_core_audio_types::AudioStreamBasicDescription;
    use objc2_core_foundation::{CFRetained, CFString};
    use std::mem::size_of;
    use std::ptr::{null, NonNull};
    use std::time::Duration;

    const GLOBAL: u32 = kAudioObjectPropertyScopeGlobal;
    const OUTPUT: u32 = kAudioObjectPropertyScopeOutput;
    /// `kAudioDevicePropertyVolumeScalar` ('volm').
    const VOLUME_SCALAR: u32 = u32::from_be_bytes(*b"volm");
    /// `kAudioDevicePropertyVolumeDecibels` ('vold').
    const VOLUME_DB: u32 = u32::from_be_bytes(*b"vold");
    /// `kAudioDevicePropertyMute` ('mute').
    const MUTE: u32 = u32::from_be_bytes(*b"mute");

    fn addr(selector: u32, scope: u32, element: u32) -> AudioObjectPropertyAddress {
        AudioObjectPropertyAddress {
            mSelector: selector,
            mScope: scope,
            mElement: element,
        }
    }

    fn has(object: AudioObjectID, a: AudioObjectPropertyAddress) -> bool {
        // SAFETY: reads a property's presence.
        unsafe { AudioObjectHasProperty(object, NonNull::from(&a)) }
    }

    fn settable(object: AudioObjectID, a: AudioObjectPropertyAddress) -> bool {
        let mut flag: u8 = 0;
        // SAFETY: a one-byte Boolean out parameter.
        unsafe {
            AudioObjectIsPropertySettable(object, NonNull::from(&a), NonNull::from(&mut flag)) == 0
                && flag != 0
        }
    }

    fn get<T: Copy>(object: AudioObjectID, a: AudioObjectPropertyAddress) -> Option<T> {
        // SAFETY: `value` is a plain-data out buffer of the size passed.
        unsafe {
            let mut value: T = std::mem::zeroed();
            let mut size = size_of::<T>() as u32;
            let status = AudioObjectGetPropertyData(
                object,
                NonNull::from(&a),
                0,
                null(),
                NonNull::from(&mut size),
                NonNull::from(&mut value).cast(),
            );
            (status == 0).then_some(value)
        }
    }

    fn get_vec<T: Copy>(object: AudioObjectID, a: AudioObjectPropertyAddress) -> Vec<T> {
        // SAFETY: the buffer is sized from the property's reported size.
        unsafe {
            let mut size = 0u32;
            if AudioObjectGetPropertyDataSize(
                object,
                NonNull::from(&a),
                0,
                null(),
                NonNull::from(&mut size),
            ) != 0
                || (size as usize) < size_of::<T>()
            {
                return Vec::new();
            }
            let mut values: Vec<T> = (0..size as usize / size_of::<T>())
                .map(|_| std::mem::zeroed())
                .collect();
            let mut size = (values.len() * size_of::<T>()) as u32;
            if AudioObjectGetPropertyData(
                object,
                NonNull::from(&a),
                0,
                null(),
                NonNull::from(&mut size),
                NonNull::new_unchecked(values.as_mut_ptr().cast()),
            ) != 0
            {
                return Vec::new();
            }
            values.truncate(size as usize / size_of::<T>());
            values
        }
    }

    fn string(object: AudioObjectID, selector: u32) -> Option<String> {
        let p: *const CFString = get(
            object,
            addr(selector, GLOBAL, kAudioObjectPropertyElementMain),
        )?;
        // SAFETY: string properties are returned with a +1 retain count.
        let s = unsafe { CFRetained::from_raw(NonNull::new(p as *mut CFString)?) };
        Some(s.to_string())
    }

    fn output_streams(device: AudioObjectID) -> Vec<AudioObjectID> {
        get_vec(
            device,
            addr(
                kAudioDevicePropertyStreams,
                OUTPUT,
                kAudioObjectPropertyElementMain,
            ),
        )
    }

    fn devices() -> Vec<AudioObjectID> {
        get_vec::<AudioObjectID>(
            kAudioObjectSystemObject as AudioObjectID,
            addr(
                kAudioHardwarePropertyDevices,
                GLOBAL,
                kAudioObjectPropertyElementMain,
            ),
        )
        .into_iter()
        .filter(|&d| !output_streams(d).is_empty())
        .collect()
    }

    /// A description as `48000 Hz 2 ch 32 bit in 4 bytes/ch [float packed]`.
    fn describe(d: &AudioStreamBasicDescription) -> String {
        let names = [
            (1u32, "float"),
            (2, "big-endian"),
            (4, "signed-int"),
            (8, "packed"),
            (16, "aligned-high"),
            (32, "non-interleaved"),
            (64, "non-mixable"),
        ];
        let flags: Vec<&str> = names
            .iter()
            .filter(|(bit, _)| d.mFormatFlags & bit != 0)
            .map(|(_, name)| *name)
            .collect();
        let id = d.mFormatID.to_be_bytes();
        let per_channel = d
            .mBytesPerFrame
            .checked_div(d.mChannelsPerFrame)
            .unwrap_or(0);
        format!(
            "{} Hz {} ch {} bit in {} bytes/ch, format '{}', flags 0x{:x} [{}]",
            d.mSampleRate,
            d.mChannelsPerFrame,
            d.mBitsPerChannel,
            per_channel,
            String::from_utf8_lossy(&id),
            d.mFormatFlags,
            flags.join(" ")
        )
    }

    fn print_streams(device: AudioObjectID) {
        let main = kAudioObjectPropertyElementMain;
        for (i, stream) in output_streams(device).into_iter().enumerate() {
            println!("    stream {i}:");
            let physical: Option<AudioStreamBasicDescription> = get(
                stream,
                addr(kAudioStreamPropertyPhysicalFormat, GLOBAL, main),
            );
            let virtual_format: Option<AudioStreamBasicDescription> = get(
                stream,
                addr(kAudioStreamPropertyVirtualFormat, GLOBAL, main),
            );
            match physical {
                Some(d) => println!("      physical now: {}", describe(&d)),
                None => println!("      physical now: (unreadable)"),
            }
            match virtual_format {
                Some(d) => println!("      virtual now:  {}", describe(&d)),
                None => println!("      virtual now:  (unreadable)"),
            }
            let mut seen = Vec::new();
            for r in get_vec::<AudioStreamRangedDescription>(
                stream,
                addr(kAudioStreamPropertyAvailablePhysicalFormats, GLOBAL, main),
            ) {
                let mut d = r.mFormat;
                d.mSampleRate = 0.0;
                let line = format!(
                    "{} ({}–{} Hz)",
                    describe(&d),
                    r.mSampleRateRange.mMinimum,
                    r.mSampleRateRange.mMaximum
                );
                if !seen.contains(&line) {
                    println!("      available:    {line}");
                    seen.push(line);
                }
            }
        }
    }

    fn print_controls(device: AudioObjectID) {
        let channels: u32 = output_streams(device)
            .into_iter()
            .filter_map(|s| {
                get::<AudioStreamBasicDescription>(
                    s,
                    addr(
                        kAudioStreamPropertyVirtualFormat,
                        GLOBAL,
                        kAudioObjectPropertyElementMain,
                    ),
                )
            })
            .map(|d| d.mChannelsPerFrame)
            .sum();
        let mut any = false;
        for element in 0..=channels {
            let label = if element == 0 {
                "master".to_owned()
            } else {
                format!("channel {element}")
            };
            let mut parts = Vec::new();
            let scalar = addr(VOLUME_SCALAR, OUTPUT, element);
            if has(device, scalar) {
                let v: Option<f32> = get(device, scalar);
                parts.push(format!(
                    "volume {} ({})",
                    v.map_or("?".into(), |v| format!("{v:.3}")),
                    if settable(device, scalar) {
                        "settable"
                    } else {
                        "fixed"
                    }
                ));
            }
            let db = addr(VOLUME_DB, OUTPUT, element);
            if has(device, db) {
                let v: Option<f32> = get(device, db);
                parts.push(format!(
                    "{} dB",
                    v.map_or("?".into(), |v| format!("{v:.1}"))
                ));
            }
            let mute = addr(MUTE, OUTPUT, element);
            if has(device, mute) {
                let v: Option<u32> = get(device, mute);
                parts.push(format!(
                    "mute {} ({})",
                    v.map_or("?".into(), |v| (v != 0).to_string()),
                    if settable(device, mute) {
                        "settable"
                    } else {
                        "fixed"
                    }
                ));
            }
            if !parts.is_empty() {
                any = true;
                println!("    {label}: {}", parts.join(", "));
            }
        }
        if !any {
            println!("    no volume or mute controls on the output");
        }
    }

    fn print_device(device: AudioObjectID) {
        let main = kAudioObjectPropertyElementMain;
        let name = string(device, kAudioObjectPropertyName).unwrap_or_default();
        let uid = string(device, kAudioDevicePropertyDeviceUID).unwrap_or_default();
        println!("{name} ({uid})");
        let hog: Option<i32> = get(device, addr(kAudioDevicePropertyHogMode, GLOBAL, main));
        let mixing_address = addr(kAudioDevicePropertySupportsMixing, GLOBAL, main);
        let mixing: Option<u32> = get(device, mixing_address);
        println!(
            "    hog mode owner: {}, mixing: {} ({})",
            match hog {
                Some(-1) => "none".to_owned(),
                Some(pid) if pid == std::process::id() as i32 => format!("{pid} (this process)"),
                Some(pid) => pid.to_string(),
                None => "?".into(),
            },
            mixing.map_or("?".into(), |m| (m != 0).to_string()),
            if settable(device, mixing_address) {
                "settable"
            } else {
                "fixed"
            }
        );
        print_streams(device);
        print_controls(device);
    }

    fn find(uid: &str) -> Option<AudioObjectID> {
        devices()
            .into_iter()
            .find(|&d| string(d, kAudioDevicePropertyDeviceUID).as_deref() == Some(uid))
    }

    pub fn run() -> Result<(), Box<dyn std::error::Error>> {
        let args: Vec<String> = std::env::args().skip(1).collect();
        for device in devices() {
            print_device(device);
            println!();
        }
        let Some(at) = args.iter().position(|a| a == "--tone") else {
            return Ok(());
        };
        let query = args.get(at + 1).map_or("default", String::as_str);
        let device = find_device(query)?;
        let format = mix_format(&device.id)?;
        let rate = format.rate;
        let mut out = open(&device.id, &OpenRequest::new(rate, format.channels))?;
        let info = out.info().clone();
        println!(
            "tone on {}: opened {} via {}",
            device.name, info.format, info.backend
        );
        if let Some(object) = find(&device.id) {
            println!("while playing:");
            print_device(object);
        }
        // 1 kHz at −20 dBFS (a tenth of full scale), three seconds.
        let amplitude = 0.1 * 8_388_607.0;
        let total = 3 * rate as usize;
        let mut frame = 0usize;
        let chunk = 480usize;
        let mut buf = vec![0i32; 2 * chunk];
        while frame < total {
            let n = chunk.min(total - frame);
            for i in 0..n {
                let t = (frame + i) as f64 / f64::from(rate);
                let v =
                    (amplitude * (2.0 * std::f64::consts::PI * 1_000.0 * t).sin()).round() as i32;
                buf[2 * i] = v;
                buf[2 * i + 1] = v;
            }
            let mut done = 0;
            while done < n {
                done += out.write(&buf[2 * done..2 * n], Some(Duration::from_secs(2)))?;
            }
            frame += n;
        }
        out.drain(Some(Duration::from_secs(5)))?;
        println!("played 3 s of 1 kHz at -20 dBFS; stats: {:?}", out.stats());
        out.close()?;
        Ok(())
    }
}
