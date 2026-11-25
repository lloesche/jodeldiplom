//! FileOutput and NullOutput: the handle logic every backend shares
//! (packing, ring, start threshold, underrun, pause, flush, drain, close),
//! on a simulated device.

use jodel_output::pack::{pack, unpack};
use jodel_output::{
    Capabilities, Clock, DeviceOutput, Error, FileOutput, NullOutput, OpenRequest, Output,
    PackError, PackMode, SampleFormat, SimClock, StreamFormat,
};
use std::time::{Duration, Instant};

fn ramp(frames: usize, channels: usize, start: i32) -> Vec<i32> {
    (0..frames * channels)
        .map(|i| (start + i as i32 * 4099) % (1 << 23))
        .collect()
}

fn packed(samples: &[i32], format: SampleFormat) -> Vec<u8> {
    let mut out = vec![0; samples.len() * format.bytes()];
    pack(samples, format, PackMode::Exact, &mut out).unwrap();
    out
}

/// A manual-clock output at 1 kHz stereo: easy frame arithmetic. Ring
/// 100 frames, prefill 20 frames.
fn manual(format: SampleFormat) -> (FileOutput, jodel_output::MemorySink) {
    let request = OpenRequest::new(1000, 2)
        .formats(&[format])
        .buffer(Duration::from_millis(100))
        .prefill(Duration::from_millis(20))
        .period(Duration::from_millis(10));
    let sink = jodel_output::MemorySink::default();
    let caps: Capabilities = [StreamFormat::new(1000, 2, format)].into_iter().collect();
    let out = FileOutput::open(sink.clone(), &caps, &request, SimClock::Manual).unwrap();
    (out, sink)
}

#[test]
fn instant_clock_captures_exactly_what_was_written() {
    for format in SampleFormat::ALL.into_iter().filter(|f| f.holds_24_bits()) {
        let (mut out, sink) =
            FileOutput::to_memory(StreamFormat::new(96_000, 2, format), SimClock::Instant);
        let input = ramp(100_000, 2, -(1 << 23));
        for chunk in input.chunks(2 * 777) {
            assert_eq!(out.write(chunk, None).unwrap(), chunk.len() / 2);
            assert_eq!(out.position().unwrap(), out.written());
        }
        out.close().unwrap();
        assert_eq!(sink.bytes(), packed(&input, format), "{format}");
        assert_eq!(unpack(&sink.bytes(), format), input);
        assert_eq!(out.stats(), Default::default());
        assert_eq!(out.written(), 100_000);
    }
}

#[test]
fn the_device_starts_at_the_prefill() {
    let (mut out, sink) = manual(SampleFormat::S24In32);
    out.write(&ramp(10, 2, 0), None).unwrap();
    // Below the 20-frame prefill: not started, time passes, nothing plays.
    assert_eq!(out.advance(50).unwrap(), 0);
    assert_eq!(out.position().unwrap(), 0);
    assert!(sink.is_empty());
    out.write(&ramp(10, 2, 20), None).unwrap();
    assert_eq!(out.advance(5).unwrap(), 5);
    assert_eq!(out.position().unwrap(), 5);
    assert_eq!(out.latency(), Duration::from_millis(15));
    assert_eq!(sink.len(), 5 * 8);
}

#[test]
fn a_dry_ring_pads_with_silence_and_counts_one_underrun() {
    let (mut out, sink) = manual(SampleFormat::S16);
    let input: Vec<i32> = (0..60).map(|i| (i + 1) << 8).collect();
    out.write(&input, None).unwrap();
    // 30 frames queued; the device plays 50: 30 stream, then 20 of silence
    // in two periods, one underrun.
    assert_eq!(out.advance(40).unwrap(), 30);
    assert_eq!(out.advance(10).unwrap(), 0);
    let stats = out.stats();
    assert_eq!(
        (stats.underruns, stats.underrun_frames, stats.truncated),
        (1, 20, 0)
    );
    assert_eq!(out.position().unwrap(), 30);
    let bytes = sink.bytes();
    assert_eq!(bytes.len(), 50 * 4);
    assert_eq!(bytes[..120], packed(&input, SampleFormat::S16)[..]);
    assert!(bytes[120..].iter().all(|&b| b == 0));
    // More data, then another dry spell: a second underrun.
    out.write(&input[..20], None).unwrap();
    out.advance(15).unwrap();
    assert_eq!(out.stats().underruns, 2);
    assert_eq!(out.position().unwrap(), 40);
}

#[test]
fn a_full_ring_blocks_the_writer_until_the_device_plays() {
    let (mut out, _sink) = manual(SampleFormat::S24Packed);
    let input = ramp(150, 2, 0);
    let t = Instant::now();
    // Ring of 100 frames: 100 fit, then the write times out.
    assert_eq!(
        out.write(&input, Some(Duration::from_millis(30))).unwrap(),
        100
    );
    assert!(t.elapsed() >= Duration::from_millis(25));
    assert_eq!(out.advance(60).unwrap(), 60);
    assert_eq!(out.write(&input[200..], Some(Duration::ZERO)).unwrap(), 50);
    assert_eq!(out.written(), 150);
}

#[test]
fn pause_holds_the_position_and_returns_early_from_a_full_write() {
    let (mut out, sink) = manual(SampleFormat::S24In32Low);
    out.write(&ramp(50, 2, 0), None).unwrap();
    out.advance(10).unwrap();
    out.pause().unwrap();
    assert!(out.is_paused());
    assert_eq!(out.advance(30).unwrap(), 0);
    assert_eq!(out.position().unwrap(), 10);
    // While paused a write that cannot fit returns at once with what fitted.
    let t = Instant::now();
    assert_eq!(out.write(&ramp(80, 2, 0), None).unwrap(), 60);
    assert!(t.elapsed() < Duration::from_millis(500));
    out.resume().unwrap();
    assert_eq!(out.advance(30).unwrap(), 30);
    assert_eq!(out.position().unwrap(), 40);
    assert_eq!(out.stats().underruns, 0);
    assert_eq!(sink.len(), 40 * 8);
}

#[test]
fn flush_discards_and_the_next_write_plays_from_there() {
    let (mut out, sink) = manual(SampleFormat::S32);
    let first = ramp(60, 2, 1);
    out.write(&first, None).unwrap();
    out.advance(10).unwrap();
    out.flush().unwrap();
    assert_eq!(out.position().unwrap(), 60);
    assert_eq!(out.written(), 60);
    // Flushed: stopped until the prefill is queued again.
    assert_eq!(out.advance(10).unwrap(), 0);
    let second = ramp(30, 2, 7);
    out.write(&second, None).unwrap();
    assert_eq!(out.advance(30).unwrap(), 30);
    assert_eq!(out.position().unwrap(), 90);
    let bytes = sink.bytes();
    assert_eq!(bytes[..80], packed(&first[..20], SampleFormat::S32)[..]);
    assert_eq!(bytes[80..], packed(&second, SampleFormat::S32)[..]);
    assert_eq!(out.stats().underruns, 0);
}

#[test]
fn resume_restarts_a_started_device_even_below_the_prefill() {
    let (mut out, _sink) = manual(SampleFormat::S24In32);
    out.write(&ramp(30, 2, 0), None).unwrap();
    assert_eq!(out.advance(30).unwrap(), 30);
    out.pause().unwrap();
    out.write(&ramp(5, 2, 0), None).unwrap();
    out.resume().unwrap();
    assert_eq!(out.advance(5).unwrap(), 5);
    assert_eq!(out.position().unwrap(), 35);
}

#[test]
fn flush_while_paused_stays_paused() {
    let (mut out, _sink) = manual(SampleFormat::S24In32);
    out.write(&ramp(40, 2, 0), None).unwrap();
    out.pause().unwrap();
    out.flush().unwrap();
    assert!(out.is_paused());
    out.write(&ramp(40, 2, 0), None).unwrap();
    assert_eq!(out.advance(10).unwrap(), 0);
    out.resume().unwrap();
    assert_eq!(out.advance(10).unwrap(), 10);
}

#[test]
fn exact_mode_rejects_a_write_without_queuing_any_of_it() {
    let request = OpenRequest::new(48_000, 2)
        .formats(&[SampleFormat::S24Packed])
        .pack_mode(PackMode::Exact);
    let caps: Capabilities = [StreamFormat::new(48_000, 2, SampleFormat::S24Packed)]
        .into_iter()
        .collect();
    let (sink, bytes) = {
        let s = jodel_output::MemorySink::default();
        (s.clone(), s)
    };
    let mut out = FileOutput::open(sink, &caps, &request, SimClock::Instant).unwrap();
    out.write(&[1, 2], None).unwrap();
    let bad = [5, 6, 7, 1 << 23];
    assert_eq!(
        out.write(&bad, None),
        Err(Error::Pack(PackError::OutOfRange {
            index: 3,
            value: 1 << 23
        }))
    );
    assert_eq!(out.written(), 1);
    assert_eq!(bytes.len(), 6);
    assert_eq!(out.stats().clipped, 0);
}

#[test]
fn saturate_mode_clamps_and_counts() {
    let (mut out, sink) = FileOutput::to_memory(
        StreamFormat::new(48_000, 2, SampleFormat::S24In32),
        SimClock::Instant,
    );
    out.write(&[1 << 23, -(1 << 23) - 5, 3, i32::MIN], None)
        .unwrap();
    assert_eq!(
        unpack(&sink.bytes(), SampleFormat::S24In32),
        [(1 << 23) - 1, -(1 << 23), 3, -(1 << 23)]
    );
    assert_eq!(out.stats().clipped, 3);
    let (mut out16, _) = FileOutput::to_memory(
        StreamFormat::new(48_000, 2, SampleFormat::S16),
        SimClock::Instant,
    );
    out16.write(&[0x101, 0x200], None).unwrap();
    assert_eq!(out16.stats().truncated, 1);
}

#[test]
fn negotiation_follows_the_preference_and_refuses_other_rates() {
    let mut caps = Capabilities::new();
    for f in [SampleFormat::S24Packed, SampleFormat::S16] {
        caps.insert(StreamFormat::new(96_000, 2, f));
    }
    let out = FileOutput::open(
        std::io::sink(),
        &caps,
        &OpenRequest::new(96_000, 2),
        SimClock::Instant,
    )
    .unwrap();
    assert_eq!(
        out.format(),
        StreamFormat::new(96_000, 2, SampleFormat::S24Packed)
    );
    for request in [OpenRequest::new(88_200, 2), OpenRequest::new(96_000, 1)] {
        assert!(matches!(
            FileOutput::open(std::io::sink(), &caps, &request, SimClock::Instant),
            Err(Error::FormatNotAccepted { .. })
        ));
    }
    let none = OpenRequest::new(96_000, 2).formats(&[]);
    assert!(matches!(
        FileOutput::open(std::io::sink(), &caps, &none, SimClock::Instant),
        Err(Error::InvalidArgument(_))
    ));
}

#[test]
fn realtime_clock_paces_and_drains() {
    let format = StreamFormat::new(8000, 2, SampleFormat::S24In32);
    let (mut out, sink) = FileOutput::to_memory(format, SimClock::Realtime);
    let input = ramp(1600, 2, 9); // 200 ms
    let clock: Clock = out.clock();
    let watcher = std::thread::spawn(move || {
        let mut last = 0;
        for _ in 0..20 {
            let p = clock.position();
            assert!(p >= last, "position went back");
            last = p;
            std::thread::sleep(Duration::from_millis(5));
        }
        last
    });
    let t = Instant::now();
    out.write(&input, None).unwrap();
    assert!(out.drain(Some(Duration::from_secs(5))).unwrap());
    let elapsed = t.elapsed();
    assert!(elapsed >= Duration::from_millis(150), "{elapsed:?}");
    assert_eq!(out.position().unwrap(), 1600);
    assert!(watcher.join().unwrap() <= 1600);
    // Draining leaves the device playing silence, which is not an underrun.
    std::thread::sleep(Duration::from_millis(50));
    assert_eq!(out.stats().underruns, 0);
    out.close().unwrap();
    let bytes = sink.bytes();
    assert_eq!(bytes[..1600 * 8], packed(&input, SampleFormat::S24In32)[..]);
    assert!(bytes[1600 * 8..].iter().all(|&b| b == 0));
}

/// The paced clock: what is written plays no faster than real time, a
/// writer that stops for a while is waited for (no silence, no underrun)
/// and its frames are not then played in a burst, and the device is not
/// one in real time.
#[test]
fn paced_clock_plays_only_what_is_written_no_faster_than_real_time() {
    let format = StreamFormat::new(8000, 2, SampleFormat::S24In32);
    let (mut out, sink) = FileOutput::to_memory(format, SimClock::Paced);
    assert!(!out.info().realtime);
    let first = ramp(800, 2, 9); // 100 ms
    let t = Instant::now();
    out.write(&first, None).unwrap();
    assert!(out.drain(Some(Duration::from_secs(5))).unwrap());
    let took = t.elapsed();
    assert!(took >= Duration::from_millis(80), "{took:?}");
    // The writer away for 100 ms: nothing is played meanwhile.
    std::thread::sleep(Duration::from_millis(100));
    assert_eq!(out.position().unwrap(), 800);
    assert_eq!(sink.len(), 800 * 8);
    // Then on, at the pace of a device again.
    let second = ramp(800, 2, 7);
    let t = Instant::now();
    out.write(&second, None).unwrap();
    assert!(out.drain(Some(Duration::from_secs(5))).unwrap());
    let took = t.elapsed();
    assert!(took >= Duration::from_millis(70), "{took:?}");
    assert_eq!(out.stats().underruns, 0);
    out.close().unwrap();
    let mut both = packed(&first, SampleFormat::S24In32);
    both.extend(packed(&second, SampleFormat::S24In32));
    assert!(sink.bytes() == both);
}

#[test]
fn closed_outputs_refuse_further_use() {
    let mut out = NullOutput::new(
        StreamFormat::new(48_000, 2, SampleFormat::S32),
        SimClock::Instant,
    );
    out.write(&[0; 96], None).unwrap();
    assert_eq!(out.position().unwrap(), 48);
    out.close().unwrap();
    assert_eq!(out.write(&[0; 2], None), Err(Error::Closed));
    assert_eq!(out.pause(), Err(Error::Closed));
    assert_eq!(out.close(), Err(Error::Closed));
    assert_eq!(out.position().unwrap(), 48);
}

#[test]
fn partial_frames_are_rejected() {
    let mut out = NullOutput::new(
        StreamFormat::new(48_000, 2, SampleFormat::S32),
        SimClock::Instant,
    );
    assert!(matches!(
        out.write(&[0; 3], None),
        Err(Error::InvalidArgument(_))
    ));
    assert_eq!(out.written(), 0);
}

#[test]
fn advance_needs_the_manual_clock() {
    let mut out = NullOutput::new(
        StreamFormat::new(48_000, 2, SampleFormat::S32),
        SimClock::Instant,
    );
    assert!(matches!(out.advance(1), Err(Error::InvalidArgument(_))));
}

#[test]
fn outputs_can_move_to_the_engine_thread() {
    fn send<T: Send>() {}
    fn send_sync<T: Send + Sync>() {}
    send::<DeviceOutput>();
    send::<FileOutput>();
    send::<Box<dyn Output>>();
    send_sync::<Clock>();
    let mut out: Box<dyn Output> = Box::new(NullOutput::new(
        StreamFormat::new(44_100, 2, SampleFormat::S16),
        SimClock::Instant,
    ));
    std::thread::spawn(move || {
        out.write(&[0; 20], None).unwrap();
        out.written()
    })
    .join()
    .map(|w| assert_eq!(w, 10))
    .unwrap();
}
