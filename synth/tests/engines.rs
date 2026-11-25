//! Each engine on its own: every preset sounds, stays bounded and is the
//! same every time; the pitched ones are in tune at every rate; the
//! filters and the timbre controls move the brightness the right way; a
//! retrigger keeps what it keeps.

mod common;

use common::measure::{centroid, cents, peak, pitch, rms};
use common::{held, on, solo};
use jodel_synth::dsp::{self, Rate};
use jodel_synth::engines::{
    analog, drum, fm, formant, lofi, modal, noise, phase, string, supersaw, AnyParams, Engines,
};
use jodel_synth::{Engine, NoteOn, VoiceMod, BLOCK};

fn presets() -> Vec<(&'static str, AnyParams)> {
    let mut all: Vec<(&str, AnyParams)> = Vec::new();
    all.extend(noise::PRESETS.map(|(n, p)| (n, p.into())));
    all.extend(analog::PRESETS.map(|(n, p)| (n, p.into())));
    all.extend(lofi::PRESETS.map(|(n, p)| (n, p.into())));
    all.extend(phase::PRESETS.map(|(n, p)| (n, p.into())));
    all.extend(modal::PRESETS.map(|(n, p)| (n, p.into())));
    all.extend(string::PRESETS.map(|(n, p)| (n, p.into())));
    all.extend(formant::PRESETS.map(|(n, p)| (n, p.into())));
    all.extend(fm::PRESETS.map(|(n, p)| (n, p.into())));
    all.extend(supersaw::PRESETS.map(|(n, p)| (n, p.into())));
    all.extend(drum::PRESETS.map(|(n, p)| (n, p.into())));
    all.extend(drum::SLAM.map(|(n, p)| (n, p.into())));
    all
}

/// `p` through the dispatch, `seconds` on `note`.
fn any(p: &AnyParams, note: f32, seconds: f32, seed: u32) -> Vec<f32> {
    let e = Engines::new(Rate::new(48_000));
    let mut v = e.voice(p);
    e.note_on(p, &mut v, &on(note, seed));
    let mut out = vec![0.0; (seconds * 48_000.0) as usize / BLOCK * BLOCK];
    for block in out.chunks_exact_mut(BLOCK) {
        let amp = e.amp(p, &mut v, 1.0);
        let mut b = [0.0; BLOCK];
        e.render(p, &mut v, &held(note, 0.0), &mut b);
        for (o, x) in block.iter_mut().zip(b) {
            *o = x * amp;
        }
    }
    e.free(&mut v);
    out
}

#[test]
fn every_preset_sounds_bounded_and_the_same_every_time() {
    for (name, p) in presets() {
        let a = any(&p, 60.0, 0.5, 7);
        assert!(a.iter().all(|x| x.is_finite()), "{name}");
        let level = rms(&a[..4_800]);
        assert!(level > 1e-3, "{name}: {level}");
        assert!(peak(&a) <= 2.0, "{name}: {}", peak(&a));
        assert_eq!(a, any(&p, 60.0, 0.5, 7), "{name}");
    }
}

#[test]
fn noise_is_seeded() {
    for p in [noise::WHOOSH, noise::CRACKLE] {
        let p = AnyParams::from(p);
        assert_ne!(any(&p, 60.0, 0.2, 1), any(&p, 60.0, 0.2, 2));
    }
}

fn in_tune<E: Engine>(e: &E, p: &E::Params, rate: u32, note: f32, cents_off: f32) -> f32
where
    E::Params: std::fmt::Debug,
{
    let x = solo(e, p, rate, note, 0.3, 1, 0.0);
    let got = pitch(&x[x.len() / 3..], rate as f32);
    let c = cents(got, dsp::hz(note + cents_off / 100.0));
    assert!(
        c.abs() < 5.0,
        "{p:?} at {rate} Hz, note {note}: {got} Hz, {c} cents"
    );
    c
}

#[test]
fn the_pitched_engines_are_in_tune_at_every_rate() {
    for rate in [44_100, 48_000, 96_000] {
        let r = Rate::new(rate);
        for note in [45.0, 57.0, 69.0, 76.0] {
            let a = analog::Analog::new(r);
            in_tune(&a, &analog::SINE, rate, note, 0.0);
            in_tune(&a, &analog::Params::default(), rate, note, 0.0);
            in_tune(&a, &analog::ZAP, rate, note, 0.0);
            let l = lofi::Lofi::new(r);
            in_tune(&l, &lofi::COIN, rate, note, 0.0);
            in_tune(&l, &lofi::BASS, rate, note, 0.0);
            let vowel = lofi::Params {
                wave: lofi::Wave::Table(lofi::Table::VoxA),
                ..lofi::Params::default()
            };
            in_tune(&l, &vowel, rate, note, 0.0);
            let ph = phase::Phase::new(r);
            in_tune(&ph, &phase::BRASS, rate, note, 3.5);
            in_tune(&ph, &phase::Params::default(), rate, note, 0.0);
            // The register's steps fall unevenly on the samples above a few
            // hundred Hz: the period holds on average, not sample by sample.
            if note <= 57.0 {
                in_tune(&noise::Noise::new(r), &noise::METAL, rate, note, 0.0);
            }
        }
    }
}

#[test]
fn the_filters_and_the_timbre_brighten_the_right_way() {
    let rate = Rate::new(48_000);
    let bright = |x: Vec<f32>| centroid(&x, 48_000.0);
    let a = analog::Analog::new(rate);
    let dark = analog::Params {
        cutoff: 300.0,
        ..analog::Params::default()
    };
    let open = analog::Params {
        cutoff: 3_000.0,
        ..dark
    };
    let (c_dark, c_open) = (
        bright(solo(&a, &dark, 48_000, 48.0, 0.2, 1, 0.0)),
        bright(solo(&a, &open, 48_000, 48.0, 0.2, 1, 0.0)),
    );
    assert!(c_open > 2.0 * c_dark, "{c_dark} {c_open}");
    let moved = bright(solo(&a, &dark, 48_000, 48.0, 0.2, 1, 3.0));
    assert!(moved > 2.0 * c_dark, "{c_dark} {moved}");

    let n = noise::Noise::new(rate);
    let low = noise::Params {
        filter: noise::Filter::BandPass,
        cutoff: 500.0,
        res: 0.85,
        ..noise::Params::default()
    };
    let high = noise::Params {
        cutoff: 4_000.0,
        ..low
    };
    let (c_low, c_high) = (
        bright(solo(&n, &low, 48_000, 60.0, 0.2, 1, 0.0)),
        bright(solo(&n, &high, 48_000, 60.0, 0.2, 1, 0.0)),
    );
    assert!(c_high > 1.5 * c_low, "{c_low} {c_high}");
    // The make-up keeps a band's level near a low-pass's.
    let lp = rms(&solo(
        &n,
        &noise::Params::default(),
        48_000,
        60.0,
        0.2,
        1,
        0.0,
    ));
    let bp = rms(&solo(&n, &high, 48_000, 60.0, 0.2, 1, 0.0));
    assert!((bp / lp) > 0.3 && (bp / lp) < 3.0, "{lp} {bp}");

    let l = lofi::Lofi::new(rate);
    let dull = lofi::Params {
        tone: 500.0,
        ..lofi::Params::default()
    };
    let clear = lofi::Params {
        tone: 6_000.0,
        ..dull
    };
    assert!(
        bright(solo(&l, &clear, 48_000, 60.0, 0.2, 1, 0.0))
            > 1.5 * bright(solo(&l, &dull, 48_000, 60.0, 0.2, 1, 0.0))
    );

    let ph = phase::Phase::new(rate);
    let sine = phase::Params::default();
    let bent = phase::Params { dcw: 0.9, ..sine };
    let (c_sine, c_bent) = (
        bright(solo(&ph, &sine, 48_000, 57.0, 0.2, 1, 0.0)),
        bright(solo(&ph, &bent, 48_000, 57.0, 0.2, 1, 0.0)),
    );
    assert!(c_sine < 300.0 && c_bent > 2.0 * c_sine, "{c_sine} {c_bent}");
}

#[test]
fn the_colour_darkens_noise() {
    let n = noise::Noise::new(Rate::new(48_000));
    let c = |color: f32| {
        let p = noise::Params {
            color,
            cutoff: 20_000.0,
            ..noise::Params::default()
        };
        centroid(&solo(&n, &p, 48_000, 60.0, 0.2, 1, 0.0), 48_000.0)
    };
    let (white, pink, brown) = (c(0.0), c(0.5), c(1.0));
    assert!(white > pink && pink > brown, "{white} {pink} {brown}");
}

#[test]
fn the_chip_envelope_steps_down_and_ends_the_voice() {
    let l = lofi::Lofi::new(Rate::new(48_000));
    let p = lofi::BLIP;
    let mut v = lofi::Voice::default();
    l.note_on(&p, &mut v, &NoteOn::default());
    // Fifteen steps, one every two frames of 240 a second: 125 ms.
    let mut blocks = 0;
    let mut last = 1.0;
    while !l.done(&v) {
        let a = l.amp(&p, &mut v, 1.0);
        assert!(a <= last);
        last = a;
        blocks += 1;
        assert!(blocks < 10_000);
    }
    let seconds = blocks as f32 * BLOCK as f32 / 48_000.0;
    assert!((seconds - 0.125).abs() < 0.01, "{seconds}");
    // Without it, the ADSR's.
    assert_eq!(l.amp(&lofi::COIN, &mut v, 0.5), 0.5);
}

#[test]
fn a_retrigger_keeps_the_filter_and_the_phases() {
    // Retriggered, a ringing ANALOG voice sounds on exactly as if it had not
    // been: its phases and its filter both carry over. Started afresh (the
    // phases and the filter at rest) it does not.
    let a = analog::Analog::new(Rate::new(48_000));
    let p = analog::Params {
        res: 0.9,
        ..analog::BRASS
    };
    let m = held(60.0, 0.0);
    let mut v = analog::Voice::default();
    let n = on(60.0, 3);
    a.note_on(&p, &mut v, &n);
    let mut out = [0.0; BLOCK];
    for _ in 0..100 {
        a.render(&p, &mut v, &m, &mut out);
    }
    let mut untouched = v;
    a.note_on(&p, &mut v, &n);
    let (mut kept, mut plain) = ([0.0; BLOCK], [0.0; BLOCK]);
    a.render(&p, &mut v, &m, &mut kept);
    a.render(&p, &mut untouched, &m, &mut plain);
    assert_eq!(kept, plain);
    let mut fresh = analog::Voice::default();
    a.note_on(&p, &mut fresh, &n);
    let mut first = [0.0; BLOCK];
    a.render(&p, &mut fresh, &m, &mut first);
    assert_ne!(kept, first);
}

#[test]
fn the_shape_moves_what_each_engine_says() {
    // Every engine's shape target is named, and the shape changes the
    // sound of each that takes one (the pulse's width for LOFI).
    for target in [
        noise::SHAPE,
        analog::SHAPE,
        lofi::SHAPE,
        phase::SHAPE,
        modal::SHAPE,
        string::SHAPE,
        formant::SHAPE,
        fm::SHAPE,
        supersaw::SHAPE,
        drum::SHAPE,
    ] {
        assert!(!target.is_empty());
    }
    let l = lofi::Lofi::new(Rate::new(48_000));
    let render = |shape: f32| {
        let mut v = lofi::Voice::default();
        l.note_on(&lofi::COIN, &mut v, &NoteOn::default());
        let m = VoiceMod {
            shape,
            ..held(60.0, 0.0)
        };
        let mut out = vec![0.0; 4 * BLOCK];
        for b in out.chunks_exact_mut(BLOCK) {
            l.render(&lofi::COIN, &mut v, &m, b);
        }
        out
    };
    assert_ne!(render(0.0), render(0.5));
}
