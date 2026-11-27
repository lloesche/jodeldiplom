use super::*;
use crate::math::{v3, V3};
use crate::render::Image;

/// A picture to measure on: a sky, a sun, a dark ground, a figure.
pub(crate) fn fixture(w: usize, h: usize) -> Image {
    let mut px = Vec::with_capacity(w * h);
    for y in 0..h {
        for x in 0..w {
            let (u, v) = ((x as f32 + 0.5) / w as f32, (y as f32 + 0.5) / h as f32);
            let sky = v3(0.25 + 0.4 * v, 0.35 + 0.3 * v, 0.7 - 0.2 * v);
            let ground = v3(0.18, 0.14, 0.1) * (1.0 + 0.3 * (u * 20.0).sin());
            let mut c = if v > 0.62 { ground } else { sky };
            let sun = ((u - 0.7).powi(2) + (v - 0.3).powi(2)).sqrt();
            if sun < 0.08 {
                c = v3(1.0, 0.92, 0.7);
            }
            if (u - 0.35).abs() < 0.05 && v > 0.35 && v < 0.8 {
                c = v3(0.6, 0.15, 0.2);
            }
            px.push(c);
        }
    }
    Image { w, h, px }
}

/// A frame of `c` over `img` at `t` of the made-up song.
pub(crate) fn drawn(c: &Checked, img: &Image, t: f64, band: Option<(f32, f32)>) -> Image {
    let moment = Moment {
        size: (img.w as u32, img.h as u32),
        t,
        seed: seed(&c.id, "a song"),
        dt: 1.0 / 30.0,
        forget: false,
        band,
        palette: crate::MEASURING_PALETTE,
    };
    let signals = demo::signals_at(t);
    render::draw(&c.frame(&moment, &signals, None), img, None)
}

fn mean(img: &Image) -> V3 {
    img.px.iter().fold(math::s3(0.0), |a, &b| a + b) / img.px.len() as f32
}

/// Every pass named apart and found by its name, its knobs named apart
/// (eight at most, not `mix`), each default within its range, its natural
/// drivers on its own knobs, what it pairs with a pass or a preset.
#[test]
fn every_pass_is_whole() {
    let mut names = std::collections::HashSet::new();
    for p in passes::PASSES {
        assert!(names.insert(p.name), "{}", p.name);
        assert_eq!(passes::named(p.name), Some(*p));
        assert!(p.knobs.len() <= 8, "{}", p.name);
        assert!(!p.about.is_empty() && p.wgsl.contains(&format!("fn look_{}(", p.name)));
        assert!(p.cost > 0.0 && p.cost <= guard::BUDGET, "{}", p.name);
        assert!((0.0..=1.0).contains(&p.relief));
        for (i, k) in p.knobs.iter().enumerate() {
            assert!(
                k.lo < k.hi && k.default >= k.lo && k.default <= k.hi,
                "{}.{}",
                p.name,
                k.name
            );
            assert!(k.name != "mix" && !k.about.is_empty());
            assert!(
                p.knobs[i + 1..].iter().all(|o| o.name != k.name),
                "{}.{}",
                p.name,
                k.name
            );
            assert!(
                k.swing >= 0.0 && k.swing <= k.hi - k.lo,
                "{}.{}",
                p.name,
                k.name
            );
            assert!(
                (0.0..=guard::MIX_LIGHT).contains(&k.light),
                "{}.{}",
                p.name,
                k.name
            );
        }
        for n in p.natural {
            let (_, k) = p
                .knob(n.knob)
                .unwrap_or_else(|| panic!("{}.{}", p.name, n.knob));
            assert!(
                k.swing > 0.0 && n.amount.abs() <= 1.0,
                "{}.{}",
                p.name,
                n.knob
            );
        }
        for &pair in p.pairs {
            assert!(
                passes::named(pair).is_some() || presets::named(pair).is_some(),
                "{}: {pair}",
                p.name
            );
        }
    }
}

/// Whatever a look asks is held: knobs to their ranges, a hue round the
/// wheel, a count to a whole number, a mix to 0 to 1, the music's amount
/// to its swing; what is not there is left out and said.
#[test]
fn every_number_is_held_to_its_range() {
    let look = Look::from_json(
        r#"{"id": "wild", "layers": [
            {"pass": "grain", "knobs": {"amount": 7, "size": -3, "nonsense": 1},
             "drive": [{"knob": "amount", "by": "kick", "amount": 40}], "mix": 3},
            {"pass": "duotone", "knobs": {"dark": 1.25}},
            {"pass": "posterize_ink", "knobs": {"steps": 3.6}},
            {"pass": "no_such_pass"}
        ]}"#,
    )
    .expect("a look");
    let c = look.check();
    assert_eq!(c.layers.len(), 3);
    let grain = &c.layers[0];
    assert_eq!(grain.knobs[0], 1.0);
    assert_eq!(grain.knobs[1], 0.6);
    assert_eq!(grain.mix, 1.0);
    let d = grain.drives[0];
    assert!(d.amount > 0.0 && d.amount <= passes::film::GRAIN.knobs[0].swing);
    assert!(
        (c.layers[1].knobs[0] - 0.25).abs() < 1e-6,
        "the hue goes round"
    );
    assert_eq!(c.layers[2].knobs[0], 4.0);
    assert!(c.notes.iter().any(|n| n.contains("no_such_pass")));
    assert!(c.notes.iter().any(|n| n.contains("nonsense")));
    // Moved by the music, still within the range.
    let mut s = Signals::QUIET;
    for d in Driver::ALL {
        s.set(d, 1.0);
    }
    let m = Moment {
        size: (64, 36),
        t: 1.0,
        seed: 1,
        dt: 0.1,
        forget: false,
        band: None,
        palette: [[0.5; 3]; 3],
    };
    for (pass, ctx) in c.frame(&m, &s, None) {
        for (i, k) in pass.knobs.iter().enumerate() {
            assert!(
                ctx.k[i] >= k.lo && ctx.k[i] <= k.hi,
                "{}.{}",
                pass.name,
                k.name
            );
        }
        assert!((0.0..=1.0).contains(&ctx.mix[0]));
    }
}

/// A look over the budget drops its dearest optional pass (the later of
/// equals), the same one every time, until it is within; one kept stays.
#[test]
fn over_the_budget_the_dearest_optional_pass_goes() {
    let layer = |p: &str, keep: bool| Layer {
        keep,
        ..Layer::of(p)
    };
    let look = Look {
        id: "heavy".into(),
        name: String::new(),
        about: String::new(),
        tags: Vec::new(),
        layers: vec![
            layer("halation", true),
            layer("glow", false),
            layer("diffusion", false),
            layer("halftone_cmyk", false),
            layer("grain", false),
        ],
    };
    let cost: f32 = look
        .layers
        .iter()
        .map(|l| passes::named(&l.pass).unwrap().cost)
        .sum();
    assert!(cost > guard::BUDGET, "{cost}");
    let c = look.check();
    assert!(c.cost() <= guard::BUDGET);
    // Glow first (as dear as halation, which is kept), then the later of
    // the two next dearest.
    assert_eq!(c.dropped, vec!["glow", "halftone_cmyk"]);
    assert_eq!(look.check(), c, "the same every time");
    assert!(c.layers.iter().any(|l| l.pass.name == "halation"));
    let many = Look {
        layers: (0..9).map(|_| Layer::of("grain")).collect(),
        ..look
    };
    assert_eq!(many.check().layers.len(), guard::MOST_LAYERS);
}

/// The music moving a look is scaled down to the flash limits: however
/// strong the drives asked, the fast ones together move at most
/// [`guard::FAST_LIGHT`] of the light, the slow ones [`guard::SLOW_LIGHT`].
#[test]
fn the_music_moves_the_light_only_so_far() {
    let look = Look::from_json(
        r#"{"id": "strobe", "layers": [
            {"pass": "lift_gamma_gain", "drive": [
                {"knob": "gain", "by": "beat", "amount": 1},
                {"knob": "gamma", "by": "kick", "amount": -1},
                {"knob": "lift", "by": "chorus", "amount": 1}]},
            {"pass": "vignette", "drive": [{"knob": "amount", "by": "snare", "amount": 1}]},
            {"pass": "grain", "drive": [{"knob": "mix", "by": "downbeat", "amount": 1}]}
        ]}"#,
    )
    .unwrap();
    let c = look.check();
    let moved = |speed: drive::Speed| -> f32 {
        c.layers
            .iter()
            .flat_map(|l| &l.drives)
            .filter(|d| d.by.speed() == speed)
            .map(|d| d.light * d.amount.abs())
            .sum()
    };
    assert!(moved(drive::Speed::Fast) <= guard::FAST_LIGHT + 1e-6);
    assert!(moved(drive::Speed::Slow) <= guard::SLOW_LIGHT + 1e-6);
    assert!(
        moved(drive::Speed::Fast) > 0.5 * guard::FAST_LIGHT,
        "scaled, not dropped"
    );
}

/// A look goes to JSON and back as it was, and the same JSON gives the
/// same frame, byte for byte.
#[test]
fn a_look_goes_to_json_and_back_and_draws_the_same() {
    for p in presets::all() {
        let back = Look::from_json(&p.to_json()).expect("read back");
        assert_eq!(&back, p, "{}", p.id);
    }
    let img = fixture(48, 27);
    let p = &presets::all()[0];
    let a = drawn(&p.check(), &img, 12.3, Some((0.7, 0.85)));
    let b = drawn(
        &Look::from_json(&p.to_json()).unwrap().check(),
        &img,
        12.3,
        Some((0.7, 0.85)),
    );
    assert_eq!(a.to_rgb(), b.to_rgb());
}

/// Each knob moves no more light across its whole range than it says
/// (the flash guard's measure), measured over the fixture with every
/// other knob at its default.
#[test]
fn every_knob_moves_no_more_light_than_it_says() {
    let img = fixture(64, 36);
    let mut worst = Vec::new();
    for p in passes::PASSES.iter().filter(|p| !p.history) {
        for (i, k) in p.knobs.iter().enumerate() {
            let at = |v: f32| {
                let mut knobs = std::collections::BTreeMap::new();
                knobs.insert(k.name.to_owned(), v);
                let look = Look {
                    id: "k".into(),
                    name: String::new(),
                    about: String::new(),
                    tags: Vec::new(),
                    layers: vec![Layer {
                        knobs,
                        drive: Some(Vec::new()),
                        ..Layer::of(p.name)
                    }],
                };
                guard::light(&drawn(&look.check(), &img, 5.0, None).px)
            };
            let steps = 8;
            let lights: Vec<f32> = (0..=steps)
                .map(|s| at(k.lo + (k.hi - k.lo) * s as f32 / steps as f32))
                .collect();
            let (lo, hi) = lights
                .iter()
                .fold((f32::MAX, f32::MIN), |(a, b), &l| (a.min(l), b.max(l)));
            let moved = hi - lo;
            if moved > k.light + 0.01 {
                worst.push(format!("{}.{}: {moved:.3} > {}", p.name, k.name, k.light));
            }
            let _ = i;
        }
    }
    assert!(worst.is_empty(), "{worst:#?}");
}

/// No pass at any knob's ends breaks the picture: no number that is none,
/// no frame all black or all white (the fixture's light kept within
/// reach), nothing past the light's bounds.
#[test]
fn no_pass_breaks_the_picture_at_its_ends() {
    let img = fixture(40, 24);
    let base = guard::light(&img.px);
    for p in passes::PASSES {
        for corner in 0..4u32 {
            let mut knobs = std::collections::BTreeMap::new();
            for (i, k) in p.knobs.iter().enumerate() {
                let v = if (corner >> (i % 2)) & 1 == 1 {
                    k.hi
                } else {
                    k.lo
                };
                knobs.insert(
                    k.name.to_owned(),
                    if corner == 3 && i % 3 == 2 {
                        k.default
                    } else {
                        v
                    },
                );
            }
            let look = Look {
                id: "ends".into(),
                name: String::new(),
                about: String::new(),
                tags: Vec::new(),
                layers: vec![Layer {
                    knobs,
                    ..Layer::of(p.name)
                }],
            };
            let out = drawn(&look.check(), &img, 33.0, None);
            assert!(
                out.px
                    .iter()
                    .all(|c| c.x.is_finite() && c.y.is_finite() && c.z.is_finite()),
                "{}",
                p.name
            );
            let l = guard::light(&out.px);
            assert!(
                l > 0.01 && l < 0.97,
                "{} at corner {corner}: {l} (the picture's {base})",
                p.name
            );
            let m = mean(&out);
            assert!(m.x <= 4.0 && m.y <= 4.0 && m.z <= 4.0);
        }
    }
}

/// In the lyric band the light stays under the ceiling (or the picture's
/// own there), whatever the pass: white words keep 3 to 1.
#[test]
fn the_lyric_band_stays_readable_under_every_pass() {
    let img = fixture(64, 40);
    let band = (0.66, 0.86);
    for p in passes::PASSES {
        let look = Look {
            id: "band".into(),
            name: String::new(),
            about: String::new(),
            tags: Vec::new(),
            layers: vec![Layer::of(p.name)],
        };
        let out = drawn(&look.check(), &img, 20.0, Some(band));
        let inner = (band.0 + 0.01, band.1 - 0.01);
        let c = guard::band_contrast(&out.px, out.w, inner);
        let was = guard::band_contrast(&img.px, img.w, inner);
        assert!(
            c >= 3.0_f32.min(was) - 0.05,
            "{}: {c} (the picture's {was})",
            p.name
        );
    }
}

/// Every preset and every pass as a look of its own, drawn through a
/// whole made-up song frame by frame (twelve a second, small, the frame
/// before fed back), over a dark picture and a light one: at most three
/// flashes in any second, never washed out or blacked out (within the
/// exposure guard), the lyrics readable throughout, no number that is
/// none, and a preset drawn at no less than 60 % of its strength over
/// the dark picture (its defaults are not extreme) and 30 % over the light
/// one (where a bright look is softened); a pass alone at 15 %.
#[test]
fn every_look_keeps_the_guards_over_a_whole_song() {
    let curves = demo::Song::new(64.0, 124.0).curves();
    let dark = fixture(20, 12);
    let light = Image {
        px: dark.px.iter().map(|&c| c * 0.5 + 0.45).collect(),
        ..dark.clone()
    };
    let band = (0.68, 0.86);
    let mut looks: Vec<&Look> = presets::all().iter().collect();
    looks.extend(presets::passes().iter().map(|(_, l)| l));
    let failed = std::sync::Mutex::new(Vec::new());
    std::thread::scope(|s| {
        for chunk in looks.chunks(looks.len().div_ceil(6)) {
            let (failed, curves, dark, light) = (&failed, &curves, &dark, &light);
            s.spawn(move || {
                for look in chunk {
                    let c = look.check();
                    for (picture, bright) in [(dark, false), (light, true)] {
                        let was = guard::exposure(&picture.px);
                        let readable = guard::band_contrast(&picture.px, picture.w, band);
                        let strength = guard::strength(&c, picture, Some(curves));
                        let r = guard::over_song(&c, picture, curves, (12.0, band));
                        let within = r.clipped <= guard::CLIPPED.max(was.clipped + 0.05) + 0.05
                            && r.mean.0 >= guard::MEAN.0.min(was.mean - 0.1) - 0.03
                            && r.mean.1 <= guard::MEAN.1.max(was.mean + 0.1) + 0.03;
                        // A preset keeps most of its strength over a dark picture;
                        // over a light one, a bright look is softened.
                        let least = match (presets::named(&look.id).is_some(), bright) {
                            (true, false) => 0.6,
                            (true, true) => 0.3,
                            (false, _) => 0.15,
                        };
                        if !(r.finite
                            && r.flashes <= 3
                            && within
                            && r.contrast >= 3.0_f32.min(readable) * 0.9
                            && strength >= least)
                        {
                            failed.lock().unwrap().push(format!(
                                "{} (mean {:.2}): {r:?}, strength {strength}",
                                look.id, was.mean
                            ));
                        }
                    }
                }
            });
        }
    });
    let failed = failed.into_inner().unwrap();
    assert!(failed.is_empty(), "{} failed: {failed:#?}", failed.len());
}
