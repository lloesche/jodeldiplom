//! Every preset held to the looks' guards over real songs' curves: the
//! analyses this machine's player keeps (its data folder's `analyses`,
//! or `JODELDIPLOM_DATA`'s), read as the player reads them, the first two
//! with their waveforms (skipped, with a note, where there are none). Each
//! preset is drawn through each whole song frame by frame, small, the
//! frame before fed back: at most three flashes in any second, never
//! washed out or blacked out, the lyrics readable, no number that is none.

use std::path::{Path, PathBuf};

use jodel_looks::math::v3;
use jodel_looks::render::Image;
use jodel_looks::{guard, presets, Curves};
use jodel_rhythm::waveform::Waveform;

/// The data folder's analyses, where this machine has them.
fn analyses() -> Option<PathBuf> {
    let data = std::env::var_os("JODELDIPLOM_DATA")
        .map(PathBuf::from)
        .or_else(|| {
            let base = std::env::var_os("LOCALAPPDATA")
                .map(PathBuf::from)
                .or_else(|| {
                    std::env::var_os("HOME")
                        .map(|h| PathBuf::from(h).join("Library/Application Support"))
                })?;
            Some(base.join("Jodeldiplom Data"))
        })?;
    let dir = data.join("analyses");
    dir.is_dir().then_some(dir)
}

/// The first `n` songs kept in `dir` with their waveforms, as curves.
fn songs(dir: &Path, n: usize) -> Vec<(String, Curves)> {
    let mut texts: Vec<PathBuf> = std::fs::read_dir(dir)
        .map(|r| r.flatten().map(|e| e.path()).collect())
        .unwrap_or_default();
    texts
        .retain(|p| p.extension().is_some_and(|x| x == "txt") && p.with_extension("wave").exists());
    texts.sort();
    let mut out = Vec::new();
    for text in texts {
        if out.len() == n {
            break;
        }
        let Ok(body) = std::fs::read_to_string(&text) else {
            continue;
        };
        let Some(key) = body
            .lines()
            .find_map(|l| l.strip_prefix("key "))
            .map(str::to_owned)
        else {
            continue;
        };
        let Some(analysis) = jodeldiplom::work::rhythm::from_text(&key, &body) else {
            continue;
        };
        let Some(wave) = std::fs::read(text.with_extension("wave"))
            .ok()
            .and_then(|b| Waveform::from_bytes(&b))
        else {
            continue;
        };
        let bars: Vec<f64> = (analysis.downbeats.iter())
            .filter_map(|&i| analysis.beats.get(i).copied())
            .collect();
        let sections = jodeldiplom::gui::backdrop::music::sections(&wave.mix, &bars);
        let name = text.file_stem().unwrap().to_string_lossy().into_owned();
        out.push((name, Curves::new(&analysis, Some(&wave), &sections, &[])));
    }
    out
}

/// A picture to draw over: a sky to a dark ground, a bright disc.
fn picture() -> Image {
    let (w, h) = (16, 10);
    let mut px = Vec::new();
    for y in 0..h {
        for x in 0..w {
            let (u, v) = ((x as f32 + 0.5) / w as f32, (y as f32 + 0.5) / h as f32);
            let mut c = if v > 0.6 {
                v3(0.18, 0.14, 0.1)
            } else {
                v3(0.3 + 0.4 * v, 0.35 + 0.3 * v, 0.7 - 0.2 * v)
            };
            if (u - 0.7).hypot(v - 0.3) < 0.12 {
                c = v3(1.0, 0.92, 0.7);
            }
            px.push(c);
        }
    }
    Image { w, h, px }
}

#[test]
fn every_preset_keeps_the_guards_over_real_songs() {
    let Some(dir) = analyses() else {
        eprintln!("no analyses kept on this machine: skipped");
        return;
    };
    let songs = songs(&dir, 2);
    if songs.is_empty() {
        eprintln!(
            "no analysis with its waveform in {}: skipped",
            dir.display()
        );
        return;
    }
    let pic = picture();
    let was = guard::exposure(&pic.px);
    let band = (0.68, 0.86);
    let readable = guard::band_contrast(&pic.px, pic.w, band);
    let looks: Vec<&jodel_looks::Look> = presets::all().iter().collect();
    let failed = std::sync::Mutex::new(Vec::new());
    std::thread::scope(|s| {
        for chunk in looks.chunks(looks.len().div_ceil(6)) {
            let (failed, songs, pic) = (&failed, &songs, &pic);
            s.spawn(move || {
                for look in chunk {
                    let c = look.check();
                    for (name, curves) in songs {
                        let r = guard::over_song(&c, pic, curves, (8.0, band));
                        let within = r.clipped
                            <= guard::CLIPPED.max(was.clipped + guard::CLIPPED_SLACK) + 0.05
                            && r.mean.0 >= guard::MEAN.0.min(was.mean - guard::MEAN_SLACK) - 0.03
                            && r.mean.1 <= guard::MEAN.1.max(was.mean + guard::MEAN_SLACK) + 0.03;
                        if !(r.finite
                            && r.flashes <= 3
                            && within
                            && r.contrast >= 3.0_f32.min(readable) * 0.9)
                        {
                            failed
                                .lock()
                                .unwrap()
                                .push(format!("{} over {name}: {r:?}", look.id));
                        }
                    }
                }
            });
        }
    });
    let failed = failed.into_inner().unwrap();
    assert!(failed.is_empty(), "{} failed: {failed:#?}", failed.len());
}
