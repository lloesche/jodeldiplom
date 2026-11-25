//! The shine held to real songs and real singing ([`jodeldiplom::core::shine`]),
//! read only (the data folder is only read, nothing kept; the songs are
//! never touched, each known by its analysis's record of it, but with
//! `--separate`).
//!
//! ```text
//! cargo run --release -p jodeldiplom --example shine -- --songs [FILTER]
//! cargo run --release -p jodeldiplom --example shine -- --take SING_FOLDER [--song SONG_PATH] [--separate] [--phrases]
//! ```
//!
//! `--songs`: every song with its stems kept, its words timed and a
//! melody (those whose path has `FILTER` in it): the original's reference
//! as the game works it out, then the original heard other everyday ways
//! than the reference and its par were ([`EVERYDAY`]: other rates, the
//! delay further off, other octave errors and pitch lost), each phrase's
//! tier: every phrase with a shine must be Diva. A line a song, `DIVA` at
//! its start when all are, and the closest any came (its shine over its
//! Diva bar, its pitch over its pitch bar). Then the other way: the
//! original heard everyday but sung weakly ([`WEAK`]: near silence, every
//! other word left out), never Diva on any phrase.
//!
//! `--take`: a take (`recordings/sing-*`), heard as the game heard it (its
//! delay as it was then, the key it was sung in) and with the delay that
//! suits it best, against the original of its song (`--song`: that song's
//! file, where the take was sung elsewhere; `--separate`: its voice
//! separated now, where its stems are not kept): its phrases by tier.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use jodel_lyrics::Lyrics;
use jodel_rhythm::waveform::{Melody, Waveform};
use jodel_rhythm::yin::{Yin, WINDOW};
use jodeldiplom::core::shine::reference::{measure_all, Hearing, Original};
use jodeldiplom::core::shine::{self, Frame, Measured, Reference, Shone, Tier, Yardstick};
use jodeldiplom::core::singing::{melodic, Cue, Cursor, Notes, Singing};
use jodeldiplom::keep;

/// The everyday hearings the original must be Diva in: none of them as
/// the reference or the par were made.
const EVERYDAY: [Hearing; 5] = [
    Hearing::clean(22_050, 0.015),
    Hearing::everyday(-0.035, 101),
    Hearing::everyday(0.035, 202),
    Hearing {
        rate: 32_000,
        late: -0.01,
        octave: 0.02,
        dropped: 0.15,
        seed: 303,
    },
    Hearing {
        rate: 44_100,
        late: 0.02,
        octave: 0.05,
        dropped: 0.05,
        seed: 404,
    },
];

/// Weak singing: the original's frames with a pitch kept one in twenty
/// (near silence), or only on every other word the shine counts on each
/// phrase (the second, the fourth and so on: half of them at most).
type Kept = fn(&Notes, &[Option<Yardstick>], &Frame, usize) -> bool;
const WEAK: [(&str, Kept); 2] = [
    ("near silence", |_, _, _, i| i % 20 == 0),
    ("every other word", |n, y, f, _| {
        n.word_at(f.t)
            .is_none_or(|((l, w), _)| counted_rank(n, y, l, w).is_none_or(|r| r % 2 == 1))
    }),
];

/// Where among line `l`'s words the shine counts (those the original is
/// heard on, its yardstick in `y`) word `w` is, if it is one.
fn counted_rank(n: &Notes, y: &[Option<Yardstick>], l: usize, w: usize) -> Option<usize> {
    let y = y.get(l)?.as_ref()?;
    let mut words: Vec<usize> = n.lines[l].notes.iter().map(|n| n.word).collect();
    words.dedup();
    let counted = (words.iter().zip(&y.word_hits))
        .filter(|(_, &h)| h >= shine::measure::WORD_LEAST)
        .map(|(&w, _)| w);
    counted.into_iter().position(|c| c == w)
}

fn data() -> PathBuf {
    jodeldiplom::dirs::Data::current().expect("a data folder").0
}

/// The analyses' waveforms by song path: the newest analysis version (of
/// those, the song as it was last changed), its waveform, and the song as
/// it was then (its path, size and time: [`jodeldiplom::cache::identity`]),
/// read from the record alone.
fn analyses(dir: &Path) -> BTreeMap<PathBuf, (u32, PathBuf, String)> {
    let mut out: BTreeMap<PathBuf, (u32, PathBuf, String)> = BTreeMap::new();
    let newer = |(v, id): (u32, &str), (old_v, old_id): (u32, &str)| {
        let changed = |id| jodeldiplom::cache::identity_parts(id).map(|p| p.2);
        (v, changed(id)) > (old_v, changed(old_id))
    };
    for e in std::fs::read_dir(dir).expect("the analyses").flatten() {
        let file = e.path();
        if file.extension().is_none_or(|x| x != "txt") {
            continue;
        }
        let Ok(text) = std::fs::read_to_string(&file) else {
            continue;
        };
        let Some(key) = text.lines().find_map(|l| l.strip_prefix("key ")) else {
            continue;
        };
        let wave = file.with_extension("wave");
        let Some((id, v)) = jodeldiplom::rhythm::key_parts(key) else {
            continue;
        };
        let Some((path, ..)) = jodeldiplom::cache::identity_parts(id) else {
            continue;
        };
        let newest = (out.get(path)).is_none_or(|(old_v, _, old)| newer((v, id), (*old_v, old)));
        if wave.is_file() && newest {
            out.insert(path.to_owned(), (v, wave, id.to_owned()));
        }
    }
    out
}

/// Our timed words of the song `id` names as the game shows them, read
/// only.
fn shown_lyrics(data: &Path, id: &str) -> Option<Lyrics> {
    let key = jodeldiplom::lyrics::store::key_of(id);
    let text = std::fs::read(data.join("lyrics").join(format!("{key}.timed.lrc"))).ok()?;
    let text = String::from_utf8_lossy(&text).into_owned();
    let lyrics = Lyrics::parse(
        jodeldiplom::cache::LYRICS_TIMED.body(&text)?,
        jodel_lyrics::Source::Unknown,
    );
    if lyrics.is_empty() {
        return None;
    }
    if matches!(lyrics.source, jodel_lyrics::Source::Transcribed(_)) {
        return Some(jodeldiplom::lyrics::relined(&lyrics).1);
    }
    Some(lyrics)
}

/// The vocals of the kept stems of the song `id` names, mono at their
/// rate, read without marking the file used.
fn kept_voice(data: &Path, id: &str) -> Option<(Vec<f32>, u32)> {
    use jodel_playback::stems::{Stem, RATE};
    let file = jodeldiplom::stems_cache::file_of(&data.join("stems"), id);
    let mut kept = jodel_playback::StemFile::open(&file).ok()?;
    let vocals = Stem::ALL.iter().position(|&s| s == Stem::Vocals)?;
    let mut voice = Vec::new();
    while (voice.len() as u64) < kept.frames() {
        let part = kept.read(voice.len() as u64, RATE as usize * 10).ok()?;
        let [l, r] = &part[vocals];
        if l.is_empty() {
            break;
        }
        voice.extend(l.iter().zip(r).map(|(a, b)| (a + b) / 2.0));
    }
    Some((voice, RATE))
}

/// The singing as the game makes it of `lyrics` and `melody`.
fn singing(lyrics: Lyrics, melody: &Melody) -> Singing {
    let mut s = Singing::default();
    s.prepare(Path::new("song"), &Arc::new(lyrics), &melodic(melody));
    s
}

/// The tier of each of `heard`'s phrases with a shine, and how near Diva
/// each came: its shine over its bar, its pitch over its pitch bar.
fn tiers(heard: &[Option<Measured>], r: &Reference) -> Vec<(usize, Option<Tier>, f64, f64)> {
    let mut out = Vec::new();
    for (l, p) in heard.iter().enumerate() {
        let Some(original) = r.phrase(l) else {
            continue;
        };
        let Some(s) = p.as_ref().map(|p| shine::shine(p, &original.yardstick)) else {
            out.push((l, None, 0.0, 0.0));
            continue;
        };
        let par = original.par;
        let bar = Tier::TOP.bar() * par.scale();
        let pitch = match (s.pitch, Tier::TOP.spec().pitch_bar) {
            (Some(p), Some(top)) => p / (top * par.pitch_scale()),
            _ => f64::INFINITY,
        };
        out.push((l, Tier::of(&s, par), s.value / bar, pitch));
    }
    out
}

fn songs(filter: Option<&str>) {
    let data = data();
    let (mut all, mut divas, mut songs_ok, mut songs) = (0, 0, 0, 0);
    let (mut weak_all, mut weak_divas, mut weak_most) = (0, 0, 0.0f64);
    let (mut rough, mut heard_enough) = (0, 0);
    for (path, (_, wave, id)) in analyses(&data.join("analyses")) {
        if filter.is_some_and(|f| !path.display().to_string().contains(f)) {
            continue;
        }
        let melody = std::fs::read(&wave)
            .ok()
            .and_then(|b| Waveform::from_bytes(&b))
            .and_then(|w| w.melody);
        let (Some(melody), Some(lyrics)) = (melody, shown_lyrics(&data, &id)) else {
            continue;
        };
        let Some((voice, rate)) = kept_voice(&data, &id) else {
            continue;
        };
        let started = std::time::Instant::now();
        let notes = Notes::new(&lyrics, &melody);
        let mut original = Original::new(&notes, &voice, rate);
        let r = original.reference();
        let yardsticks = yardsticks_of(&r);
        let made = started.elapsed().as_secs_f64();
        let mut worst = (f64::INFINITY, f64::INFINITY);
        let (mut fails, mut weak_fails) = (Vec::new(), Vec::new());
        let mut n = 0;
        for how in &EVERYDAY {
            let heard = original.measure(how, &yardsticks);
            for (l, tier, over, pitch) in tiers(&heard, &r) {
                n += 1;
                worst = (worst.0.min(over), worst.1.min(pitch));
                if tier != Some(Tier::TOP) {
                    fails.push(format!(
                        "{:?} at {} Hz {:+.0} ms: phrase {l} {tier:?} ({over:.2} {pitch:.2})",
                        how.seed,
                        how.rate,
                        how.late * 1000.0
                    ));
                }
            }
        }
        // Sung weakly: never Diva.
        let frames = original.frames(&EVERYDAY[1]);
        for (name, kept) in WEAK {
            let mut voiced = 0;
            let weak: Vec<Frame> = (frames.iter())
                .map(|f| {
                    let keep = f.cents.is_some() && {
                        voiced += 1;
                        kept(&notes, &yardsticks, f, voiced)
                    };
                    Frame {
                        cents: f.cents.filter(|_| keep),
                        ..*f
                    }
                })
                .collect();
            let heard = measure_all(&notes, &weak, &yardsticks);
            for (l, tier, over, _) in tiers(&heard, &r) {
                weak_all += 1;
                weak_most = weak_most.max(over);
                if tier == Some(Tier::TOP) {
                    weak_divas += 1;
                    weak_fails.push(format!("sung weakly, {name}: phrase {l} Diva ({over:.2})"));
                }
            }
        }
        let shining = r.shining();
        let enough = (r.phrases.iter().flatten())
            .filter(|p| shine::shines(&p.yardstick))
            .count();
        heard_enough += enough;
        rough += enough - shining;
        all += n;
        divas += n - fails.len();
        songs += 1;
        let ok = fails.is_empty() && weak_fails.is_empty();
        songs_ok += usize::from(ok);
        println!(
            "{} {:>3} phrases shine of {:>3} heard enough ({:>3} in all), {:>3} hearings Diva of {:>3}; closest: shine {:.2} of its bar, pitch {:.2}; made in {made:.1} s  {}",
            if ok { "DIVA" } else { "MISS" },
            shining,
            enough,
            r.phrases.len(),
            n - fails.len(),
            n,
            worst.0,
            worst.1,
            path.file_name().unwrap_or_default().to_string_lossy()
        );
        for f in fails.iter().chain(&weak_fails) {
            println!("    {f}");
        }
    }
    println!(
        "\n{songs_ok} of {songs} songs Diva on every phrase in every hearing and never sung weakly; {divas} of {all} phrase hearings Diva; sung weakly, {weak_divas} of {weak_all} Diva (the most {weak_most:.2} of its bar); {rough} of {heard_enough} phrases heard enough too rough to shine"
    );
}

/// The reference's yardsticks, a phrase a line.
fn yardsticks_of(r: &Reference) -> Vec<Option<Yardstick>> {
    r.phrases
        .iter()
        .map(|p| p.as_ref().map(|p| p.yardstick.clone()))
        .collect()
}

/// A take's frames: each at the time in the song it was sung (no delay
/// taken off), its pitch moved back into the song's own key; and the
/// sample it was heard at.
fn take_frames(
    voice: &[f32],
    rate: u32,
    marks: &[(usize, f64)],
    keys: &[(usize, i8)],
) -> Vec<(usize, f64, Option<f32>)> {
    let mut out = Vec::new();
    let mut yin = Yin::new(rate);
    let mut m = 0;
    for (i, &x) in voice.iter().enumerate() {
        while m + 1 < marks.len() && marks[m + 1].0 <= i {
            m += 1;
            let ((k, t), (k0, t0)) = (marks[m], marks[m - 1]);
            if (t - (t0 + (k - k0) as f64 / f64::from(rate))).abs() > 0.05 {
                yin = Yin::new(rate);
            }
        }
        let (k, t0) = marks.get(m).copied().unwrap_or((0, 0.0));
        let key = keys
            .iter()
            .rev()
            .find(|&&(at, _)| at <= i)
            .map_or(0, |&(_, s)| s);
        yin.push(&[x], |cents| {
            let t = t0 + (i - k) as f64 / f64::from(rate) - WINDOW / 2.0;
            out.push((i, t, cents.map(|c| c - 100.0 * f32::from(key))));
        });
    }
    out
}

/// `frames` (time, pitch) heard by the game's singing against `r`, sung
/// to the end: each phrase as it last shone (a seek back hears it again).
fn sung(
    singing: &Singing,
    r: &Arc<Reference>,
    frames: impl Iterator<Item = (f64, Option<f32>)>,
) -> BTreeMap<usize, Shone> {
    let mut s = singing.clone();
    s.restart();
    s.set_reference(Some(Arc::clone(r)));
    let mut out = BTreeMap::new();
    let mut cursor = Cursor::default();
    let mut read = |s: &Singing, cursor: &mut Cursor| {
        for m in s.moments_since(cursor).moments {
            if let Cue::Phrase(p) = m.cue {
                out.insert(p.line, p);
            }
        }
    };
    for (t, c) in frames {
        s.hear(t, c);
        read(&s, &mut cursor);
    }
    s.finish();
    read(&s, &mut cursor);
    out
}

/// The phrases of `shone` by tier: a line.
fn spread(label: &str, shone: &BTreeMap<usize, Shone>, all: usize) -> f64 {
    let count = |tier: Option<Tier>| shone.values().filter(|p| p.tier == tier).count();
    let tiers: Vec<String> = (Tier::ALL.iter().rev())
        .map(|&t| format!("{} {}", t.name(), count(Some(t))))
        .chain([format!("none {}", count(None))])
        .collect();
    let mut v: Vec<f32> = shone.values().map(Shone::of_top).collect();
    v.sort_by(f32::total_cmp);
    let median = v.get(v.len() / 2).copied().unwrap_or(0.0);
    println!(
        "  {label:<34} {} of {all} phrases: {}; median {median:.2} of {}'s bar",
        shone.len(),
        tiers.join(" · "),
        Tier::TOP.name()
    );
    f64::from(median)
}

fn take(folder: &Path, song: Option<PathBuf>, separate: bool, show: bool) {
    let read =
        |name: &str| std::fs::read(folder.join(name)).unwrap_or_else(|e| panic!("{name}: {e}"));
    let (voice, rate) = keep::read_wav(&read(keep::VOICE)).expect("the voice");
    let marks = keep::read_times(&String::from_utf8_lossy(&read(keep::TIMES)));
    let lyrics = Lyrics::parse(
        &String::from_utf8_lossy(&read(keep::LYRICS)),
        jodel_lyrics::Source::Unknown,
    );
    let melody = Waveform::from_bytes(&read(keep::MELODY))
        .and_then(|w| w.melody)
        .expect("a melody");
    let about = String::from_utf8_lossy(&read(keep::ABOUT)).into_owned();
    let path = song.unwrap_or_else(|| {
        PathBuf::from(
            about
                .lines()
                .find_map(|l| l.strip_prefix("song "))
                .expect("its song"),
        )
    });
    println!("take {} of {}", folder.display(), path.display());
    let singing = singing(lyrics, &melody);
    let (original, orate) = if separate {
        (
            jodeldiplom::lyrics::timer::separate_16k(&path, false)
                .expect("separated")
                .0,
            16_000,
        )
    } else {
        let analysed = analyses(&data().join("analyses"));
        let (_, _, id) = analysed.get(&path).expect("the song analysed");
        kept_voice(&data(), id).expect("the song's stems kept")
    };
    let r = Arc::new(Original::new(&singing.notes, &original, orate).reference());
    let all = r.shining();
    let lates: Vec<(usize, f64)> = about
        .lines()
        .filter_map(|l| l.strip_prefix("late "))
        .filter_map(|l| {
            let mut w = l.split_whitespace();
            Some((w.next()?.parse().ok()?, w.next()?.parse().ok()?))
        })
        .collect();
    let frames = take_frames(&voice, rate, &marks, &keep::read_keys(&about));
    let late_at = |k: usize| {
        lates
            .iter()
            .rev()
            .find(|&&(at, _)| at <= k)
            .map_or(lates.first().map_or(0.2, |l| l.1), |l| l.1)
    };
    let game = sung(
        &singing,
        &r,
        frames.iter().map(|&(k, t, c)| (t - late_at(k), c)),
    );
    spread("as the game heard it", &game, all);
    let mut best = (0.0, f64::NEG_INFINITY, BTreeMap::new());
    for ms in (0..=400).step_by(20) {
        let late = f64::from(ms) / 1000.0;
        let got = sung(&singing, &r, frames.iter().map(|&(_, t, c)| (t - late, c)));
        let mut v: Vec<f32> = got.values().map(Shone::of_top).collect();
        v.sort_by(f32::total_cmp);
        let median = f64::from(v.get(v.len() / 2).copied().unwrap_or(0.0));
        if median > best.1 {
            best = (late, median, got);
        }
    }
    spread(
        &format!("at the delay best for it, {:.0} ms", best.0 * 1000.0),
        &best.2,
        all,
    );
    if show {
        for (l, p) in &game {
            println!(
                "    phrase {l:>3} at {:>6.1} s: {:.2} of {}'s bar, {:?}",
                p.end,
                p.of_top(),
                Tier::TOP.name(),
                p.tier
            );
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let value = |name: &str| {
        args.iter()
            .position(|a| a == name)
            .and_then(|k| args.get(k + 1))
            .cloned()
    };
    if args.iter().any(|a| a == "--songs") {
        songs(value("--songs").as_deref().filter(|f| !f.starts_with("--")));
        return;
    }
    let folder = value("--take").expect("--songs or --take");
    take(
        Path::new(&folder),
        value("--song").map(PathBuf::from),
        args.iter().any(|a| a == "--separate"),
        args.iter().any(|a| a == "--phrases"),
    );
}
