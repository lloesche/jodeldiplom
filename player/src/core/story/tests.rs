//! The plan's reading and its rules on good and bad answers, the prompt
//! (a snapshot, every count it asks for), the streamed prose, the look
//! the shaders make and the song as the plan is made from it; no model
//! needed.

use super::check::{read, text_word};
use super::prompt::prompt;
use super::prose::Prose;
use super::rules::{self, COUNTS};
use super::*;

/// A song of twelve lines over two minutes (words of our own).
pub(crate) fn song() -> Song {
    let words = [
        "I walked down to the harbour",
        "The boats were sleeping in the rain",
        "You came running with your coat undone",
        "And we laughed at the grey sky",
        "Carry me over the water",
        "Carry me home before the night",
        "My father mended nets all winter",
        "His hands were rough as rope",
        "Now the lighthouse turns for no one",
        "And the gulls have gone inland",
        "Carry me over the water",
        "Carry me home before the night",
    ];
    let lines = (words.iter().enumerate())
        .map(|(k, w)| Line {
            start: 8.0 + 9.0 * k as f64,
            end: 8.0 + 9.0 * k as f64 + 7.5,
            text: (*w).to_owned(),
        })
        .collect();
    Song {
        title: "Harbour Song".into(),
        artist: "The Example Band".into(),
        album: "Grey Water".into(),
        tags: vec![("date".into(), "1994".into())],
        duration: 120.0,
        lines,
    }
}

/// The story of [`GOOD`].
const STORY: &str = "A woman remembers the harbour of her youth and a friend who ran to her in the rain. Her father's work is over and the town is empty, and she longs to be carried home.";

/// A good answer for [`song`], fenced as the model writes it.
pub(crate) const GOOD: &str = r##"```json
{"story": "A woman remembers the harbour of her youth and a friend who ran to her in the rain. Her father's work is over and the town is empty, and she longs to be carried home.",
 "characters": [{"name": "narrator", "looks": "Woman, late 20s, slender, long dark hair, yellow raincoat"},
                {"name": "you", "looks": "Man, late 20s, tall, curly red hair, open navy coat"},
                {"name": "father", "looks": "Man, 60s, broad, grey beard, wool sweater"}],
 "scenes": [
  {"first_line": 0, "last_line": 1, "section": "verse 1", "shot": "wide", "characters": ["narrator"], "shows": "The narrator walks to a rainy harbour.", "prompt": "A small harbour in steady rain at dusk, narrator walking along the quay, moored fishing boats in the foreground, grey hills behind"},
  {"first_line": 2, "last_line": 3, "section": "verse 1", "shot": "medium", "characters": ["narrator", "you"], "shows": "The two laugh under a grey sky.", "prompt": "Rain-soaked quay in the afternoon, you running toward narrator with his coat open, both laughing, puddles in front, low clouds over the sea"},
  {"first_line": 4, "last_line": 5, "section": "chorus", "shot": "wide", "characters": [], "shows": "The sea at nightfall.", "prompt": "Wide dark sea at nightfall, a small rowing boat crossing toward distant lights, waves in the foreground, a fading sky above"},
  {"first_line": 6, "last_line": 7, "section": "verse 2", "shot": "close-up", "characters": ["father"], "shows": "The father mends nets.", "prompt": "Close view of father's rough hands mending a fishing net by lamplight in winter, coils of rope in front, a frosted window behind"},
  {"first_line": 8, "last_line": 9, "section": "verse 2", "shot": "wide", "characters": [], "shows": "An empty lighthouse turns.", "prompt": "Lonely lighthouse on a rocky point at night, its beam sweeping over empty water, rocks in the foreground, clouds drifting behind"},
  {"first_line": 10, "last_line": 11, "section": "chorus", "shot": "medium", "characters": ["narrator"], "shows": "The narrator looks out over the water.", "prompt": "Narrator standing at the end of a wooden pier at dusk, wind in her hair, the pier's posts in front, the open sea and a faint shore behind"}
 ],
 "art": {"knowledge": "I do not know this band.", "confidence": "low", "genre": "folk rock", "era": "90s",
  "style": "watercolor", "style_keywords": "watercolor, soft washes, rain, muted greys, paper texture, 1990s",
  "palette": ["#3a4a5a", "#8a9aa8", "#d9c27a", "#2b2b2b", "#f0ece0"], "motifs": ["boats", "rain", "lighthouse"],
  "why": "Soft washes fit a remembered, rainy town."},
 "shader": {"preset": "folk", "add": ["grain"], "knobs": {"canvas": {"amount": 0.5}, "grain": {"amount": 0.45, "size": 2}},
            "drive": [{"pass": "liquid", "knob": "amount", "by": "chorus", "amount": 0.6}], "mood": ["longing", "rain"]}}
```"##;

/// [`GOOD`] with `from` replaced by `to` (which must be there).
fn bad(from: &str, to: &str) -> String {
    assert!(GOOD.contains(from), "{from}");
    GOOD.replacen(from, to, 1)
}

/// The error reading `answer`, which must have one.
fn refused(answer: &str) -> String {
    read(answer, &song()).expect_err("refused")
}

/// A good answer is read whole: its characters, its scenes with their
/// times from the lines (the first from the song's start, the last to its
/// end), its art, its look; kept and read back the same.
#[test]
fn a_good_answer_is_read_and_its_scenes_timed() {
    let plan = read(GOOD, &song()).expect("a good plan");
    assert_eq!(plan.characters.len(), 3);
    assert_eq!(plan.scenes.len(), 6);
    let spans: Vec<(f64, f64)> = plan.scenes.iter().map(|s| (s.from, s.to)).collect();
    assert_eq!(spans[0], (0.0, 26.0));
    assert_eq!(spans[1], (26.0, 44.0));
    assert_eq!(spans[5], (98.0, 120.0));
    assert_eq!(plan.scenes[3].shot, Shot::CloseUp);
    assert_eq!(plan.art.confidence, Confidence::Low);
    assert_eq!(plan.art.style, Style::Watercolor);
    assert_eq!(plan.art.palette[0], Rgb([0x3a, 0x4a, 0x5a]));
    assert_eq!(plan.art.palette[0].to_string(), "#3a4a5a");
    assert_eq!(Plan::from_json(&plan.to_json()), Ok(plan));
}

/// Every rule broken is refused with an error that says which and where.
#[test]
fn a_broken_answer_is_refused_precisely() {
    let seven: Vec<String> = (0..7)
        .map(|k| format!("{{\"name\": \"person {k}\", \"looks\": \"Someone\"}}"))
        .collect();
    let characters =
        &GOOD[GOOD.find("[{\"name\"").unwrap()..GOOD.find("],\n \"scenes\"").unwrap() + 1];
    let cases: Vec<(String, &str)> = vec![
        ("no braces at all".into(), "holds no JSON object"),
        (bad("\"story\": \"A", "\"stroy\": \"A"), "unknown field `stroy`"),
        (
            bad(
                "\"section\": \"verse 1\", \"shot\": \"wide\"",
                "\"section\": \"verse 1\", \"shot\": \"wide\", \"shot\": \"wide\"",
            ),
            "duplicate field `shot`",
        ),
        (
            bad(
                "\"section\": \"verse 1\", \"shot\": \"wide\"",
                "\"section\": \"verse 1\", \"shot\": \"wide\", \"shot\": \"wide\"",
            ),
            "(write each field of an object once)",
        ),
        (bad("\"art\": {", ", \"art\": {"), "does not read"),
        (
            bad("\"first_line\": 0,", "\"first_line\": 0, \"from\": 3,"),
            "unknown field `from`",
        ),
        (bad(&format!("\"{STORY}\""), "\" \""), "\"story\" is empty"),
        (
            bad(characters, &format!("[{}]", seven.join(", "))),
            "7 characters: at most 6 were asked for",
        ),
        // The scenes' lines.
        (
            bad("\"first_line\": 2, \"last_line\": 3", "\"first_line\": 3, \"last_line\": 3"),
            "line 2 is in no scene (scene 1 starts at line 3)",
        ),
        (
            bad("\"first_line\": 2, \"last_line\": 3", "\"first_line\": 1, \"last_line\": 3"),
            "scenes 0 and 1 both take line 1",
        ),
        (
            bad("\"first_line\": 10, \"last_line\": 11", "\"first_line\": 10, \"last_line\": 10"),
            "line 11 is in no scene: the last scene must end at line 11",
        ),
        (
            bad("\"first_line\": 10, \"last_line\": 11", "\"first_line\": 10, \"last_line\": 12"),
            "scene 5 ends at line 12, but the last line is 11",
        ),
        (
            bad("\"first_line\": 6, \"last_line\": 7", "\"first_line\": 7, \"last_line\": 6"),
            "scene 3 starts at line 7 after its last line 6",
        ),
        // The scenes.
        (
            bad("[\"father\"]", "[\"mother\"]"),
            "scene 3 shows \"mother\", who is not among the characters (narrator, you, father)",
        ),
        (
            bad("[\"narrator\", \"you\"]", "[\"narrator\", \"narrator\"]"),
            "scene 1 names \"narrator\" twice",
        ),
        (
            bad("{\"name\": \"father\"", "{\"name\": \"Father\""),
            "character 2's name \"Father\" is not one or two lowercase words",
        ),
        (bad("{\"name\": \"father\"", "{\"name\": \"you\""), "two characters are named \"you\""),
        (
            bad("Lonely lighthouse", "Lonely lighthouse with a poster"),
            "scene 4's prompt asks for writing in the picture (\"poster\")",
        ),
        (
            bad("\"prompt\": \"A small harbour in steady rain at dusk, narrator walking along the quay, moored fishing boats in the foreground, grey hills behind\"", "\"prompt\": \"A harbour\""),
            "scene 0's prompt: 2 words: 15 to 35 were asked for",
        ),
        (bad("\"shot\": \"wide\"", "\"shot\": \"aerial\""), "unknown variant `aerial`"),
        // The art.
        (bad("\"style\": \"watercolor\"", "\"style\": \"fresco\""), "unknown variant `fresco`"),
        (bad("\"#3a4a5a\"", "\"slate\""), "the colour \"slate\" is not \"#rrggbb\""),
        (
            bad("\"#3a4a5a\", \"#8a9aa8\", \"#d9c27a\", ", ""),
            "the palette: 2 colours: 5 were asked for",
        ),
        (
            bad("[\"boats\", \"rain\", \"lighthouse\"]", "[]"),
            "0 recurring objects or symbols from the lyrics: 3 to 5 were asked for",
        ),
        (
            bad("watercolor, soft washes, rain, muted greys, paper texture, 1990s", "watercolor"),
            "\"style_keywords\": 1 comma-separated keywords: 6 to 10 were asked for",
        ),
        (
            bad("paper texture, 1990s", "paper texture, 1990s, hand lettering"),
            "\"style_keywords\" ask for writing (\"lettering\")",
        ),
        (bad("\"confidence\": \"low\"", "\"confidence\": \"unsure\""), "unknown variant `unsure`"),
        // The shaders.
        (
            bad("\"preset\": \"folk\"", "\"preset\": \"folk-forever\""),
            "the shader's preset \"folk-forever\" is not in the catalogue",
        ),
        (bad("\"mood\": [\"longing\", \"rain\"]", "\"mood\": []"), "0 mood words: 2 to 4 were asked for"),
        (bad("\"size\": 2", "\"size\": 4"), "grain.size is 4: its range is 0.6 to 3"),
        (bad("\"size\": 2", "\"sise\": 2"), "grain has no knob \"sise\" (its knobs: amount, size, chroma)"),
        (bad("\"add\": [\"grain\"]", "\"add\": []"), "grain is moved, but the look has no grain (its passes: watercolor, canvas, liquid; add it to \"add\")"),
        (bad("\"add\": [\"grain\"]", "\"add\": [\"grain\", \"laser\"]"), "there is no pass \"laser\""),
        (
            bad("\"add\": [\"grain\"], \"knobs\": {", "\"add\": [\"grain\", \"cutout\"], \"knobs\": {\"cutout\": {\"layers\": 3.5}, "),
            "cutout.layers is 3.5: a count, a whole number",
        ),
        (bad("\"canvas\": {\"amount\": 0.5}", "\"canvas\": 0.5"), "invalid type: floating point `0.5`, expected a map"),
        (
            bad("\"add\": [\"grain\"]", "\"add\": [\"grain\", \"dust\"]").replacen("\"pass\": \"liquid\", \"knob\": \"amount\"", "\"pass\": \"dust\", \"knob\": \"lit\"", 1),
            "the music cannot move dust.lit",
        ),
        (
            bad("\"amount\": 0.6}]", "\"amount\": 0.6}, {\"pass\": \"liquid\", \"knob\": \"amount\", \"by\": \"bass\", \"amount\": 0.2}]"),
            "liquid.amount is moved twice",
        ),
        (bad("\"by\": \"chorus\"", "\"by\": \"weather\""), "unknown variant `weather`"),
        (bad("\"amount\": 0.6", "\"amount\": 1.6"), "liquid.amount is moved by 1.6: -1 to 1"),
        (bad("\"pass\": \"liquid\"", "\"pass\": \"laser\""), "laser is moved, but the look has no laser"),
    ];
    for (answer, says) in cases {
        let why = refused(&answer);
        assert!(why.contains(says), "{says:?} not in {why:?}");
    }
    // Every rule broken is said at once, so one retry can mend them all.
    let both = bad("Lonely lighthouse", "Lonely lighthouse with a poster").replacen(
        "\"preset\": \"folk\"",
        "\"preset\": \"folk-forever\"",
        1,
    );
    let why = refused(&both);
    assert!(
        why.contains("scene 4's prompt asks for writing") && why.contains("\"folk-forever\""),
        "{why}"
    );
    // Too few scenes for two minutes: two, where 6 to 12 were asked for.
    let mut few: serde_json::Value =
        serde_json::from_str(&GOOD[GOOD.find('{').unwrap()..=GOOD.rfind('}').unwrap()]).unwrap();
    let scenes = few["scenes"].as_array_mut().unwrap();
    scenes.truncate(2);
    scenes[1]["last_line"] = 11.into();
    assert!(refused(&few.to_string()).contains("2 scenes: 6 to 12 were asked for"));
}

/// The words that ask for writing are found as whole words only, and no
/// art style asks for any.
#[test]
fn no_style_asks_for_writing() {
    assert_eq!(
        text_word("a fountain by the postern gate, its texture worn"),
        None
    );
    assert_eq!(text_word("a torn Poster on the door"), Some("Poster"));
    for s in Style::ALL {
        assert_eq!(text_word(s.words()), None, "{}", s.name());
        assert!(
            !s.words().contains("poster") && !s.name().contains("poster"),
            "{}",
            s.name()
        );
        let read: Style = serde_json::from_str(&format!("\"{}\"", s.name())).unwrap();
        assert_eq!(read, *s);
    }
}

/// The prompt, its catalogue left out, is as the snapshot has it
/// (`tests/snapshots/story_prompt.txt`; `UPDATE_SNAPSHOTS=1` writes it
/// anew); the same song always gives the same prompt; it says every
/// count an answer is held to.
#[test]
fn the_prompt_is_as_its_snapshot_and_says_every_count() {
    let song = song();
    let asked = prompt(&song);
    assert_eq!(asked, prompt(&song), "the same prompt every time");
    let brief = jodel_looks::catalogue::brief();
    assert!(asked.contains(brief));
    for count in COUNTS.iter().chain([&rules::scenes(&song)]) {
        assert!(asked.contains(&count.said()), "{count:?}");
    }
    let shown = asked.replace(brief, "<the catalogue>\n");
    let file =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/snapshots/story_prompt.txt");
    if std::env::var_os("UPDATE_SNAPSHOTS").is_some() {
        durable::write_atomic(&file, shown.as_bytes()).unwrap();
    }
    let kept = std::fs::read_to_string(&file).unwrap_or_default();
    assert!(
        kept == shown,
        "the story's prompt changed (UPDATE_SNAPSHOTS=1 to keep it):\n{shown}"
    );
    assert_eq!(rules::scenes(&song).asked, (6, 12));
}

/// The scenes asked for stay within what fits in the answer, however long
/// the song.
#[test]
fn a_long_song_is_asked_no_more_scenes_than_fit() {
    let mut long = song();
    long.duration = 600.0;
    long.lines = (0..120)
        .map(|k| Line {
            start: 5.0 * k as f64,
            end: 5.0 * k as f64 + 4.0,
            text: "la la la".into(),
        })
        .collect();
    let scenes = rules::scenes(&long);
    let fit = (rules::MOST_ANSWER - rules::ANSWER_REST) / rules::TOKENS_A_SCENE;
    assert_eq!(scenes.asked.1, fit);
    assert!(scenes.asked.0 <= scenes.asked.1);
}

/// The prose of the good answer streamed in pieces of any size is the
/// same: the story, every character with its looks, what every scene
/// shows, the art's style and why, and nothing else.
#[test]
fn the_prose_of_an_answer_streams_out_of_its_json() {
    let plan = read(GOOD, &song()).unwrap();
    let mut want = format!("{STORY}\n\n");
    for c in &plan.characters {
        want.push_str(&format!("{}: {}\n", c.name, c.looks));
    }
    for s in &plan.scenes {
        want.push_str(&format!("· {}\n", s.shows));
    }
    want.push_str(&format!(
        "\nThe look: {}\n{}\n",
        plan.art.style.name(),
        plan.art.why
    ));
    let chars: Vec<char> = GOOD.chars().collect();
    for size in [1, 2, 3, 7, 64, chars.len()] {
        let mut prose = Prose::new();
        let got: String = (chars.chunks(size))
            .map(|c| prose.feed(&c.iter().collect::<String>()))
            .collect();
        assert_eq!(got, want, "pieces of {size}");
    }
}

/// Escapes are read whole however the pieces split them, a line break as
/// a space; a key of the same name elsewhere is not shown.
#[test]
fn the_prose_reads_escapes_and_only_its_fields() {
    let answer = r##"{"story": "She said \"go\"\nand left été 🚗.", "shader": {"name": "not shown", "knobs": {"why": 1}},
 "art": {"style": "watercolor", "palette": ["#000000"], "why": "Soft."}}"##;
    let want = "She said \"go\" and left été 🚗.\n\n\nThe look: watercolor\nSoft.\n";
    // Half a pair alone is dropped, and the string still ends at its quote.
    let lone = r#"{"story": "a\ud83d", "art": {"why": "b\ud83dx\ud83d\n"}}"#;
    let mut prose = Prose::new();
    assert_eq!(prose.feed(lone), "a\n\nbx \n");
    for size in [1, 2, 5, answer.len()] {
        let mut prose = Prose::new();
        let chars: Vec<char> = answer.chars().collect();
        let got: String = (chars.chunks(size))
            .map(|c| prose.feed(&c.iter().collect::<String>()))
            .collect();
        assert_eq!(got, want, "pieces of {size}");
    }
}

/// The shaders make a look of their preset: the pass added after its
/// passes, the knobs moved, the music moving what was asked on top of the
/// pass's own drives (a mix too), its mood among the tags; checked,
/// nothing of it is held or dropped.
#[test]
fn the_shaders_make_a_valid_look() {
    // "canvas" is the preset's already: it stays where it is.
    let answer = bad("\"add\": [\"grain\"]", "\"add\": [\"canvas\", \"grain\"]").replacen(
        "\"by\": \"chorus\", \"amount\": 0.6}]",
        "\"by\": \"chorus\", \"amount\": 0.6}, {\"pass\": \"grain\", \"knob\": \"mix\", \"by\": \"quiet\", \"amount\": -0.5}]",
        1,
    );
    let look = read(&answer, &song()).unwrap().look;
    assert_eq!(look.id, "folk+song");
    assert!(look
        .tags
        .ends_with(&["longing".to_owned(), "rain".to_owned()]));
    let passes: Vec<&str> = look.layers.iter().map(|l| l.pass.as_str()).collect();
    assert_eq!(passes, ["watercolor", "canvas", "liquid", "grain"]);
    let layer = |pass: &str| look.layers.iter().find(|l| l.pass == pass).unwrap();
    assert_eq!(layer("canvas").knobs.get("amount"), Some(&0.5));
    assert_eq!(layer("grain").knobs.get("size"), Some(&2.0));
    let liquid = layer("liquid").drive.as_ref().expect("asked");
    let amount: Vec<_> = liquid.iter().filter(|d| d.knob == "amount").collect();
    assert_eq!(
        amount.len(),
        1,
        "its own amount driver replaced: {liquid:?}"
    );
    assert_eq!(
        (amount[0].by, amount[0].amount),
        (jodel_looks::Driver::Chorus, 0.6)
    );
    let grain = layer("grain").drive.as_ref().expect("asked");
    assert!(grain
        .iter()
        .any(|d| d.knob == "mix" && d.by == jodel_looks::Driver::Quiet));
    let folk = jodel_looks::presets::named("folk").unwrap();
    assert_eq!(layer("canvas").drive, folk.layers[1].drive, "not asked");
    assert!(look.faults().is_empty());
    let checked = look.check();
    assert!(checked.notes.is_empty(), "{:?}", checked.notes);
    assert_eq!(checked.layers.len(), look.layers.len());
}

/// A shot is told to the image model by the name the model writes it
/// with.
#[test]
fn a_shot_is_named_as_the_model_writes_it() {
    for shot in [Shot::Wide, Shot::Medium, Shot::CloseUp] {
        let written = serde_json::to_string(&shot).unwrap();
        assert_eq!(written, format!("\"{}\"", shot.name()));
    }
}

/// Shaders over the budget are taken without their dearest passes (as the
/// catalogue tells the model), none of them kept: what went cost more
/// than anything that stayed; the look within the budget, nothing more
/// dropped.
#[test]
fn shaders_over_the_budget_lose_their_dearest_passes() {
    let asked = [
        "watercolor",
        "canvas",
        "liquid",
        "grain",
        "glow",
        "halation",
    ];
    let answer = bad(
        "\"add\": [\"grain\"]",
        "\"add\": [\"grain\", \"glow\", \"halation\"]",
    );
    let look = read(&answer, &song()).expect("taken").look;
    let kept: Vec<&str> = look.layers.iter().map(|l| l.pass.as_str()).collect();
    assert_eq!(kept, ["watercolor", "canvas", "liquid", "grain"]);
    let cost = |p: &str| jodel_looks::passes::named(p).unwrap().cost;
    let gone = asked.iter().filter(|p| !kept.contains(p));
    for g in gone {
        assert!(kept.iter().all(|&k| cost(g) >= cost(k)), "{g}");
    }
    assert!(look.faults().is_empty(), "{:?}", look.faults());
    assert!(look.check().dropped.is_empty());
}

/// A song from a file's tags and its lyrics: the title, artist and album;
/// each telling tag once (no technical ones, none empty); its length;
/// its sung lines only (no break, no backing line, none untimed), each
/// from its start to its last word's end, else its source's end, else
/// the next line's start, else a while after it.
#[test]
fn a_song_is_made_of_its_tags_and_sung_lines() {
    use jodel_lyrics::{Line as Sung, Lyrics, Source, Synced, Word};
    let timed = |text: &str, start: f64, end: f64| Word {
        start: Some(start),
        end: Some(end),
        ..Word::new(text)
    };
    let words = |text: &str| text.split(' ').map(Word::new).collect::<Vec<_>>();
    let mut backing = Sung::from_text("(ooh ooh)");
    backing.time = Some(14.0);
    let lyrics = Lyrics {
        lines: vec![
            Sung {
                words: vec![timed("I", 10.0, 10.4), timed("walked", 10.5, 12.0)],
                ..Sung::default()
            },
            Sung {
                is_break: true,
                time: Some(13.0),
                ..Sung::default()
            },
            backing,
            Sung {
                words: words("Carry me over"),
                synced: Some(Synced {
                    start: 20.0,
                    end: Some(24.0),
                }),
                ..Sung::default()
            },
            Sung {
                words: words("never timed"),
                ..Sung::default()
            },
            Sung {
                words: words("Carry me home"),
                time: Some(26.0),
                ..Sung::default()
            },
            Sung {
                words: words("the end"),
                time: Some(30.0),
                ..Sung::default()
            },
        ],
        source: Source::Lrclib,
        timing: Default::default(),
    };
    let info = jodel_playback::TrackInfo {
        path: "/music/harbour.flac".into(),
        container: jodel_playback::Container::Flac,
        codec: jodel_playback::Codec::Flac,
        rate: 48_000,
        channels: 2,
        bits: 16,
        float: false,
        bitrate: None,
        frames: Some(48_000 * 120),
        tags: jodel_playback::Tags {
            title: Some("Harbour Song".into()),
            artist: Some("The Example Band".into()),
            album: None,
            track_number: None,
            all: [
                ("ENCODER", "x 1.0"),
                ("DATE", "1994"),
                ("date", "2000"),
                ("Genre", "folk"),
                ("comment", " "),
            ]
            .map(|(k, v)| (k.to_owned(), v.to_owned()))
            .into(),
        },
        cover: None,
    };
    let song = Song::of(&info, &lyrics);
    assert_eq!(
        (
            song.title.as_str(),
            song.artist.as_str(),
            song.album.as_str()
        ),
        ("Harbour Song", "The Example Band", "")
    );
    assert_eq!(
        song.tags,
        [
            ("date".to_owned(), "1994".to_owned()),
            ("genre".to_owned(), "folk".to_owned())
        ]
    );
    assert_eq!(song.duration, 120.0);
    let lines: Vec<(f64, f64, &str)> = (song.lines.iter())
        .map(|l| (l.start, l.end, l.text.as_str()))
        .collect();
    assert_eq!(
        lines,
        [
            (10.0, 12.0, "I walked"),
            (20.0, 24.0, "Carry me over"),
            (26.0, 30.0, "Carry me home"),
            (30.0, 30.0 + rules::UNTIMED_LINE_SECONDS, "the end"),
        ]
    );
}
