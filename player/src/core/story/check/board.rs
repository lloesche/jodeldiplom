//! A storyboard, not a portrait session: the scenes drawn looked at
//! together ([`Read::board`]), asked once in the follow-up (all that was
//! found in one question, so the rewrites never pull apart) where more of
//! them than [`BOARD`] allows show one character near enough to read as a
//! portrait, are close-ups, or share one shot and angle; the scenes the
//! model rewrites merged ([`Read::redraw`]). Never a refusal: what is not
//! answered stays as it was.

use serde_json::{Map, Value};

use super::super::cast;
use super::super::follow::{self, figures, Finding, Question};
use super::super::repair::{number, text};
use super::super::rules::BOARD;
use super::super::{Angle, Shot};
use super::fields;
use super::Read;

/// The words of a close-up's prompt that say it shows a face.
const FACE: [&str; 7] = [
    "face",
    "faces",
    "eyes",
    "profile",
    "expression",
    "gaze",
    "lips",
];

/// The words of a close-up's prompt that say it shows a detail of a
/// person, not the face.
const DETAIL: [&str; 9] = [
    "hand", "hands", "fingers", "fist", "feet", "foot", "shoes", "boots", "back",
];

/// Whether a character in `scene` could read as its portrait: a medium
/// shot at eye level or from low; a close-up from there, unless its
/// prompt shows a detail (hands, feet) and no face. From behind, over the
/// shoulder, through a window, from high or far, it cannot.
fn portrait(scene: &super::Drafted) -> bool {
    let facing = matches!(scene.angle, Angle::EyeLevel | Angle::Low);
    let words: Vec<String> = (scene.prompt.split(|c: char| !c.is_alphanumeric()))
        .map(str::to_lowercase)
        .collect();
    let says = |all: &[&str]| words.iter().any(|w| all.contains(&w.as_str()));
    match scene.shot {
        Shot::Medium => facing,
        Shot::CloseUp => facing && (says(&FACE) || !says(&DETAIL)),
        Shot::Wide => false,
    }
}

/// The number a rewritten scene's fields give it: none where it names
/// none (no `scene`, `number`, `place` or `index` field), else the number
/// it names (from 1, as the model numbered its scenes), none where that
/// is no number.
fn place(fields: &Map<String, Value>) -> Option<Option<usize>> {
    let key = ["scene", "number", "place", "index"]
        .iter()
        .find_map(|key| fields.get(*key))?;
    Some(match key {
        Value::String(s) => figures(s),
        other => number(Some(other))
            .filter(|n| *n >= 1.0 && n.fract() == 0.0)
            .map(|n| n as usize),
    })
}

impl Read {
    /// The scenes drawn (none shown again), by their places.
    pub(super) fn drawn(&self) -> Vec<usize> {
        (0..self.scenes.len())
            .filter(|&k| self.scenes[k].same_as.is_none())
            .collect()
    }

    /// `scenes` (their places) each with the model's own number for it.
    fn numbered(&self, scenes: &[usize]) -> Vec<(usize, usize)> {
        scenes.iter().map(|&k| (k, self.scenes[k].at)).collect()
    }

    /// The storyboard's question, where it has one ([`follow::redraw`]:
    /// all found asked at once): the characters who could read as a
    /// portrait too often ([`Read::posed`]), the close-ups ([`Read::close`])
    /// and one camera ([`Read::camera`], unless it is the close-ups again).
    pub(super) fn board(&self) -> Option<Question> {
        let drawn = self.drawn();
        let close = self.close(&drawn);
        let camera =
            (self.camera(&drawn)).filter(|c| close.as_ref().is_none_or(|f| f.scenes != c.scenes));
        let mut found = self.posed(&drawn);
        found.extend(close);
        found.extend(camera);
        (!found.is_empty()).then(|| follow::redraw(&found, drawn.len()))
    }

    /// Each character near enough to read as a portrait ([`portrait`]) in
    /// more of the `drawn` scenes than [`BOARD`] allows: as many of them
    /// asked to be rewritten as leave [`crate::core::story::rules::Board::leave_cast`].
    fn posed(&self, drawn: &[usize]) -> Vec<Finding> {
        let d = drawn.len();
        let near = |k: usize, name: &String| {
            let s = &self.scenes[k];
            portrait(s) && (cast::cast(&s.prompt, &s.characters, &self.characters).0).contains(name)
        };
        (self.characters.iter())
            .filter_map(|c| {
                let near: Vec<usize> = (drawn.iter().copied())
                    .filter(|&k| near(k, &c.name))
                    .collect();
                let k = BOARD.rewrite(near.len(), d, BOARD.leave_cast);
                (BOARD.asks(near.len(), d))
                    .then(|| follow::posed(&c.name, &self.numbered(&near), k))
            })
            .collect()
    }

    /// The close-ups among the `drawn` scenes, where more than [`BOARD`]
    /// allows.
    fn close(&self, drawn: &[usize]) -> Option<Finding> {
        let close: Vec<usize> = (drawn.iter().copied())
            .filter(|&k| self.scenes[k].shot == Shot::CloseUp)
            .collect();
        let k = BOARD.rewrite(close.len(), drawn.len(), BOARD.leave_shots);
        (BOARD.asks(close.len(), drawn.len())).then(|| follow::close(&self.numbered(&close), k))
    }

    /// The `drawn` scenes of the shot and angle most of them share (the
    /// first seen of equals), where more than [`BOARD`] allows.
    fn camera(&self, drawn: &[usize]) -> Option<Finding> {
        let mut tally: Vec<((Shot, Angle), Vec<usize>)> = Vec::new();
        for &k in drawn {
            let camera = (self.scenes[k].shot, self.scenes[k].angle);
            match tally.iter_mut().find(|(c, _)| *c == camera) {
                Some((_, scenes)) => scenes.push(k),
                None => tally.push((camera, vec![k])),
            }
        }
        let ((shot, angle), same) = (tally.into_iter()).reduce(|most, next| {
            if next.1.len() > most.1.len() {
                next
            } else {
                most
            }
        })?;
        let k = BOARD.rewrite(same.len(), drawn.len(), BOARD.leave_shots);
        (BOARD.asks(same.len(), drawn.len()))
            .then(|| follow::camera(shot, angle, &self.numbered(&same), k))
    }

    /// The scenes rewritten in `answer` (the list of every rewritten scene
    /// in the patch: [`follow::answers`]; none: not answered), merged,
    /// those of their fields each has (its shot, angle, characters, what
    /// it shows and its prompt). Each is the scene drawn its number names
    /// (the model's own); one with no number the next of `asked` not yet
    /// rewritten; one whose number names no scene drawn (or one rewritten
    /// already) is left out. What was taken, in words.
    pub(super) fn redraw(&mut self, asked: &[usize], answer: Option<&Value>) -> String {
        let items = answer
            .and_then(Value::as_array)
            .map(Vec::as_slice)
            .unwrap_or_default();
        let drawn = self.drawn();
        let mut done: Vec<usize> = Vec::new();
        let mut said = Vec::new();
        for fields in items.iter().filter_map(Value::as_object) {
            let scene = match place(fields) {
                None => asked.iter().copied().find(|k| !done.contains(k)),
                Some(n) => (drawn.iter().copied())
                    .find(|&k| Some(self.scenes[k].at + 1) == n && !done.contains(&k)),
            };
            let Some(k) = scene else {
                let named = fields.get("scene").map_or("none".into(), Value::to_string);
                said.push(format!(
                    "a rewrite of scene {named} left out (no scene drawn left)"
                ));
                continue;
            };
            done.push(k);
            let mut notes = Vec::new();
            let s = &mut self.scenes[k];
            if let Some(shot) = text(fields.get("shot")) {
                s.shot = fields::shot(Some(&shot), &mut notes);
            }
            if let Some(angle) = text(fields.get("angle")) {
                s.angle = fields::angle(Some(&angle), &mut notes);
            }
            if fields.contains_key("characters") {
                s.characters = fields::seen(fields.get("characters"), &self.characters, &mut notes);
            }
            for (field, to) in [("shows", &mut s.shows), ("prompt", &mut s.prompt)] {
                if let Some(t) = text(fields.get(field)).filter(|t| !t.is_empty()) {
                    *to = t;
                }
            }
            said.push(format!(
                "scene {k} rewritten ({} shot, {}): {:?}",
                s.shot.name(),
                s.angle.name(),
                s.prompt
            ));
            self.notes.extend(notes);
        }
        if said.is_empty() {
            return "no scene rewritten".to_owned();
        }
        said.join("; ")
    }
}

#[cfg(test)]
mod tests {
    use serde_json::Value;

    use super::super::super::follow::{self, What};
    use super::super::super::tests::{song, GOOD};
    use super::super::super::{Angle, Plan, Shot};
    use super::super::{fields, read};

    /// [`GOOD`] with every angle left out (each eye level then).
    fn no_angles(answer: &str) -> String {
        ["high", "eye level", "bird's-eye", "low"]
            .iter()
            .fold(answer.to_owned(), |a, angle| {
                a.replace(&format!("\"angle\": \"{angle}\", "), "")
            })
    }

    /// [`GOOD`] with its narrator near (medium, at eye level or from low)
    /// in three of its five scenes drawn.
    fn posed() -> String {
        GOOD.replacen(
            "\"shot\": \"wide\", \"angle\": \"bird's-eye\"",
            "\"shot\": \"medium\", \"angle\": \"eye level\"",
            1,
        )
        .replacen(
            "a small rowing boat crossing",
            "narrator rowing a small boat",
            1,
        )
        .replacen(
            "\"shot\": \"wide\", \"angle\": \"low\"",
            "\"shot\": \"medium\", \"angle\": \"low\"",
            1,
        )
        .replacen(
            "Lonely lighthouse on",
            "narrator walking to a lonely lighthouse on",
            1,
        )
    }

    /// The plan of `answer` with the storyboard's question answered by
    /// `patch`, and what was mended.
    fn rewritten(answer: &str, patch: &str) -> (Plan, Vec<String>) {
        let mut read = read(answer, &song()).expect("something to draw");
        let questions = read.questions();
        let mut notes = Vec::new();
        let answers = follow::answers(patch, &questions, &mut notes);
        read.settle(&questions, &answers);
        read.plan()
    }

    /// A character near enough to read as a portrait in more than half
    /// the scenes drawn is asked once (scenes from behind, far or from
    /// high not counted), as many to be rewritten as leave half, the
    /// scenes named by the model's own numbers. Every rewrite in the patch
    /// is merged, however the model laid them out (a key each, a bare
    /// list): each the scene its number names, one with no number the next
    /// asked, one naming no scene drawn left out; the others stay as they
    /// were.
    #[test]
    fn a_portrait_session_is_asked_to_be_redrawn_and_every_rewrite_merged() {
        let read = read(&posed(), &song()).expect("something to draw");
        let questions = read.questions();
        let whats: Vec<&What> = questions.iter().map(|q| &q.what).collect();
        let scenes = vec![1, 2, 4];
        assert_eq!(whats, [&What::Redraw { scenes }]);
        let words = follow::ask(&questions);
        assert!(
            words.contains(
                "1. Of the 5 scenes drawn, the storyboard reads as a portrait session: Scenes 2, 3 \
                 and 5 show \"narrator\" near and facing us (1 of them could show the place, an \
                 object or an action, \"narrator\" small in the frame, seen from behind, or not \
                 there). Choose 1 of these scenes and rewrite them."
            ),
            "{words}"
        );
        let three = r#"{"scene": 3, "shot": "wide", "angle": "aerial view", "characters": [], "shows": "The boat drifts out alone.", "prompt": "An empty rowing boat drifting on the dark sea at nightfall, distant lights"}"#;
        let stray = r#"{"scene": 12, "prompt": "A gull on a post"}"#;
        let bare = r#"{"shows": "The catch comes in.", "prompt": "The quay at dusk, fishermen hauling crates past the boats, rain on the stones"}"#;
        for patch in [
            format!("{{\"1\": {three}, \"2\": {stray}, \"3\": {bare}}}"),
            format!("[{three}, {stray}, {bare}]"),
        ] {
            let (plan, notes) = rewritten(&posed(), &patch);
            let three = &plan.scenes[2];
            assert_eq!((three.shot, three.angle), (Shot::Wide, Angle::BirdsEye));
            assert_eq!(three.shows, "The boat drifts out alone.");
            assert!(three.characters.is_empty());
            let two = &plan.scenes[1];
            assert!(two.prompt.starts_with("The quay at dusk"), "{patch}");
            assert!(two.characters.is_empty());
            assert_eq!(
                (two.shot, two.angle),
                (Shot::Medium, Angle::EyeLevel),
                "as it was"
            );
            assert_eq!(plan.scenes[4].characters, ["narrator"], "not rewritten");
            assert!(plan.scenes.iter().all(|s| s.prompt != "A gull on a post"));
            let said = |s: &str| notes.iter().any(|n| n.contains(s));
            assert!(
                said("the angle \"aerial view\" taken as \"bird's-eye\""),
                "{notes:?}"
            );
            assert!(
                said("question 1: scene 2 rewritten (wide shot, bird's-eye)"),
                "{notes:?}"
            );
            assert!(said("a rewrite of scene 12 left out"), "{notes:?}");
        }
    }

    /// Close-ups that dominate are asked (as many as leave a third), one
    /// camera that is the close-ups again not twice; one shot and angle of
    /// more than half asked as such. Not answered, the scenes stay as they
    /// were: never a refusal.
    #[test]
    fn close_ups_and_one_camera_are_asked_and_never_refused() {
        let close = no_angles(&GOOD.replace("\"shot\": \"wide\"", "\"shot\": \"close-up\""));
        let mut read = read(&close, &song()).expect("something to draw");
        let questions = read.questions();
        let whats: Vec<&What> = questions.iter().map(|q| &q.what).collect();
        assert_eq!(
            whats,
            [&What::Redraw {
                scenes: vec![0, 2, 3, 4]
            }]
        );
        let words = follow::ask(&questions);
        assert!(words.contains(
            "Scenes 1, 3, 4 and 5 are all close-ups (3 of them could be a wide shot of the place \
             or a medium shot of an action). Choose 3 of these scenes and rewrite them."
        ));
        read.settle(&questions, &[Some(Value::Null)]);
        let (plan, notes) = read.plan();
        assert_eq!(plan.scenes.len(), 6);
        assert!(plan.scenes.iter().all(|s| s.angle == Angle::EyeLevel));
        assert_eq!(plan.scenes[0].shot, Shot::CloseUp);
        assert!(
            notes.iter().any(|n| n == "question 1: no scene rewritten"),
            "{notes:?}"
        );
        // One camera: four of five medium shots at eye level.
        let medium = no_angles(&GOOD.replace("\"shot\": \"wide\"", "\"shot\": \"medium\""));
        let one_camera = super::super::read(&medium, &song()).expect("something to draw");
        let board = one_camera.board().expect("asked");
        assert!(board.asks.contains(
            "Scenes 1, 2, 3 and 5 are all medium shots at the angle \"eye level\" (3 of them"
        ));
    }

    /// A good storyboard is asked nothing; neither is one with exactly half
    /// its scenes close-ups, nor a short one where two of its three are (a
    /// character in two of three scenes is no portrait session).
    #[test]
    fn a_storyboard_half_or_short_is_asked_nothing() {
        let good = read(GOOD, &song()).expect("something to draw");
        assert_eq!(good.board(), None);
        let half = GOOD
            .replacen("\"shot\": \"wide\"", "\"shot\": \"close-up\"", 1)
            .replacen(
                "{\"first_line\": 10, \"section\": \"chorus\", \"same_as\": 4}",
                "{\"first_line\": 10, \"section\": \"chorus\", \"shot\": \"close-up\", \"angle\": \"low\", \"shows\": \"A rope.\", \"prompt\": \"A wet rope on a bollard\"}",
                1,
            );
        let half = read(&half, &song()).expect("something to draw");
        assert_eq!(half.drawn().len(), 6);
        assert_eq!(half.board(), None, "three close-ups of six");
        let cut = &GOOD[..GOOD.find("Close view of father").unwrap()];
        let short = cut
            .replacen("\"shot\": \"wide\"", "\"shot\": \"close-up\"", 1)
            .replacen("\"shot\": \"medium\"", "\"shot\": \"close-up\"", 1);
        let short = read(&short, &song()).expect("something to draw");
        assert_eq!(short.board(), None, "two close-ups of three");
        // A close-up of the narrator's hands is no portrait of her.
        let hands = posed()
            .replacen(
                "\"shot\": \"medium\", \"angle\": \"low\"",
                "\"shot\": \"close-up\", \"angle\": \"low\"",
                1,
            )
            .replacen(
                "narrator walking to a lonely lighthouse on",
                "narrator's hands on the door of a lonely lighthouse on",
                1,
            );
        let hands = read(&hands, &song()).expect("something to draw");
        assert_eq!(hands.board(), None, "near in two of five");
    }

    /// An angle is read by its name, else by the whole words it says, else
    /// eye level (said).
    #[test]
    fn an_angle_is_read_as_it_was_meant() {
        let mut notes = Vec::new();
        for angle in Angle::ALL {
            assert_eq!(fields::angle(Some(angle.name()), &mut notes), angle);
            let written = serde_json::to_string(&angle).unwrap();
            assert_eq!(written, format!("\"{}\"", angle.name()));
        }
        assert!(notes.is_empty());
        for (written, angle) in [
            ("Worm's-eye view", Angle::Low),
            ("top-down", Angle::BirdsEye),
            ("back view", Angle::FromBehind),
            ("over-the-shoulder shot", Angle::OverTheShoulder),
            ("through the café window", Angle::ThroughAWindow),
            ("high angle", Angle::High),
            ("dutch tilt", Angle::EyeLevel),
            ("against a yellow background", Angle::EyeLevel),
        ] {
            assert_eq!(fields::angle(Some(written), &mut notes), angle, "{written}");
        }
        assert_eq!(fields::angle(None, &mut notes), Angle::EyeLevel);
        assert_eq!(notes.len(), 8);
    }
}
