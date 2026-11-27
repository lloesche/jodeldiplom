//! A song's scenes as the story's drawing step leaves them in the song's
//! folder ([`super::folder`]): a picture a scene, each with its span in the
//! song, and the song's look over them, described in [`FILE`] with what
//! made them (a name the drawing step gives them: they are derived from
//! the song's plan, which stays the source).

use std::path::{Path, PathBuf};

use jodel_looks::Look;

use crate::store::cache;

/// The scenes' description's name in their folder.
pub const FILE: &str = "scenes.txt";

/// What starts the line of what made them.
const MADE: &str = "made ";
/// What starts the line of the look.
const LOOK: &str = "look ";
/// What starts a scene's line.
const SCENE: &str = "scene ";

/// A scene: when it shows and its picture.
#[derive(Clone, Debug, PartialEq)]
pub struct Scene {
    /// Where its span in the song starts, in seconds.
    pub start: f64,
    /// Where it ends.
    pub end: f64,
    /// Its picture's file.
    pub picture: PathBuf,
}

/// A song's scenes, in the song's order, the look over them (the song's
/// own, which the karaoke draws instead of its theme's), and what made
/// them, as their maker names it.
#[derive(Clone, Debug, PartialEq)]
pub struct Scenes {
    /// What made them.
    pub made: String,
    /// The scenes.
    pub scenes: Vec<Scene>,
    /// The look over them.
    pub look: Look,
}

impl Scenes {
    /// As kept in `dir` ([`FILE`]): the line of what made them, `made
    /// NAME`, the look's, `look JSON` (on one line), then a line a scene,
    /// `scene START END PICTURE`, the picture named as in `dir` (where it
    /// is there) or by its whole path.
    pub fn to_text(&self, dir: &Path) -> String {
        let look = serde_json::to_string(&self.look).expect("a look writes as JSON");
        let mut body = format!("{MADE}{}\n{LOOK}{look}\n", self.made);
        for s in &self.scenes {
            let picture = s.picture.strip_prefix(dir).unwrap_or(&s.picture);
            body += &format!("{SCENE}{:.3} {:.3} {}\n", s.start, s.end, picture.display());
        }
        cache::STORY_SCENES.wrap(&body)
    }

    /// The scenes of a [`Scenes::to_text`] kept in `dir`, their pictures
    /// there; none for a text of another version or with a line amiss.
    pub fn from_text(text: &str, dir: &Path) -> Option<Scenes> {
        let body = cache::STORY_SCENES.body(text)?;
        let mut lines = body.lines().filter(|l| !l.trim().is_empty());
        let made = lines.next()?.strip_prefix(MADE)?.to_owned();
        let look = Look::from_json(lines.next()?.strip_prefix(LOOK)?).ok()?;
        let scenes = lines
            .map(|l| {
                let (start, rest) = l.strip_prefix(SCENE)?.split_once(' ')?;
                let (end, picture) = rest.split_once(' ')?;
                Some(Scene {
                    start: start.parse().ok()?,
                    end: end.parse().ok()?,
                    picture: dir.join(picture),
                })
            })
            .collect::<Option<Vec<_>>>()?;
        Some(Scenes { made, scenes, look })
    }

    /// The scenes kept in `dir`, where there are.
    pub fn read(dir: &Path) -> Option<Scenes> {
        let file = crate::store::mover::locate(&dir.join(FILE));
        Scenes::from_text(&std::fs::read_to_string(file).ok()?, dir)
    }

    /// Keeps them in `dir` (the drawing step's last write: the pictures
    /// are there before).
    pub fn write(&self, dir: &Path) -> Result<(), String> {
        let path = crate::store::mover::current(&dir.join(FILE));
        durable::write_atomic(&path, self.to_text(dir).as_bytes())
            .map_err(|e| format!("{}: {e}", path.display()))
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// A look of one pass with a knob moved, as a story's plan has one.
    pub(crate) fn look() -> Look {
        let mut grain = jodel_looks::Layer::of("grain");
        grain.knobs.insert("amount".into(), 0.45);
        Look {
            id: "folk.song".into(),
            name: String::new(),
            about: String::new(),
            tags: vec!["rain".into()],
            layers: vec![grain],
        }
    }

    #[test]
    fn the_scenes_description_reads_back_as_written() {
        let dir = Path::new("story");
        let scenes = Scenes {
            made: "0123456789abcdef".into(),
            scenes: vec![
                Scene {
                    start: 1.5,
                    end: 9.25,
                    picture: dir.join("00.png"),
                },
                Scene {
                    start: 9.25,
                    end: 20.0,
                    picture: dir.join("a scene with spaces.png"),
                },
            ],
            look: look(),
        };
        let text = scenes.to_text(dir);
        assert!(text.contains("\nscene 1.500 9.250 00.png\n"), "{text}");
        assert_eq!(Scenes::from_text(&text, dir), Some(scenes.clone()));
        // Kept elsewhere, the pictures are read there.
        let moved = Scenes::from_text(&text, Path::new("elsewhere")).unwrap();
        assert_eq!(
            moved.scenes[0].picture,
            Path::new("elsewhere").join("00.png")
        );
        // A line amiss: none.
        assert_eq!(Scenes::from_text(&(text + "scene 3\n"), dir), None);
    }

    /// The drawing step's write, read back from the song's folder.
    #[test]
    fn scenes_written_to_a_folder_are_read_back_from_it() {
        let dir = std::env::temp_dir().join(format!("jodel-scenes-text-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let scenes = Scenes {
            made: "0123456789abcdef".into(),
            scenes: vec![Scene {
                start: 3.0,
                end: 12.5,
                picture: dir.join("00.png"),
            }],
            look: look(),
        };
        assert_eq!(Scenes::read(&dir), None);
        scenes.write(&dir).unwrap();
        assert_eq!(Scenes::read(&dir), Some(scenes));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
