//! What the image model (FLUX.2 [klein] 4B) is asked for a plan's
//! pictures ([`pictures`]): a portrait of each character, from its looks
//! in the plan's art style, then a picture of each scene drawn (a scene
//! shown again takes the picture of the one it shows), from its prompt
//! with the characters named by their looks, its shot and the style, the
//! portraits of the characters seen as its reference pictures. Every
//! prompt ends asking for one single picture ([`ONE_PICTURE`]) and no text
//! at all ([`NO_TEXT`]): the model is distilled and takes no negative
//! prompt, so the wishes go in the prompt itself. The no-text wording is
//! the image models' bake-off's, the one klein won with.

use super::{Character, Plan, Scene};

/// What every prompt asks before [`NO_TEXT`]: one picture, one moment.
/// Given reference pictures of the people in it, klein at times drew a
/// grid of them instead (a scene on the owner's Mac came out four panels
/// of one woman, its prompt asking for none).
pub const ONE_PICTURE: &str =
    "One single photograph of one moment: not a collage, grid, split screen or panels.";

/// What every prompt ends with: nothing written in the picture.
pub const NO_TEXT: &str = "Pure picture with no text anywhere: no writing, no letters, no words, \
                           no numbers, no captions, no logos, no watermark.";

/// A portrait's size: square, the size klein draws best at (the
/// bake-off's), its head then taken as a scene's reference picture.
pub const PORTRAIT: (usize, usize) = (1024, 1024);

/// A scene's size: wide, as the bake-off drew them.
pub const SCENE: (usize, usize) = (1344, 768);

// Both as the image model takes them: sides multiples of 16.
const _: () = assert!(PORTRAIT.0.is_multiple_of(16) && PORTRAIT.1.is_multiple_of(16));
const _: () = assert!(SCENE.0.is_multiple_of(16) && SCENE.1.is_multiple_of(16));

/// The most characters' portraits a scene takes as reference pictures
/// (the first seen).
pub const MOST_REFERENCES: usize = 3;

/// The plan's own style keywords added after the style's words (the
/// first ones: the style's words lead).
const STYLE_KEYWORDS: usize = 2;

/// The parts of a character's looks that name it in a prompt (the first:
/// who it is and its age).
const SHORT_LOOKS: usize = 2;

/// What the image model is asked for one picture: its prompt, its size,
/// and the portraits it takes as its reference pictures, in order (a
/// scene's, by their places among the portraits; a portrait has none).
#[derive(Clone, Debug, PartialEq)]
pub struct Ask {
    /// Its prompt.
    pub prompt: String,
    /// Its size, width and height.
    pub size: (usize, usize),
    /// The portraits it takes.
    pub refs: Vec<usize>,
}

/// A plan's pictures: a portrait of each of its characters, in their
/// order, a picture of each of its scenes drawn, in theirs, and which of
/// those each scene shows.
#[derive(Clone, Debug, PartialEq)]
pub struct Pictures {
    /// The portraits.
    pub portraits: Vec<Ask>,
    /// The scenes' pictures.
    pub scenes: Vec<Ask>,
    /// The picture each scene shows (its place among `scenes`).
    pub shown: Vec<usize>,
}

/// The pictures of `plan`.
pub fn pictures(plan: &Plan) -> Pictures {
    let style = style(plan);
    let portraits = (plan.characters.iter())
        .map(|c| Ask {
            prompt: format!(
                "{style}, portrait of {}, head and shoulders, facing the viewer, plain soft background. \
                 {ONE_PICTURE} {NO_TEXT}",
                clause(&c.looks)
            ),
            size: PORTRAIT,
            refs: Vec::new(),
        })
        .collect();
    let mut scenes = Vec::new();
    let mut shown: Vec<usize> = Vec::new();
    for s in &plan.scenes {
        match s.same_as.and_then(|j| shown.get(j).copied()) {
            Some(picture) => shown.push(picture),
            None => {
                shown.push(scenes.len());
                scenes.push(scene(plan, &style, s));
            }
        }
    }
    Pictures {
        portraits,
        scenes,
        shown,
    }
}

/// The picture of `scene`, in `style`: its prompt with the characters
/// named by their looks, its shot, the style, the looks of those seen,
/// then which reference picture each of the first of them is.
fn scene(plan: &Plan, style: &str, scene: &Scene) -> Ask {
    let seen: Vec<(usize, &Character)> = (scene.characters.iter())
        .filter_map(|name| (plan.characters.iter().enumerate()).find(|(_, c)| c.name == *name))
        .collect();
    let named = named(&scene.prompt, plan);
    let shot = format!("{} shot", scene.shot.name());
    let mut parts = vec![style, named.as_str(), shot.as_str()];
    parts.extend(seen.iter().map(|(_, c)| c.looks.as_str()));
    let described: Vec<&str> = (parts.into_iter().map(clause))
        .filter(|p| !p.is_empty())
        .collect();
    let refs: Vec<(usize, &Character)> = seen.into_iter().take(MOST_REFERENCES).collect();
    let mut sentences = vec![format!("{}.", described.join(", "))];
    sentences.extend(refs.iter().enumerate().map(|(k, (_, c))| {
        format!(
            "The {} looks exactly like the person in image {}.",
            short(&c.looks),
            k + 1
        )
    }));
    sentences.push(ONE_PICTURE.to_owned());
    sentences.push(NO_TEXT.to_owned());
    Ask {
        prompt: sentences.join(" "),
        size: SCENE,
        refs: refs.into_iter().map(|(k, _)| k).collect(),
    }
}

/// The plan's style in words: its style's own words first, then the
/// first of the plan's keywords.
fn style(plan: &Plan) -> String {
    let keywords: Vec<&str> = (plan.art.style_keywords.split(','))
        .map(str::trim)
        .filter(|k| !k.is_empty())
        .take(STYLE_KEYWORDS)
        .collect();
    let mut style = vec![plan.art.style.words()];
    style.extend(keywords);
    style.join(", ")
}

/// `text` without the spaces and full stops at its ends.
fn clause(text: &str) -> &str {
    text.trim().trim_end_matches('.').trim_end()
}

/// The article a name is said with ("the"), taken with it where it comes
/// right before it.
const ARTICLE: &str = "the ";

/// The words that come before a noun and take the place of "the": an
/// article or a possessive.
const DETERMINERS: [&str; 10] = [
    "the", "a", "an", "my", "your", "his", "her", "its", "our", "their",
];

/// A character as a prompt names it, from `looks`: its first parts
/// ([`SHORT_LOOKS`]), in lower case, without a determiner before.
fn short(looks: &str) -> String {
    let parts: Vec<&str> = (clause(looks).split(','))
        .map(str::trim)
        .take(SHORT_LOOKS)
        .collect();
    let short = parts.join(", ").to_lowercase();
    let first = short.split(' ').next().unwrap_or_default();
    if DETERMINERS.contains(&first) {
        short[first.len()..].trim_start().to_owned()
    } else {
        short
    }
}

/// `prompt` with each character named by its short looks ([`short`]).
fn named(prompt: &str, plan: &Plan) -> String {
    let names: Vec<(&str, String)> = (plan.characters.iter())
        .map(|c| (c.name.as_str(), short(&c.looks)))
        .collect();
    replace_names(prompt, &names)
}

/// `text` with every whole word of a name of `names` (ASCII case ignored;
/// the longest name first where two start at once) replaced by what it is
/// paired with, in one pass (what replaces a name is never searched for
/// another): after a determiner of its own ([`DETERMINERS`]) as it is,
/// else with "the" before it (the [`ARTICLE`] right before it, in any
/// case, taken with it).
fn replace_names(text: &str, names: &[(&str, String)]) -> String {
    let word = |c: Option<char>| c.is_some_and(|c| c.is_alphanumeric() || c == '_');
    // ASCII lower case keeps every byte where it was.
    let lower = text.to_ascii_lowercase();
    let mut names: Vec<(String, &str)> = (names.iter())
        .map(|(name, by)| (name.to_ascii_lowercase(), by.as_str()))
        .collect();
    names.sort_by_key(|(name, _)| std::cmp::Reverse(name.len()));
    let (mut out, mut from, mut at) = (String::new(), 0, 0);
    while let Some(c) = text[at..].chars().next() {
        let starts = !word(text[..at].chars().next_back());
        let found = (names.iter()).find(|(name, _)| {
            starts && lower[at..].starts_with(name) && !word(text[at + name.len()..].chars().next())
        });
        let Some((name, by)) = found else {
            at += c.len_utf8();
            continue;
        };
        // "The" right before is said as "the"; another determiner stays.
        let article =
            lower[..at].ends_with(ARTICLE) && !word(text[..at - ARTICLE.len()].chars().next_back());
        let before = lower[..at]
            .trim_end()
            .rsplit(|c: char| !c.is_alphanumeric())
            .next();
        let determined = !article && before.is_some_and(|w| DETERMINERS.contains(&w));
        out.push_str(&text[from..if article { at - ARTICLE.len() } else { at }]);
        if !determined {
            out.push_str("the ");
        }
        out.push_str(by);
        at += name.len();
        from = at;
    }
    out.push_str(&text[from..]);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::story::tests::{planned, GOOD};

    /// A portrait of each character, then a picture of each scene drawn
    /// (one shown again takes the picture it shows), each with its size
    /// and the portraits of those seen; the characters named by their
    /// looks, the style's words first, and every prompt asking for no
    /// text.
    #[test]
    fn a_plan_s_pictures_are_its_portraits_and_its_scenes() {
        let plan = planned(GOOD);
        let Pictures {
            portraits,
            scenes,
            shown,
        } = pictures(&plan);
        assert_eq!(portraits.len(), plan.characters.len());
        assert_eq!(plan.scenes[5].same_as, Some(2), "the chorus again");
        assert_eq!(scenes.len(), plan.scenes.len() - 1);
        assert_eq!(shown, [0, 1, 2, 3, 4, 2]);
        assert_eq!(portraits[0].size, PORTRAIT);
        assert_eq!(
            portraits[0].prompt,
            format!(
                "watercolor painting, soft washes, paper texture, loose brush strokes, watercolor, \
                 soft washes, portrait of Woman, late 20s, slender, long dark hair, yellow raincoat, \
                 head and shoulders, facing the viewer, plain soft background. {ONE_PICTURE} \
                 {NO_TEXT}"
            )
        );
        let two = &scenes[1];
        assert_eq!((two.size, &two.refs), (SCENE, &vec![0, 1]));
        assert_eq!(
            two.prompt,
            format!(
                "watercolor painting, soft washes, paper texture, loose brush strokes, watercolor, \
                 soft washes, Rain-soaked quay in the afternoon, the man, late 20s running toward \
                 the woman, late 20s with his coat open, both laughing, puddles in front, low \
                 clouds over the sea, medium shot, Woman, late 20s, slender, long dark hair, \
                 yellow raincoat, Man, late 20s, tall, curly red hair, open navy coat. The woman, \
                 late 20s looks exactly like the person in image 1. The man, late 20s looks \
                 exactly like the person in image 2. {ONE_PICTURE} {NO_TEXT}"
            )
        );
        // A scene with no one in it takes no portrait.
        assert!(scenes[2].refs.is_empty());
        let every = portraits.iter().chain(&scenes);
        let one_and_no_text = format!("{ONE_PICTURE} {NO_TEXT}");
        assert!(every
            .into_iter()
            .all(|a| a.prompt.ends_with(&one_and_no_text)));
    }

    /// A name is replaced as a whole word in any case, "the" before it but
    /// after a determiner of its own, never inside another word, and what
    /// replaces it is not searched again.
    #[test]
    fn a_name_is_replaced_as_a_whole_word() {
        let names = [
            ("father", "old man".to_owned()),
            ("man", "boy".to_owned()),
            ("you", "girl".to_owned()),
        ];
        assert_eq!(
            replace_names("Father mends; The father's nets; fathers", &names),
            "the old man mends; the old man's nets; fathers"
        );
        assert_eq!(
            replace_names("you and your man, été, a man", &names),
            "the girl and your boy, été, a boy"
        );
        assert_eq!(
            short("A woman in her 20s, slender, dark hair."),
            "woman in her 20s, slender"
        );
    }
}
