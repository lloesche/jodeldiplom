//! A song's story scenes put where the story's drawing step leaves them
//! and made ready as the player makes them: the scenes of `FOLDER` (its
//! `scenes.txt` and pictures, as a drawing step made them elsewhere)
//! copied into the song's folder in the data folder's `scenes/` (named by
//! the song's key: `SONG` is only read), each picture taken apart as the
//! story's last step does ([`take_apart`], its parts kept by its bytes in
//! `scene-parts/`; run again, read back), then made ready as the backdrop
//! takes it ([`prepare`]). For each scene it prints how it moves (the
//! warp, or its layers and their cuts), its heads, where its subject is,
//! and how long each step took (the depth and the matte on the GPU, the
//! faces and the split on the processor). Then the song is asked of the
//! song playing's story as the player asks it when the song plays, and
//! the story it holds for the app is shown to be that one. The models are
//! the scenes' depth and cut-out, from the data folder's `models` (or
//! `--models DIR`); `--data DIR` for another data folder than the usual.
//!
//! ```text
//! cargo run --release -p jodeldiplom --example story_scenes -- SONG FOLDER [--data DIR] [--models DIR]
//! ```

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use jodeldiplom::store::dirs::Data;
use jodeldiplom::work::scenes::{
    picture_hash, prepare, song_key, take_apart, GpuNets, PlayingStory, Scene, SceneModels,
    SceneStore, Scenes, Split, TakenApart, DEPTH_FILE, MATTE_FILE,
};
use jodeldiplom::work::stages::{self, Stage};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let song = PathBuf::from(args.first().expect("SONG"));
    let source = PathBuf::from(args.get(1).expect("FOLDER"));
    let after = |flag: &str| args.iter().skip_while(|a| *a != flag).nth(1);
    let data = after("--data").map_or_else(
        || Data::chosen(None).expect("a data folder"),
        |d| Data(PathBuf::from(d)),
    );
    let models = after("--models").map_or_else(|| data.models(), PathBuf::from);
    let store = || SceneStore::new(data.scenes(), data.scene_parts());
    // The scenes copied in, as the drawing step would leave them.
    let made = Scenes::read(&source).expect("the folder's scenes.txt");
    let dir = store().folder(&song_key(&song));
    std::fs::create_dir_all(&dir).expect("the song's folder");
    let scenes = Scenes {
        made: made.made.clone(),
        scenes: made
            .scenes
            .iter()
            .map(|s| {
                let name = s.picture.file_name().expect("a picture's name");
                let bytes = std::fs::read(&s.picture).expect("the picture");
                durable::write_atomic(&dir.join(name), &bytes).expect("the picture copied");
                Scene {
                    picture: dir.join(name),
                    ..s.clone()
                }
            })
            .collect(),
        look: made.look.clone(),
    };
    scenes.write(&dir).expect("the scenes described");
    println!("{} scenes in {}", scenes.scenes.len(), dir.display());
    let nets = GpuNets::new(SceneModels {
        depth: models.join(DEPTH_FILE),
        matte: models.join(MATTE_FILE),
    });
    let started = Instant::now();
    let found = store().scenes_of(&song).expect("the song's scenes");
    let taken = stages::run(&[Stage::Scenes], "the scenes", || {
        take_apart(&found, &store(), &nets, &|| true)
    });
    let Ok(TakenApart::All(took)) = taken else {
        panic!("taken apart: {taken:?}");
    };
    let story = prepare(&found, &store()).expect("made ready");
    let all = started.elapsed().as_secs_f64();
    drop(nets);
    let name = |p: &Path| {
        p.file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .into_owned()
    };
    for (i, (scene, t)) in found.scenes.iter().zip(&took).enumerate() {
        let ready = &story.scenes[i];
        let bytes = std::fs::read(&scene.picture).expect("the picture");
        let parts = store().load(&picture_hash(&bytes)).expect("kept");
        let split = match parts.split {
            Split::Warp => "warp".to_owned(),
            Split::Layers { cuts, .. } => format!("{cuts:?}"),
        };
        let took = t.map_or("kept".to_owned(), |t| t.to_string());
        println!(
            "{i:2} {}: {split}, {} heads, subject at ({:.2}, {:.2}): {took}",
            name(&scene.picture),
            parts.heads.len(),
            ready.focus[0],
            ready.focus[1],
        );
    }
    println!("{} scenes ready in {all:.1} s", story.scenes.len());
    // As the player: the song playing's story made ready, and held.
    let playing = PlayingStory::start(store(), Arc::new(|| {}));
    playing.want(&song);
    let started = Instant::now();
    let held = loop {
        if let Some(held) = playing.story() {
            break held;
        }
        assert!(
            started.elapsed() < Duration::from_secs(120),
            "no story held"
        );
        std::thread::sleep(Duration::from_millis(50));
    };
    assert_eq!(held.id, story.id);
    println!(
        "the song playing's story is held for the app: {} scenes, {:.1} s after the song was asked",
        held.scenes.len(),
        started.elapsed().as_secs_f64()
    );
}
