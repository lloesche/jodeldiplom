//! The preview's players ([`crate::core::preview`] is what it is meant to
//! do): each file previewed plays on a player of its own, a guest on the
//! mixer ([`jodel_output::mixer`]), from a third in. It fades in from its
//! first audio while everything heard before (the song, the file before)
//! fades out from that very frame: one crossfade on the mixer's clock. At
//! the end the file fades out and the song back in. The song's player is
//! never touched; it runs on under it.
//!
//! At most two players at a time. A player faded all the way out is let
//! go of, and so is one whose file has ended or failed. A file asked for
//! takes the place of one not heard yet (still loading: silent, it goes at
//! once, [`Fader::withdraw`]: the mixer says whether its fade in has
//! begun after all, and then it is heard, and stays, so that what fell as
//! it rose is never left down), and waits while there are two (the one
//! heard and one fading out); only the latest file asked for waits. With
//! nothing open on the device, a second player waits until the first has
//! opened it, so that
//! it is offered the device's format and resamples to it. A player let go
//! of is shut out at the mixer at once ([`Fader::close`]), so with no
//! player left the next file opens the device at its own format.

use std::path::{Path, PathBuf};
use std::sync::mpsc::Receiver;
use std::time::Duration;

use jodel_output::mixer::{Fader, Mixer, Ramp, Role, Start, Strip};
use jodel_playback::{Command, Event, Player, PlayerOptions, Settings};

use crate::core::preview::SHARE;

/// How long each fade takes: a file in, what it replaces out, the song
/// back in.
pub const FADE: Duration = Duration::from_millis(200);
/// The most players at a time.
const VOICES: usize = 2;

/// A file previewed, on a player of its own.
struct Voice {
    player: Player,
    events: Receiver<Event>,
    fader: Fader,
    path: PathBuf,
}

impl Voice {
    /// Not heard yet: still at 0, its fade in waiting for its first audio.
    fn unheard(&self) -> bool {
        self.fader.gain() == 0.0 && !self.fader.settled()
    }

    /// Faded all the way out.
    fn gone(&self) -> bool {
        self.fader.settled() && self.fader.gain() == 0.0
    }

    fn fade_out(&self) {
        self.fader.set(Ramp::fade(0.0, FADE, Start::Now));
    }
}

/// Lets go of `voices`: their inputs close on the mixer at once, so none
/// can open the device again, and their players stop (on a thread of
/// their own: a file on a share can take a while to close).
fn let_go(voices: Vec<Voice>) {
    if voices.is_empty() {
        return;
    }
    for v in &voices {
        v.fader.close();
        let _ = v.player.send(Command::Stop);
    }
    std::thread::spawn(move || drop(voices));
}

/// The preview's players.
pub struct Previews {
    /// The song's fader.
    lead: Fader,
    /// The strip the players are on.
    strip: Strip,
    settings: Settings,
    /// Oldest first.
    voices: Vec<Voice>,
    /// The file asked for last, waiting for a place.
    waiting: Option<PathBuf>,
    /// Ended: done once its players are gone.
    ended: bool,
}

impl Previews {
    /// Previews over the song on `lead`, played on `strip` with `settings`
    /// (resampled to the device as it runs, no stems).
    pub fn new(lead: Fader, strip: Strip, settings: Settings) -> Previews {
        Previews {
            lead,
            strip,
            settings: Settings {
                resample: true,
                ..settings
            },
            voices: Vec::new(),
            waiting: None,
            ended: false,
        }
    }

    /// Plays `path` in place of what is heard, and of what is loading.
    pub fn play(&mut self, mixer: &Mixer, path: PathBuf) {
        self.ended = false;
        self.waiting = Some(path);
        // What the mixer last said is a period old: it decides.
        let (gone, kept) = std::mem::take(&mut self.voices)
            .into_iter()
            .partition(|v| v.unheard() && v.fader.withdraw());
        self.voices = kept;
        let_go(gone);
        self.start(mixer);
    }

    /// Nothing for now: what is heard fades out.
    pub fn silent(&mut self) {
        self.waiting = None;
        for v in &self.voices {
            v.fade_out();
        }
    }

    /// Over: what is heard fades out, the song back in.
    pub fn end(&mut self) {
        self.silent();
        self.lead.set(Ramp::fade(1.0, FADE, Start::Now));
        self.ended = true;
    }

    /// Whether it has ended and its players are gone.
    pub fn done(&self) -> bool {
        self.ended && self.voices.is_empty()
    }

    /// How many players there are.
    pub fn players(&self) -> usize {
        self.voices.len()
    }

    /// Takes in what the players said, lets go of those done, and starts
    /// the file waiting; returns what there is to tell.
    pub fn pump(&mut self, mixer: &Mixer) -> Vec<String> {
        let mut said = Vec::new();
        let (mut kept, mut done) = (Vec::new(), Vec::new());
        for v in std::mem::take(&mut self.voices) {
            let mut ended = false;
            while let Ok(event) = v.events.try_recv() {
                match event {
                    Event::Error { error, .. } => {
                        said.push(format!("Cannot preview {}: {error}", name(&v.path)));
                        ended = true;
                    }
                    Event::Notice { message, .. } => said.push(message),
                    // Played to its end, its output played out.
                    Event::QueueEnded => ended = true,
                    _ => {}
                }
            }
            if ended || v.gone() {
                done.push(v);
            } else {
                kept.push(v);
            }
        }
        self.voices = kept;
        let_go(done);
        self.start(mixer);
        said
    }

    /// The file waiting, on a player of its own, if there is room: it
    /// crossfades in from its first audio, from everything heard.
    fn start(&mut self, mixer: &Mixer) {
        // With nothing open, the first player opens the device: the next
        // one waits to be offered its format.
        let first_open = mixer.timebase().rate == 0 && !self.voices.is_empty();
        if self.voices.len() >= VOICES || first_open {
            return;
        }
        let Some(path) = self.waiting.take() else {
            return;
        };
        let (target, fader) = mixer.input(&self.strip, Role::Guest, 0.0);
        // Everything heard: those not heard yet were let go of when this
        // file was asked for.
        let out: Vec<&Fader> = self
            .voices
            .iter()
            .map(|v| &v.fader)
            .chain([&self.lead])
            .collect();
        // Set before the player opens its output: the fade is on what it
        // loads.
        mixer.crossfade(&fader, Start::FirstAudio, &out, FADE);
        let options = PlayerOptions {
            settings: self.settings,
            ..PlayerOptions::default()
        };
        let (player, events) = Player::new(Box::new(target), options);
        let _ = player.send(Command::Load(vec![path.clone()]));
        let _ = player.send(Command::PlayAt {
            index: 0,
            share: SHARE,
        });
        self.voices.push(Voice {
            player,
            events,
            fader,
            path,
        });
    }
}

fn name(path: &Path) -> String {
    path.file_name().map_or_else(
        || path.display().to_string(),
        |n| n.to_string_lossy().into(),
    )
}
