//! The rules between what moves the head on air: whether a jump, a key, a
//! map or a transition may start while something else moves it, who holds
//! the playing track's route, and the reason the user is given when a move
//! may not be made ([`Refusal::reason`], every one of them here). What the
//! head is doing is read in one place ([`Engine::situation`]), and the
//! rules are a table over it ([`Situation::allows`]); `player/INTERACTIONS.md`
//! tells them as the player has them.

use super::moving::Move;
use super::track::{Back, Track};
use super::Engine;

/// A move asked of the head on air.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Action {
    /// A jump within the playing track ([`crate::Command::Jump`]).
    Jump,
    /// Another key for the playing track now ([`crate::Command::Key`]).
    Key,
    /// The key asked for, on a track as it is installed or run into.
    KeyOnTrack,
    /// The playing track read along a map from the frame written next (a
    /// transition's glide).
    Map,
    /// A transition armed ([`crate::Command::Transition`]).
    Transition,
}

/// Why a move may not be made now.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Refusal {
    TransitionSetUp,
    KeyShifted,
    /// The playing track is on a map, or waits to be.
    TrackOnMap,
    /// A track is on a map (and not leaving it), or waits to be.
    MapUnderWay,
    CountIn,
    NoOutput,
    Channels(u16),
    OffRoute,
    NoLanes,
    RouteMeeting,
}

impl Refusal {
    /// The reason, as the user is told it (after "No jumps while", "The key
    /// stays as it is:" and the like).
    pub(super) fn reason(self) -> String {
        match self {
            Refusal::TransitionSetUp => "a transition is set up".into(),
            Refusal::KeyShifted => "the key is shifted".into(),
            Refusal::TrackOnMap => "the track is read along a time map".into(),
            Refusal::MapUnderWay => "a track is read along a time map".into(),
            Refusal::CountIn => "its count-in is playing".into(),
            Refusal::NoOutput => "no output is open".into(),
            Refusal::Channels(n) => {
                format!("it has {n} channels, and stems are made of two at most")
            }
            Refusal::OffRoute => "it is not on its own route".into(),
            Refusal::NoLanes => "the output takes no stems beside it".into(),
            Refusal::RouteMeeting => "its move to another route is meeting it".into(),
        }
    }
}

/// What the head on air is doing, as far as the rules care.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct Situation {
    /// A transition is set up (armed, or further).
    pub(super) transition: bool,
    /// A track plays.
    pub(super) playing: bool,
    /// The key it plays in.
    pub(super) key: i8,
    /// It is read along a map (`Some`), and whether it is leaving it.
    pub(super) on_map: Option<bool>,
    /// A move onto a map waits for its frame.
    pub(super) map_waiting: bool,
    /// A count-in's silence is being written before it.
    pub(super) count_in: bool,
    /// The lanes of the output open, if one is.
    pub(super) lanes: Option<u16>,
    /// The channels it is played in.
    pub(super) channels: u16,
    /// It plays its own route as the settings make it, or a map: not a
    /// switch's new route decoded ahead.
    pub(super) own_route: bool,
    /// A move to another route is meeting it.
    pub(super) route_meeting: bool,
}

impl Situation {
    /// Whether `action` may be made now, or why not: the rules, a line
    /// each.
    pub(super) fn allows(&self, action: Action) -> Result<(), Refusal> {
        match action {
            Action::Jump if self.transition => Err(Refusal::TransitionSetUp),
            Action::Jump if self.key != 0 && self.on_map.is_some() => Err(Refusal::KeyShifted),
            Action::Jump if self.on_map.is_some() || self.map_waiting => Err(Refusal::TrackOnMap),
            Action::Jump => Ok(()),
            Action::Key if self.transition => Err(Refusal::TransitionSetUp),
            Action::Key if !self.playing => Ok(()),
            Action::Key | Action::KeyOnTrack => self.map_ready(),
            Action::Map if self.count_in => Err(Refusal::CountIn),
            Action::Map => self.map_ready(),
            Action::Transition if self.key != 0 => Err(Refusal::KeyShifted),
            Action::Transition if self.map_under_way() => Err(Refusal::MapUnderWay),
            Action::Transition => Ok(()),
        }
    }

    /// Whether the track can be read along a map at all now (its stems,
    /// as its parts).
    fn map_ready(&self) -> Result<(), Refusal> {
        match self.lanes {
            None => Err(Refusal::NoOutput),
            Some(_) if self.channels > 2 => Err(Refusal::Channels(self.channels)),
            Some(_) if !self.own_route => Err(Refusal::OffRoute),
            Some(lanes) if lanes <= 1 => Err(Refusal::NoLanes),
            Some(_) if self.route_meeting => Err(Refusal::RouteMeeting),
            Some(_) => Ok(()),
        }
    }

    /// Whether a track is read along a map and not leaving it, or waits to
    /// be.
    pub(super) fn map_under_way(&self) -> bool {
        self.map_waiting || self.on_map == Some(false)
    }
}

impl Engine {
    /// What the head on air is doing.
    pub(super) fn situation(&self) -> Situation {
        let track = self.head.playing.as_ref();
        Situation {
            transition: self.transition.is_some(),
            playing: track.is_some(),
            key: track.map_or(0, Track::key),
            on_map: track.and_then(|t| t.along.as_ref()).map(|a| a.leave),
            map_waiting: matches!(self.head.moving, Move::MapAt(_)),
            count_in: self.head.lead > 0,
            lanes: self.head.out.as_ref().map(|o| o.output.lanes()),
            channels: track.map_or(0, |t| t.route.channels),
            own_route: track.is_some_and(Track::as_heard),
            route_meeting: self.head.switch.as_ref().is_some_and(|s| s.spliced),
        }
    }

    /// Whether `action` may be made on the head on air now, or why not.
    pub(super) fn allows(&self, action: Action) -> Result<(), Refusal> {
        self.situation().allows(action)
    }

    /// Whether the head on air plays in a key not its own.
    pub(super) fn head_shifted(&self) -> bool {
        self.situation().key != 0
    }

    /// Whether the playing track's route is held: on a map, or waiting to
    /// be, any switch but the way back it asked for waits (its end routes
    /// it again); while a transition is prepared or mixed, its playing
    /// track is where the next one is linked to, and only a way back off a
    /// map moves it.
    pub(super) fn route_held(&self) -> bool {
        let on_map = self.head.playing.as_ref().and_then(|t| t.along.as_ref());
        let map_holds = matches!(self.head.moving, Move::MapAt(_))
            || on_map.is_some_and(|a| a.back != Back::Asked);
        map_holds || (self.transition_holds_route() && on_map.is_none())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A track on its own route, on an output of lanes, nothing else under
    /// way.
    fn plain() -> Situation {
        Situation {
            playing: true,
            lanes: Some(5),
            channels: 2,
            own_route: true,
            ..Situation::default()
        }
    }

    /// Each rule, as `player/INTERACTIONS.md` tells it: what is under way,
    /// and what each move gets.
    #[test]
    fn the_rules_between_moves() {
        use Action::*;
        use Refusal::*;
        /// What is under way, the situation, and what each move gets.
        type Row = (&'static str, Situation, [Result<(), Refusal>; 5]);
        let ok = Ok(());
        let rows: [Row; 10] = [
            ("nothing under way", plain(), [ok, ok, ok, ok, ok]),
            (
                "a transition set up",
                Situation {
                    transition: true,
                    ..plain()
                },
                [Err(TransitionSetUp), Err(TransitionSetUp), ok, ok, ok],
            ),
            (
                "the key shifted",
                Situation {
                    key: 2,
                    on_map: Some(false),
                    ..plain()
                },
                [Err(KeyShifted), ok, ok, ok, Err(KeyShifted)],
            ),
            (
                "on a map",
                Situation {
                    on_map: Some(false),
                    ..plain()
                },
                [Err(TrackOnMap), ok, ok, ok, Err(MapUnderWay)],
            ),
            (
                "leaving a map",
                Situation {
                    on_map: Some(true),
                    ..plain()
                },
                [Err(TrackOnMap), ok, ok, ok, ok],
            ),
            (
                "waiting for a map",
                Situation {
                    map_waiting: true,
                    ..plain()
                },
                [Err(TrackOnMap), ok, ok, ok, Err(MapUnderWay)],
            ),
            (
                "a count-in",
                Situation {
                    count_in: true,
                    ..plain()
                },
                [ok, ok, ok, Err(CountIn), ok],
            ),
            (
                "five channels",
                Situation {
                    channels: 5,
                    ..plain()
                },
                [ok, Err(Channels(5)), Err(Channels(5)), Err(Channels(5)), ok],
            ),
            (
                "no lanes",
                Situation {
                    lanes: Some(1),
                    ..plain()
                },
                [ok, Err(NoLanes), Err(NoLanes), Err(NoLanes), ok],
            ),
            (
                "a route switch meeting it",
                Situation {
                    route_meeting: true,
                    ..plain()
                },
                [
                    ok,
                    Err(RouteMeeting),
                    Err(RouteMeeting),
                    Err(RouteMeeting),
                    ok,
                ],
            ),
        ];
        for (what, situation, want) in rows {
            let got = [Jump, Key, KeyOnTrack, Map, Transition].map(|a| situation.allows(a));
            assert_eq!(got, want, "{what}");
        }
        // With nothing playing a key is for the track picked next.
        let idle = Situation::default();
        assert_eq!(idle.allows(Key), Ok(()));
        assert_eq!(idle.allows(KeyOnTrack), Err(NoOutput));
    }

    #[test]
    fn every_refusal_has_its_reason() {
        assert_eq!(
            Refusal::Channels(6).reason(),
            "it has 6 channels, and stems are made of two at most"
        );
        assert_eq!(
            Refusal::TrackOnMap.reason(),
            "the track is read along a time map"
        );
    }
}
