//! The LRCLIB web service (`lrclib.net`): its addresses and its answers.
//! The requests themselves are the caller's, so this crate needs no web
//! client.
//!
//! - `GET /api/get?artist_name=&track_name=&album_name=&duration=` finds
//!   one record, matching the duration within about 2 s; 404 when none.
//! - `GET /api/search?track_name=&artist_name=` lists records; the best
//!   is the one whose title and artist match, nearest the duration, timed
//!   before untimed.
//! - A record marked `instrumental` has no words.

use crate::json::Json;
use crate::tokens::normalize;
use crate::{Found, Lyrics, Source, Track};

/// The service's address.
pub const BASE: &str = "https://lrclib.net";

/// How far a search result's length may be from the track's, in seconds.
pub const DURATION_SLACK: f64 = 5.0;

/// `text` for a query string: letters, digits and `-._~` as they are,
/// every other byte of its UTF-8 as `%XX`.
pub fn encode(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for b in text.bytes() {
        if b.is_ascii_alphanumeric() || matches!(b, b'-' | b'.' | b'_' | b'~') {
            out.push(char::from(b));
        } else {
            out.push_str(&format!("%{b:02X}"));
        }
    }
    out
}

/// The track's title and artist, when it has both (the service needs
/// them).
fn names(track: &Track) -> Option<(&str, &str)> {
    let title = track
        .title
        .as_deref()
        .map(str::trim)
        .filter(|t| !t.is_empty())?;
    let artist = track
        .artist
        .as_deref()
        .map(str::trim)
        .filter(|a| !a.is_empty())?;
    Some((title, artist))
}

/// The address that gets the one record for `track` from `base`.
pub fn get_url(base: &str, track: &Track) -> Option<String> {
    let (title, artist) = names(track)?;
    let mut url = format!(
        "{}/api/get?artist_name={}&track_name={}",
        base.trim_end_matches('/'),
        encode(artist),
        encode(title)
    );
    if let Some(album) = track
        .album
        .as_deref()
        .map(str::trim)
        .filter(|a| !a.is_empty())
    {
        url.push_str("&album_name=");
        url.push_str(&encode(album));
    }
    if let Some(d) = track.duration.filter(|d| *d > 0.0) {
        url.push_str(&format!("&duration={}", d.round() as u64));
    }
    Some(url)
}

/// The address that searches `base` for `track` by title and artist.
pub fn search_url(base: &str, track: &Track) -> Option<String> {
    let (title, artist) = names(track)?;
    Some(format!(
        "{}/api/search?track_name={}&artist_name={}",
        base.trim_end_matches('/'),
        encode(title),
        encode(artist)
    ))
}

/// One of the service's records.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Record {
    /// Its id there.
    pub id: Option<u64>,
    /// The title.
    pub title: Option<String>,
    /// The artist.
    pub artist: Option<String>,
    /// The album.
    pub album: Option<String>,
    /// The length in seconds.
    pub duration: Option<f64>,
    /// Whether the track has no words.
    pub instrumental: bool,
    /// The text without times.
    pub plain: Option<String>,
    /// The text as LRC.
    pub synced: Option<String>,
}

impl Record {
    /// The record in a JSON object.
    pub fn from_json(v: &Json) -> Option<Record> {
        let text = |key: &str| {
            v.get(key)
                .and_then(Json::as_str)
                .map(str::to_owned)
                .filter(|s| !s.trim().is_empty())
        };
        if !matches!(v, Json::Object(_)) {
            return None;
        }
        Some(Record {
            id: v.get("id").and_then(Json::as_f64).map(|n| n as u64),
            title: text("trackName"),
            artist: text("artistName"),
            album: text("albumName"),
            duration: v.get("duration").and_then(Json::as_f64),
            instrumental: v.get("instrumental").and_then(Json::as_bool) == Some(true),
            plain: text("plainLyrics"),
            synced: text("syncedLyrics"),
        })
    }

    /// The text: the plain one, the synced one (its times to be dropped)
    /// only where there is no other.
    pub fn text(&self) -> Option<&str> {
        self.plain.as_deref().or(self.synced.as_deref())
    }

    /// What the record says of the track: its words, without any of the
    /// times it may carry ([`Lyrics::parse_text`]).
    pub fn found(&self) -> Found {
        if self.instrumental {
            return Found::Instrumental;
        }
        match self.text().map(|t| Lyrics::parse_text(t, Source::Lrclib)) {
            Some(lyrics) if !lyrics.is_empty() => Found::Lyrics(lyrics),
            _ => Found::NotFound,
        }
    }
}

/// The record in the answer of [`get_url`].
pub fn parse_get(body: &str) -> Option<Record> {
    Record::from_json(&Json::parse(body)?)
}

/// The records in the answer of [`search_url`].
pub fn parse_search(body: &str) -> Vec<Record> {
    Json::parse(body)
        .as_ref()
        .and_then(Json::as_array)
        .map(|items| items.iter().filter_map(Record::from_json).collect())
        .unwrap_or_default()
}

/// A name compared loosely: its letters and digits only, in upper case,
/// without diacritics.
fn loose(name: &str) -> String {
    name.split_whitespace()
        .map(|w| normalize(w).unwrap_or_else(|| w.to_uppercase()))
        .collect::<String>()
        .chars()
        .filter(|c| c.is_alphanumeric())
        .collect()
}

/// The best of `records` for `track`: its title and artist the same
/// (loosely, or one holding the other), within [`DURATION_SLACK`] of its
/// length when both are known, nearest that length. Records with no text
/// count only when instrumental.
pub fn best<'a>(records: &'a [Record], track: &Track) -> Option<&'a Record> {
    let (title, artist) = names(track)?;
    let (title, artist) = (loose(title), loose(artist));
    let alike =
        |a: &str, b: &str| !a.is_empty() && !b.is_empty() && (a.contains(b) || b.contains(a));
    records
        .iter()
        .filter(|r| r.instrumental || r.text().is_some())
        .filter(|r| r.title.as_deref().is_some_and(|t| alike(&loose(t), &title)))
        .filter(|r| {
            r.artist
                .as_deref()
                .is_some_and(|a| alike(&loose(a), &artist))
        })
        .map(|r| {
            let off = match (r.duration, track.duration) {
                (Some(a), Some(b)) => (a - b).abs(),
                _ => DURATION_SLACK,
            };
            (r, off)
        })
        .filter(|&(_, off)| off <= DURATION_SLACK)
        .min_by(|a, b| a.1.total_cmp(&b.1))
        .map(|(r, _)| r)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Detail;

    fn track() -> Track {
        Track {
            title: Some("Café del Mar".into()),
            artist: Some("Energy 52".into()),
            album: Some("Café del Mar & More".into()),
            duration: Some(241.6),
            ..Track::default()
        }
    }

    #[test]
    fn addresses_carry_the_track() {
        assert_eq!(
            get_url(BASE, &track()).unwrap(),
            "https://lrclib.net/api/get?artist_name=Energy%2052&track_name=Caf%C3%A9%20del%20Mar\
             &album_name=Caf%C3%A9%20del%20Mar%20%26%20More&duration=242"
        );
        assert_eq!(
            search_url("http://127.0.0.1:9/", &track()).unwrap(),
            "http://127.0.0.1:9/api/search?track_name=Caf%C3%A9%20del%20Mar&artist_name=Energy%2052"
        );
        let nameless = Track {
            artist: None,
            ..track()
        };
        assert_eq!(get_url(BASE, &nameless), None);
    }

    #[test]
    fn records_are_read() {
        let body = r#"{"id":3396226,"trackName":"I Want to Live","artistName":"Borislav Slavov",
            "albumName":"Baldur's Gate 3","duration":233,"instrumental":false,
            "plainLyrics":"I feel your breath\nUpon my neck",
            "syncedLyrics":"[00:17.12] I feel your breath\n[00:47.10] Upon my neck"}"#;
        let r = parse_get(body).unwrap();
        assert_eq!(r.id, Some(3_396_226));
        assert_eq!(r.duration, Some(233.0));
        let Found::Lyrics(l) = r.found() else {
            panic!("no lyrics");
        };
        assert_eq!(l.source, Source::Lrclib);
        // The words only: LRCLIB's times are never used.
        assert_eq!(l.timing.detail, Detail::None);
        assert_eq!(l.lines[1].time, None);
        assert_eq!(l.lines[1].text(), "Upon my neck");
        // A record with only a timed text: its words, its times dropped.
        let synced = r#"{"id":2,"trackName":"x","artistName":"y","instrumental":false,
            "plainLyrics":null,"syncedLyrics":"[00:17.12] <00:17.12>I <00:17.50>feel\n[00:20.00]\n[00:21.00] on"}"#;
        let Found::Lyrics(l) = parse_get(synced).unwrap().found() else {
            panic!("no lyrics");
        };
        assert_eq!(l.lines.len(), 3);
        assert!(l.lines[1].is_break, "the empty timed line: a break");
        assert!(l.lines.iter().all(|line| line.time.is_none()
            && line
                .words
                .iter()
                .all(|w| w.start.is_none() && w.end.is_none())));
        assert_eq!(l.lines[0].text(), "I feel");
        let instrumental = r#"{"id":1,"trackName":"x","artistName":"y","instrumental":true,
            "plainLyrics":null,"syncedLyrics":null}"#;
        assert_eq!(
            parse_get(instrumental).unwrap().found(),
            Found::Instrumental
        );
        assert_eq!(
            parse_get(r#"{"code":404,"name":"TrackNotFound"}"#)
                .unwrap()
                .found(),
            Found::NotFound
        );
        assert_eq!(parse_search("not json"), Vec::new());
    }

    #[test]
    fn the_best_search_result_matches_names_and_length() {
        let record = |id, title: &str, artist: &str, duration, synced: bool| Record {
            id: Some(id),
            title: Some(title.into()),
            artist: Some(artist.into()),
            duration: Some(duration),
            plain: Some("la".into()),
            synced: synced.then(|| "[00:01.00]la".into()),
            ..Record::default()
        };
        let records = vec![
            record(1, "Something Else", "Energy 52", 241.0, true),
            record(2, "Cafe Del Mar", "Energy 52", 242.0, false),
            record(3, "Café del Mar (Radio Edit)", "Energy 52", 230.0, true),
            record(4, "CAFE DEL MAR", "ENERGY 52", 244.0, true),
            record(5, "Café del Mar", "Energy 52", 300.0, true),
        ];
        // 3 is too short, 5 too long, 1 another song; 2 is nearest in
        // length (being timed counts for nothing).
        assert_eq!(best(&records, &track()).and_then(|r| r.id), Some(2));
        assert_eq!(best(&records[3..4], &track()).and_then(|r| r.id), Some(4));
        assert_eq!(best(&records[..2], &track()).and_then(|r| r.id), Some(2));
        assert_eq!(best(&records[..1], &track()), None);
    }
}
