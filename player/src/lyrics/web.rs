//! The web: GET requests, and LRCLIB asked through them, one request at a
//! time and at most one a second.

use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use jodel_lyrics::lrclib::{self, Record};
use jodel_lyrics::{Found, Lyrics, LyricsSource, Track};

/// What the player tells web services it is.
pub const USER_AGENT: &str = concat!("jodeldiplom/", env!("CARGO_PKG_VERSION"));

/// The longest `Retry-After` waited out; a longer one gives up for now.
pub const LONGEST_WAIT: Duration = Duration::from_secs(120);

/// How long to wait when a busy answer gives no `Retry-After`.
pub(super) const BUSY_WAIT: Duration = Duration::from_secs(10);

/// How many times one address is asked while the service is busy.
const TRIES: usize = 3;

/// A web service's answer.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Response {
    /// The HTTP status.
    pub status: u16,
    /// The body, as text.
    pub body: String,
    /// The `Retry-After` header, when it gives seconds.
    pub retry_after: Option<Duration>,
}

/// Something that makes GET requests.
pub trait Http: Send + Sync {
    /// The answer to `GET url`; `Err` when there was none (no network, a
    /// timeout).
    fn get(&self, url: &str) -> Result<Response, String>;
}

/// The `Retry-After` value in seconds (its date form is not read).
pub fn retry_after(value: &str) -> Option<Duration> {
    value.trim().parse().ok().map(Duration::from_secs)
}

/// GET requests over the web, through `ureq` (HTTPS by rustls).
pub struct Web {
    agent: ureq::Agent,
}

impl Web {
    /// A client with the player's `User-Agent` and a 20-second limit a
    /// request, through the proxy the environment names, if any.
    pub fn new() -> Web {
        Web::with(true)
    }

    pub(super) fn with(env_proxy: bool) -> Web {
        let mut config = ureq::Agent::config_builder()
            .user_agent(USER_AGENT)
            .http_status_as_error(false)
            .timeout_global(Some(Duration::from_secs(20)));
        if !env_proxy {
            config = config.proxy(None);
        }
        Web {
            agent: config.build().into(),
        }
    }
}

impl Default for Web {
    fn default() -> Self {
        Web::new()
    }
}

impl Http for Web {
    fn get(&self, url: &str) -> Result<Response, String> {
        let mut answer = self.agent.get(url).call().map_err(|e| e.to_string())?;
        let retry = answer
            .headers()
            .get("retry-after")
            .and_then(|v| v.to_str().ok())
            .and_then(retry_after);
        let status = answer.status().as_u16();
        let body = answer
            .body_mut()
            .read_to_string()
            .map_err(|e| e.to_string())?;
        Ok(Response {
            status,
            body,
            retry_after: retry,
        })
    }
}

/// The least time from one request to LRCLIB to the next.
pub const SPACING: Duration = Duration::from_secs(1);

/// The LRCLIB service as a source. Its clones share when the next request
/// may go, so all of them together keep to [`SPACING`].
#[derive(Clone)]
pub struct Lrclib {
    http: Arc<dyn Http>,
    base: String,
    /// When the last request was let go (or is to go).
    last: Arc<Mutex<Option<Instant>>>,
}

impl Lrclib {
    /// LRCLIB at `base` ([`lrclib::BASE`]), asked through `http`.
    pub fn new(http: Arc<dyn Http>, base: impl Into<String>) -> Lrclib {
        Lrclib {
            http,
            base: base.into(),
            last: Arc::new(Mutex::new(None)),
        }
    }

    /// The body of `GET url`: `None` for a 404. Each try asked
    /// [`SPACING`] after the one before it (by any clone) at the soonest; a
    /// busy answer is waited out and asked again. `wait` waits, and says
    /// `false` to give up; nothing is locked while it waits or the request
    /// goes.
    fn ask(&self, url: &str, wait: &dyn Fn(Duration) -> bool) -> Result<Option<String>, String> {
        for _ in 0..TRIES {
            let until = self.slot();
            if !until.is_zero() && !wait(until) {
                return Err("stopped".into());
            }
            let answer = self.http.get(url)?;
            match answer.status {
                200 => return Ok(Some(answer.body)),
                404 => return Ok(None),
                429 | 503 => {
                    let pause = answer.retry_after.unwrap_or(BUSY_WAIT);
                    if pause > LONGEST_WAIT {
                        return Err(format!("LRCLIB asks to wait {} s", pause.as_secs()));
                    }
                    if !wait(pause) {
                        return Err("stopped".into());
                    }
                }
                status => return Err(format!("LRCLIB answered {status}")),
            }
        }
        Err("LRCLIB stayed busy".into())
    }

    /// Takes the next slot for a request, [`SPACING`] after the last one
    /// taken (or now): how long until it.
    fn slot(&self) -> Duration {
        let mut last = self.last.lock().unwrap_or_else(|e| e.into_inner());
        let now = Instant::now();
        let at = last.map_or(now, |l| (l + SPACING).max(now));
        *last = Some(at);
        at - now
    }

    /// The record LRCLIB has for `track`: the one its `get` finds (when
    /// it has text or is instrumental), else the best of a search. `None`
    /// when there is none, or the track has no title and artist to ask
    /// by.
    pub fn fetch(
        &self,
        track: &Track,
        wait: &dyn Fn(Duration) -> bool,
    ) -> Result<Option<Record>, String> {
        let Some(get) = lrclib::get_url(&self.base, track) else {
            return Ok(None);
        };
        if let Some(record) = self.ask(&get, wait)?.as_deref().and_then(lrclib::parse_get) {
            if record.found() != Found::NotFound {
                return Ok(Some(record));
            }
        }
        let Some(search) = lrclib::search_url(&self.base, track) else {
            return Ok(None);
        };
        let records = self
            .ask(&search, wait)?
            .map(|body| lrclib::parse_search(&body))
            .unwrap_or_default();
        Ok(lrclib::best(&records, track).cloned())
    }

    /// The texts LRCLIB's search has for `track` (each once, at most
    /// [`MOST_CANDIDATES`]), for when the one found does not fit the song.
    pub fn candidates(
        &self,
        track: &Track,
        wait: &dyn Fn(Duration) -> bool,
    ) -> Result<Vec<Lyrics>, String> {
        let Some(search) = lrclib::search_url(&self.base, track) else {
            return Ok(Vec::new());
        };
        let records = self
            .ask(&search, wait)?
            .map(|body| lrclib::parse_search(&body))
            .unwrap_or_default();
        let mut out: Vec<Lyrics> = Vec::new();
        for record in &records {
            if let Found::Lyrics(lyrics) = record.found() {
                let words = jodel_lyrics::fit::lyrics_words(&lyrics);
                if !out
                    .iter()
                    .any(|o| jodel_lyrics::fit::lyrics_words(o) == words)
                {
                    out.push(lyrics);
                }
            }
            if out.len() == MOST_CANDIDATES {
                break;
            }
        }
        Ok(out)
    }
}

/// LRCLIB as the player asks it: through [`Web`], at [`lrclib::BASE`].
pub fn lrclib_of() -> Lrclib {
    Lrclib::new(Arc::new(Web::new()), lrclib::BASE)
}

/// The most of LRCLIB's other texts a song's is compared with.
pub const MOST_CANDIDATES: usize = 12;

impl LyricsSource for Lrclib {
    fn name(&self) -> &str {
        "lrclib"
    }

    fn find(&self, track: &Track) -> Result<Found, String> {
        let wait = |pause| {
            thread::sleep(pause);
            true
        };
        Ok(self
            .fetch(track, &wait)?
            .map_or(Found::NotFound, |r| r.found()))
    }
}
