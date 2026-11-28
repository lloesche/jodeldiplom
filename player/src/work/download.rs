//! Files fetched from the web: filled beside where they go (a
//! [`durable::Resumable`]), checked against their length and SHA-256, then
//! put in place. A download cut off goes on where it stopped the next time
//! ([`download`]).

use std::io::{Read, Write};
use std::path::Path;

use durable::Resumable;

/// Why a download stopped.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Fault {
    /// The web let it down (no network, the server, the download cut
    /// off): trying again later may mend it.
    Web(String),
    /// What the server sends is not the file (its length or its SHA-256
    /// another): trying again would get the same.
    Wrong(String),
    /// The disk let it down: the data folder could not be written to.
    Disk(String),
}

impl Fault {
    /// Why, in words.
    pub fn why(&self) -> &str {
        match self {
            Fault::Web(why) | Fault::Wrong(why) | Fault::Disk(why) => why,
        }
    }
}

fn disk(e: impl std::fmt::Display) -> Fault {
    Fault::Disk(e.to_string())
}

fn web(e: impl std::fmt::Display) -> Fault {
    Fault::Web(format!("the download failed: {e}"))
}

/// Downloads `url` to `to`, telling `progress` the bytes so far and in
/// all: filled in `to`'s resumable file beside it, checked against its
/// length `bytes` and its SHA-256 `sha256`, then given `to`'s name (over a
/// damaged file there). What an earlier download left there (cut off, or
/// the player closed) is gone on from ([`fill`]); one that is not the file
/// once whole is removed. Going on is safe because the whole is checked
/// against the built-in SHA-256 at the end. Where it fails that check
/// after going on from bytes kept (torn by a crash, or of another copy of
/// the file: the disk's fault, not the server's), the file is filled once
/// more from its start; only a fresh download that is not the file is
/// [`Fault::Wrong`].
pub fn download(
    url: &str,
    to: &Path,
    bytes: u64,
    sha256: &str,
    mut progress: impl FnMut(u64, u64),
) -> Result<(), Fault> {
    let mut part = Resumable::open(to).map_err(disk)?;
    let (mut got, resumed) = checked(url, &mut part, (bytes, sha256), &mut progress);
    if resumed && matches!(got, Err(Fault::Wrong(_))) {
        part.keep_only(0).map_err(disk)?;
        got = checked(url, &mut part, (bytes, sha256), &mut progress).0;
    }
    match got {
        Ok(()) => part.commit().map_err(disk),
        Err(Fault::Wrong(why)) => {
            let _ = part.discard();
            Err(Fault::Wrong(why))
        }
        Err(fault) => Err(fault),
    }
}

/// `part` filled with `url`'s file ([`fill`]) and checked against its
/// length and SHA-256: whether it is the file, and whether bytes kept from
/// before went into it.
fn checked(
    url: &str,
    part: &mut Resumable,
    (bytes, sha256): (u64, &str),
    progress: &mut impl FnMut(u64, u64),
) -> (Result<(), Fault>, bool) {
    let mut resumed = false;
    let filled = fill(url, part, bytes, progress, &mut resumed);
    let checked = filled.and_then(|(hash, done)| {
        let digest = hash.hex();
        (done == bytes && digest == sha256)
            .then_some(())
            .ok_or_else(|| {
                Fault::Wrong(format!(
                    "the download of {url} is not the file ({done} bytes, SHA-256 {digest})"
                ))
            })
    });
    (checked, resumed)
}

/// `part` filled with `url`'s `bytes`: the hash of what it holds then, and
/// its length; `resumed` set while bytes it kept from before are in it.
/// What it kept is hashed again first (more than `bytes` is thrown away);
/// kept whole, the network is not asked. Otherwise only the rest is asked
/// for (a range); a server that sends the whole again has the file filled
/// from its start, one that refuses the range, or sends another part
/// (another start, another or no total), is asked for the whole. More than
/// `bytes` sent is not the file ([`Fault::Wrong`]); cut off, what came is
/// kept on the disk, to be gone on from.
fn fill(
    url: &str,
    part: &mut Resumable,
    bytes: u64,
    progress: &mut impl FnMut(u64, u64),
    resumed: &mut bool,
) -> Result<(Sha256, u64), Fault> {
    let (mut hash, mut done) = kept(part, bytes)?;
    *resumed = done > 0;
    progress(done, bytes);
    if done == bytes {
        return Ok((hash, done));
    }
    let mut response = match ask(url, done) {
        Err(ureq::Error::StatusCode(416)) if done > 0 => {
            (hash, done, *resumed) = (Sha256::new(), 0, false);
            part.keep_only(0).map_err(disk)?;
            ask(url, 0).map_err(web)?
        }
        asked => asked.map_err(web)?,
    };
    if done > 0 && !goes_on(&response, done, bytes) {
        (hash, done, *resumed) = (Sha256::new(), 0, false);
        part.keep_only(0).map_err(disk)?;
        progress(0, bytes);
        // Another part than the rest is not the file's start: the whole
        // asked for. A whole sent again is it.
        if response.status().as_u16() != 200 {
            response = ask(url, 0).map_err(web)?;
        }
    }
    if done == 0 && response.status().as_u16() != 200 {
        return Err(web(format_args!(
            "{} answered for the whole file",
            response.status()
        )));
    }
    let mut body = (response.body_mut().with_config())
        .limit(bytes - done)
        .reader();
    let mut buf = vec![0u8; 1 << 16];
    loop {
        let n = match body.read(&mut buf) {
            Ok(n) => n,
            Err(e) => {
                return Err(match ureq::Error::from(e) {
                    ureq::Error::BodyExceedsLimit(_) => Fault::Wrong(format!(
                        "the download of {url} is longer than the file's {bytes} bytes"
                    )),
                    e => {
                        let _ = part.sync();
                        web(e)
                    }
                })
            }
        };
        if n == 0 {
            break;
        }
        part.write_all(&buf[..n]).map_err(disk)?;
        hash.update(&buf[..n]);
        done += n as u64;
        progress(done, bytes);
    }
    if done < bytes {
        let _ = part.sync();
        return Err(Fault::Web(format!(
            "the download of {url} was cut off at {done} of {bytes} bytes"
        )));
    }
    Ok((hash, done))
}

/// The hash and the length of what `part` kept, its file's `bytes` at most
/// (one longer is emptied: it is not the file).
fn kept(part: &mut Resumable, bytes: u64) -> Result<(Sha256, u64), Fault> {
    let mut hash = Sha256::new();
    if part.kept().map_err(disk)? > bytes {
        part.keep_only(0).map_err(disk)?;
        return Ok((hash, 0));
    }
    let mut buf = vec![0u8; 1 << 20];
    let mut done = 0u64;
    loop {
        let n = part.read(&mut buf).map_err(disk)?;
        if n == 0 {
            break;
        }
        hash.update(&buf[..n]);
        done += n as u64;
    }
    part.keep_only(done).map_err(disk)?;
    Ok((hash, done))
}

/// `url` asked for from its byte `from` on (0: all of it).
fn ask(url: &str, from: u64) -> Result<ureq::http::Response<ureq::Body>, ureq::Error> {
    let agent: ureq::Agent = ureq::Agent::config_builder()
        .user_agent(crate::work::lyrics::web::USER_AGENT)
        .build()
        .into();
    let mut request = agent.get(url);
    if from > 0 {
        request = request.header("Range", format!("bytes={from}-"));
    }
    request.call()
}

/// Whether `response` is the rest of a file of `bytes` from its byte
/// `from` on: a part (206) whose range starts there and whose total is
/// the file's.
fn goes_on(response: &ureq::http::Response<ureq::Body>, from: u64, bytes: u64) -> bool {
    let range = (response.headers().get("content-range")).and_then(|v| v.to_str().ok());
    let part = range.and_then(|r| r.strip_prefix("bytes "));
    let (span, total) = part.and_then(|p| p.split_once('/')).unwrap_or_default();
    response.status().as_u16() == 206
        && span
            .split_once('-')
            .is_some_and(|(start, _)| start == from.to_string())
        && total == bytes.to_string()
}

/// SHA-256 (FIPS 180-4), to check a download.
pub struct Sha256 {
    state: [u32; 8],
    block: Vec<u8>,
    length: u64,
}

impl Default for Sha256 {
    fn default() -> Self {
        Sha256::new()
    }
}

impl Sha256 {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];

    /// A hash of nothing yet.
    pub fn new() -> Sha256 {
        Sha256 {
            state: [
                0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
                0x5be0cd19,
            ],
            block: Vec::with_capacity(64),
            length: 0,
        }
    }

    /// Takes in `data`.
    pub fn update(&mut self, mut data: &[u8]) {
        self.length += data.len() as u64;
        while !data.is_empty() {
            let take = (64 - self.block.len()).min(data.len());
            self.block.extend_from_slice(&data[..take]);
            data = &data[take..];
            if self.block.len() == 64 {
                let block: [u8; 64] = self.block[..].try_into().unwrap();
                self.compress(&block);
                self.block.clear();
            }
        }
    }

    fn compress(&mut self, block: &[u8; 64]) {
        let mut w = [0u32; 64];
        for (i, word) in block.as_chunks::<4>().0.iter().enumerate() {
            w[i] = u32::from_be_bytes(*word);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }
        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = self.state;
        for (&k, &wi) in Self::K.iter().zip(&w) {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ (!e & g);
            let t1 = h
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(k)
                .wrapping_add(wi);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let t2 = s0.wrapping_add(maj);
            (h, g, f, e, d, c, b, a) = (g, f, e, d.wrapping_add(t1), c, b, a, t1.wrapping_add(t2));
        }
        for (s, v) in self.state.iter_mut().zip([a, b, c, d, e, f, g, h]) {
            *s = s.wrapping_add(v);
        }
    }

    /// The digest, in lowercase hex.
    pub fn hex(mut self) -> String {
        let bits = self.length.wrapping_mul(8);
        self.update(&[0x80]);
        while self.block.len() != 56 {
            self.update(&[0]);
        }
        self.update(&bits.to_be_bytes());
        self.state.iter().map(|w| format!("{w:08x}")).collect()
    }
}

#[cfg(test)]
#[expect(
    clippy::disallowed_methods,
    reason = "tests: partial files as an earlier download left them"
)]
mod tests {
    use super::*;
    use std::fs;
    use std::io::BufRead;
    use std::net::TcpListener;
    use std::path::PathBuf;
    use std::sync::mpsc::Receiver;

    /// How the made-up server answers a request: from the file's bytes and
    /// where the range asked for starts (none asked: `None`), the whole
    /// answer.
    type Answer = Box<dyn Fn(&[u8], Option<usize>) -> Vec<u8> + Send>;

    /// The whole file, a status of 200.
    fn whole() -> Answer {
        Box::new(|data, _| {
            let head = format!("HTTP/1.1 200 OK\r\nContent-Length: {}\r\n\r\n", data.len());
            let mut out = head.into_bytes();
            out.extend(data);
            out
        })
    }

    /// The whole file said, its first `cut` bytes sent.
    fn cut(cut: usize) -> Answer {
        Box::new(move |data, _| {
            let mut out = whole()(data, None);
            out.truncate(out.len() - data.len() + cut);
            out
        })
    }

    /// The range asked for, a status of 206.
    fn ranged() -> Answer {
        Box::new(|data, from| {
            let from = from.expect("a range");
            let head = format!(
                "HTTP/1.1 206 Partial Content\r\nContent-Range: bytes {from}-{}/{}\r\nContent-Length: {}\r\n\r\n",
                data.len() - 1,
                data.len(),
                data.len() - from
            );
            let mut out = head.into_bytes();
            out.extend(&data[from..]);
            out
        })
    }

    /// A part that is not the rest asked for (a status of 206): from `skip`
    /// bytes past where it was asked, its total said as `total` (`*`: not
    /// known).
    fn another_part(skip: usize, total: &'static str) -> Answer {
        Box::new(move |data, from| {
            let from = from.expect("a range") + skip;
            let head = format!(
                "HTTP/1.1 206 Partial Content\r\nContent-Range: bytes {from}-{}/{total}\r\nContent-Length: {}\r\n\r\n",
                data.len() - 1,
                data.len() - from
            );
            let mut out = head.into_bytes();
            out.extend(&data[from..]);
            out
        })
    }

    /// Another file of the same length, whole.
    fn another_file() -> Answer {
        Box::new(|data, _| {
            let other: Vec<u8> = data.iter().rev().copied().collect();
            whole()(&other, None)
        })
    }

    /// The range refused (416).
    fn refused() -> Answer {
        Box::new(|_, _| b"HTTP/1.1 416 Range Not Satisfiable\r\nContent-Length: 0\r\n\r\n".to_vec())
    }

    /// A file longer than the one expected.
    fn longer() -> Answer {
        Box::new(|data, _| {
            let more: Vec<u8> = data.iter().chain(b"and more").copied().collect();
            whole()(&more, None)
        })
    }

    /// A server on this computer answering each request with the next of
    /// `answers` for `data`: its address, and the range each request asked
    /// for (`""`: none).
    fn server(data: &[u8], answers: Vec<Answer>) -> (String, Receiver<String>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = format!("http://{}/model.bin", listener.local_addr().unwrap());
        let (tx, rx) = std::sync::mpsc::channel();
        let data = data.to_vec();
        std::thread::spawn(move || {
            for (stream, answer) in listener.incoming().zip(answers) {
                let mut stream = stream.unwrap();
                let mut reader = std::io::BufReader::new(stream.try_clone().unwrap());
                let mut range = String::new();
                loop {
                    let mut line = String::new();
                    reader.read_line(&mut line).unwrap();
                    if line.trim().is_empty() {
                        break;
                    }
                    if let Some(r) = line.to_ascii_lowercase().strip_prefix("range: bytes=") {
                        range = r.trim().to_owned();
                    }
                }
                let _ = tx.send(range.clone());
                let from = range.trim_end_matches('-').parse().ok();
                let _ = stream.write_all(&answer(&data, from));
                let _ = stream.flush();
            }
        });
        (address, rx)
    }

    /// A file's bytes and SHA-256, a folder for it, and where it goes.
    fn file(name: &str) -> (Vec<u8>, String, PathBuf, PathBuf) {
        let dir =
            std::env::temp_dir().join(format!("jodel-download-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let data: Vec<u8> = (0..300_000u32).map(|i| (i * 7 % 251) as u8).collect();
        let mut hash = Sha256::new();
        hash.update(&data);
        let to = dir.join("models").join("model.bin");
        (data, hash.hex(), dir, to)
    }

    /// `to`'s resumable file holding `bytes`.
    fn left(to: &Path, bytes: &[u8]) {
        fs::create_dir_all(to.parent().unwrap()).unwrap();
        fs::write(Resumable::part_of(to), bytes).unwrap();
    }

    /// A download cut off keeps what came, and the next one goes on from
    /// there: only the rest asked for, the whole checked.
    #[test]
    fn a_download_cut_off_goes_on_where_it_stopped() {
        let (data, sha, dir, to) = file("cut");
        let (url, ranges) = server(&data, vec![cut(120_000), ranged()]);
        let len = data.len() as u64;
        let first = download(&url, &to, len, &sha, |_, _| {});
        assert!(matches!(first, Err(Fault::Web(_))), "{first:?}");
        assert!(!to.exists());
        assert_eq!(ranges.recv().unwrap(), "", "the first asks for all");
        let mut seen = Vec::new();
        download(&url, &to, len, &sha, |done, _| seen.push(done)).unwrap();
        assert_eq!(ranges.recv().unwrap(), "120000-", "the second for the rest");
        assert_eq!(seen.first(), Some(&120_000), "{seen:?}");
        assert_eq!(fs::read(&to).unwrap(), data);
        assert!(!Resumable::part_of(&to).exists());
        let _ = fs::remove_dir_all(&dir);
    }

    /// A server that sends the whole again for a range has the file filled
    /// from its start; one that refuses the range, or sends another part
    /// (from another byte, or of a total not known), is asked for the whole
    /// (that part never written as the file's start); one longer than the
    /// file kept is thrown away and the whole asked for.
    #[test]
    fn a_range_not_given_or_bytes_too_many_start_the_file_afresh() {
        let (data, sha, dir, to) = file("afresh");
        let len = data.len() as u64;
        for (answers, asked) in [
            (vec![whole()], vec!["120000-"]),
            (vec![refused(), whole()], vec!["120000-", ""]),
            (
                vec![another_part(1000, "300000"), whole()],
                vec!["120000-", ""],
            ),
            (vec![another_part(0, "*"), whole()], vec!["120000-", ""]),
        ] {
            left(&to, &data[..120_000]);
            let (url, ranges) = server(&data, answers);
            download(&url, &to, len, &sha, |_, _| {}).unwrap();
            assert_eq!(ranges.try_iter().collect::<Vec<_>>(), asked);
            assert_eq!(fs::read(&to).unwrap(), data);
            fs::remove_file(&to).unwrap();
        }
        let too_many: Vec<u8> = data.iter().chain(b"and more").copied().collect();
        left(&to, &too_many);
        let (url, ranges) = server(&data, vec![whole()]);
        download(&url, &to, len, &sha, |_, _| {}).unwrap();
        assert_eq!(ranges.recv().unwrap(), "", "all of it");
        assert_eq!(fs::read(&to).unwrap(), data);
        let _ = fs::remove_dir_all(&dir);
    }

    /// Kept whole (cut off between its last byte and its name), it is
    /// checked and put in place with no network asked; kept wrong at its
    /// length (torn, or another copy's), it is the disk's fault: the whole
    /// is fetched once more, and is the file.
    #[test]
    fn a_whole_download_kept_needs_no_network() {
        let (data, sha, dir, to) = file("kept-whole");
        let len = data.len() as u64;
        left(&to, &data);
        download("http://127.0.0.1:9/nowhere", &to, len, &sha, |_, _| {}).unwrap();
        assert_eq!(fs::read(&to).unwrap(), data);
        let mut wrong = data.clone();
        wrong[0] ^= 1;
        left(&to, &wrong);
        let (url, ranges) = server(&data, vec![whole()]);
        download(&url, &to, len, &sha, |_, _| {}).unwrap();
        assert_eq!(ranges.try_iter().collect::<Vec<_>>(), [""]);
        assert_eq!(fs::read(&to).unwrap(), data);
        let _ = fs::remove_dir_all(&dir);
    }

    /// Bytes kept that are wrong (a tail torn by a crash) make the file
    /// gone on from fail its check: it is filled once more from its start,
    /// and is the file then, not given up as another.
    #[test]
    fn kept_bytes_wrong_are_fetched_afresh_and_right() {
        let (data, sha, dir, to) = file("kept-wrong");
        let mut torn = data[..120_000].to_vec();
        torn[119_999] ^= 1;
        left(&to, &torn);
        let (url, ranges) = server(&data, vec![ranged(), whole()]);
        download(&url, &to, data.len() as u64, &sha, |_, _| {}).unwrap();
        assert_eq!(ranges.try_iter().collect::<Vec<_>>(), ["120000-", ""]);
        assert_eq!(fs::read(&to).unwrap(), data);
        assert!(!Resumable::part_of(&to).exists());
        let _ = fs::remove_dir_all(&dir);
    }

    /// A server whose file is longer than the one expected, or another of
    /// its length, sends what is not the file when fetched afresh: said so
    /// ([`Fault::Wrong`]), nothing kept.
    #[test]
    fn a_file_longer_than_expected_or_another_is_not_the_file() {
        let (data, sha, dir, to) = file("longer");
        for answer in [longer(), another_file()] {
            let (url, ranges) = server(&data, vec![answer]);
            let got = download(&url, &to, data.len() as u64, &sha, |_, _| {});
            assert!(matches!(got, Err(Fault::Wrong(_))), "{got:?}");
            assert_eq!(ranges.try_iter().count(), 1, "asked once");
            assert!(!Resumable::part_of(&to).exists());
            assert!(!to.exists());
        }
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn sha256_gives_the_standard_digests() {
        let hex = |data: &[u8]| {
            let mut h = Sha256::new();
            h.update(data);
            h.hex()
        };
        assert_eq!(
            hex(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        // Two blocks, fed in odd pieces.
        let long = b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq";
        let mut h = Sha256::new();
        for piece in long.chunks(7) {
            h.update(piece);
        }
        assert_eq!(
            h.hex(),
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
        );
    }
}
