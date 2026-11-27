//! Gemma's tokeniser: byte-pair encoding over its 262 144 pieces, as the
//! checkpoint's `tokenizer.json` defines it (read once when the model is
//! packed, kept in the packed file).
//!
//! - The text is cut at the special tokens (`<bos>`, `<|turn>`, …), the
//!   longest that matches first; each special is its own id.
//! - Between them, every space becomes `▁`, and the whole stretch is one
//!   word (the checkpoint's split on spaces finds none after that).
//! - The word starts as its characters (a character with no piece of its
//!   own as the pieces of its UTF-8 bytes, `<0x41>`), then the pair of
//!   neighbours with the lowest merge rank is merged, the leftmost first,
//!   until no pair merges: what Hugging Face's `tokenizers` does.
//!
//! Decoding turns `▁` back into spaces and byte pieces into their bytes,
//! the specials left out.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};
use std::path::Path;

use jodel_tensors::json::Json;

/// The token that ends the model's turn, and the others that end its
/// answer (the generation config's).
pub const STOP: [u32; 3] = [1, 106, 50];

/// The tokeniser.
pub struct Tokenizer {
    /// Each id's piece, as bytes (`▁` as its UTF-8).
    pieces: Vec<Vec<u8>>,
    ids: HashMap<Vec<u8>, u32>,
    /// Each mergeable pair of ids: its rank and the id it makes.
    merges: HashMap<(u32, u32), (u32, u32)>,
    /// The special tokens, the longest first.
    specials: Vec<(Vec<u8>, u32)>,
    /// The piece of each byte (`<0x00>` …), where there is one.
    bytes: [Option<u32>; 256],
}

/// `▁`, which stands for a space in the pieces.
const SPACE: &str = "\u{2581}";

impl Tokenizer {
    /// The tokeniser of a checkpoint's `tokenizer.json` (a byte-pair model
    /// with its added tokens).
    pub fn from_json(path: &Path) -> Result<Tokenizer, String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let json = Json::parse(&text).ok_or("tokenizer.json is not JSON")?;
        let model = json.get("model").ok_or("tokenizer.json has no model")?;
        let Some(Json::Object(vocab)) = model.get("vocab") else {
            return Err("tokenizer.json has no vocabulary".into());
        };
        let mut pieces = vec![Vec::new(); vocab.len()];
        for (piece, id) in vocab {
            let id = id
                .as_f64()
                .map(|n| n as usize)
                .filter(|&i| i < pieces.len())
                .ok_or("a piece's id is out of range")?;
            pieces[id] = piece.as_bytes().to_vec();
        }
        let ids: HashMap<Vec<u8>, u32> = (0..)
            .zip(&pieces)
            .map(|(i, p): (u32, &Vec<u8>)| (p.clone(), i))
            .collect();
        let id = |s: &str| ids.get(s.as_bytes()).copied();
        let mut merges = Vec::new();
        for m in model
            .get("merges")
            .and_then(Json::as_array)
            .ok_or("tokenizer.json has no merges")?
        {
            // A pair as two strings, or (older files) one "a b".
            let (a, b) = match m {
                Json::Array(pair) => match pair.as_slice() {
                    [a, b] => (a.as_str(), b.as_str()),
                    _ => (None, None),
                },
                Json::String(s) => match s.split_once(' ') {
                    Some((a, b)) => (Some(a), Some(b)),
                    None => (None, None),
                },
                _ => (None, None),
            };
            let (a, b) = a.zip(b).ok_or("a merge that is not a pair")?;
            let (Some(x), Some(y)) = (id(a), id(b)) else {
                return Err(format!("a merge of pieces not in the vocabulary: {a} {b}"));
            };
            merges.push((x, y));
        }
        let mut specials = Vec::new();
        for t in json
            .get("added_tokens")
            .and_then(Json::as_array)
            .unwrap_or_default()
        {
            if t.get("special").and_then(Json::as_bool) == Some(true) {
                let content = t.get("content").and_then(Json::as_str);
                let tid = t.get("id").and_then(Json::as_f64);
                if let (Some(c), Some(i)) = (content, tid) {
                    specials.push(i as u32);
                    debug_assert_eq!(id(c), Some(i as u32));
                }
            }
        }
        Tokenizer::new(pieces, &merges, &specials)
    }

    /// From its pieces, its merges (pairs of ids, in rank order) and its
    /// special tokens' ids.
    fn new(
        pieces: Vec<Vec<u8>>,
        merges: &[(u32, u32)],
        specials: &[u32],
    ) -> Result<Tokenizer, String> {
        let ids: HashMap<Vec<u8>, u32> = (0..)
            .zip(&pieces)
            .map(|(i, p): (u32, &Vec<u8>)| (p.clone(), i))
            .collect();
        let mut merged = HashMap::with_capacity(merges.len());
        for (rank, &(a, b)) in (0..).zip(merges) {
            let piece = |i: u32| pieces.get(i as usize).ok_or("a merge's id is out of range");
            let joined = [piece(a)?.as_slice(), piece(b)?].concat();
            let made = *ids
                .get(&joined)
                .ok_or("a merge that makes no piece of the vocabulary")?;
            merged.entry((a, b)).or_insert((rank, made));
        }
        let mut specials: Vec<(Vec<u8>, u32)> = specials
            .iter()
            .filter_map(|&i| Some((pieces.get(i as usize)?.clone(), i)))
            .collect();
        specials.sort_by_key(|(p, _)| Reverse(p.len()));
        let mut bytes = [None; 256];
        for (b, slot) in bytes.iter_mut().enumerate() {
            *slot = ids.get(format!("<0x{b:02X}>").as_bytes()).copied();
        }
        Ok(Tokenizer {
            pieces,
            ids,
            merges: merged,
            specials,
            bytes,
        })
    }

    /// The number of pieces.
    pub fn len(&self) -> usize {
        self.pieces.len()
    }

    /// Whether it has no pieces.
    pub fn is_empty(&self) -> bool {
        self.pieces.is_empty()
    }

    /// Its bytes for the packed file: the pieces, the merges in rank
    /// order, the specials.
    pub(super) fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend((self.pieces.len() as u32).to_le_bytes());
        for p in &self.pieces {
            out.extend((p.len() as u16).to_le_bytes());
            out.extend(p);
        }
        let mut merges: Vec<(u32, u32, u32)> = self
            .merges
            .iter()
            .map(|(&(a, b), &(r, _))| (r, a, b))
            .collect();
        merges.sort_unstable();
        out.extend((merges.len() as u32).to_le_bytes());
        for (_, a, b) in merges {
            out.extend(a.to_le_bytes());
            out.extend(b.to_le_bytes());
        }
        out.extend((self.specials.len() as u32).to_le_bytes());
        for (_, id) in &self.specials {
            out.extend(id.to_le_bytes());
        }
        out
    }

    /// From [`Tokenizer::to_bytes`]'s.
    pub(super) fn from_bytes(data: &[u8]) -> Result<Tokenizer, String> {
        let mut r = Reader { data, at: 0 };
        let count = r.u32()? as usize;
        let mut pieces = Vec::with_capacity(count);
        for _ in 0..count {
            let len = r.take(2).map(|b| u16::from_le_bytes([b[0], b[1]]))? as usize;
            pieces.push(r.take(len)?.to_vec());
        }
        let count = r.u32()? as usize;
        let mut merges = Vec::with_capacity(count);
        for _ in 0..count {
            merges.push((r.u32()?, r.u32()?));
        }
        let count = r.u32()? as usize;
        let specials = (0..count).map(|_| r.u32()).collect::<Result<Vec<_>, _>>()?;
        Tokenizer::new(pieces, &merges, &specials)
    }

    /// The ids of `text`: its special tokens as they are, the rest
    /// byte-pair encoded.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let bytes = text.as_bytes();
        let mut out = Vec::new();
        let (mut from, mut at) = (0, 0);
        while at < bytes.len() {
            let special = self
                .specials
                .iter()
                .find(|(p, _)| bytes[at..].starts_with(p));
            match special {
                Some((p, id)) => {
                    self.word(&text[from..at], &mut out);
                    out.push(*id);
                    at += p.len();
                    from = at;
                }
                None => at += 1,
            }
        }
        self.word(&text[from..], &mut out);
        out
    }

    /// The ids of a stretch of text between specials (one word).
    fn word(&self, text: &str, out: &mut Vec<u32>) {
        if text.is_empty() {
            return;
        }
        let text = text.replace(' ', SPACE);
        // Each symbol: its id, and the symbols before and after it
        // (merged-away ones are skipped through these).
        let mut ids: Vec<u32> = Vec::with_capacity(text.len());
        let mut buf = [0u8; 4];
        for c in text.chars() {
            let s = c.encode_utf8(&mut buf);
            match self.ids.get(s.as_bytes()) {
                Some(&id) => ids.push(id),
                None => ids.extend(s.bytes().filter_map(|b| self.bytes[b as usize])),
            }
        }
        let n = ids.len();
        let mut prev: Vec<isize> = (0..n as isize).map(|i| i - 1).collect();
        let mut next: Vec<usize> = (1..=n).collect();
        let mut alive = vec![true; n];
        // The lowest rank first, then the leftmost.
        let mut queue = BinaryHeap::new();
        let pair = |a: u32, b: u32| self.merges.get(&(a, b)).copied();
        for i in 0..n.saturating_sub(1) {
            if let Some((rank, made)) = pair(ids[i], ids[i + 1]) {
                queue.push(Reverse((rank, i, made)));
            }
        }
        while let Some(Reverse((_, i, made))) = queue.pop() {
            if !alive[i] || next[i] >= n {
                continue;
            }
            let j = next[i];
            // A stale entry: the pair is not what it was.
            if pair(ids[i], ids[j]).map(|m| m.1) != Some(made) {
                continue;
            }
            ids[i] = made;
            alive[j] = false;
            next[i] = next[j];
            if next[j] < n {
                prev[next[j]] = i as isize;
            }
            if prev[i] >= 0 {
                let p = prev[i] as usize;
                if let Some((rank, m)) = pair(ids[p], ids[i]) {
                    queue.push(Reverse((rank, p, m)));
                }
            }
            if next[i] < n {
                if let Some((rank, m)) = pair(ids[i], ids[next[i]]) {
                    queue.push(Reverse((rank, i, m)));
                }
            }
        }
        out.extend((0..n).filter(|&i| alive[i]).map(|i| ids[i]));
    }

    /// The text of `ids`, the special tokens left out.
    pub fn decode(&self, ids: &[u32]) -> String {
        let mut bytes = Vec::new();
        for &id in ids {
            if self.specials.iter().any(|&(_, s)| s == id) {
                continue;
            }
            let Some(piece) = self.pieces.get(id as usize) else {
                continue;
            };
            match byte_of(piece) {
                Some(b) => bytes.push(b),
                None => match std::str::from_utf8(piece) {
                    Ok(s) => bytes.extend(s.replace(SPACE, " ").into_bytes()),
                    Err(_) => bytes.extend(piece),
                },
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }

    /// The prompt of one user turn as the model's chat template writes it
    /// (no thinking): the turn, then the model's turn opened.
    pub fn chat(&self, user: &str) -> Vec<u32> {
        self.encode(&format!(
            "<bos><|turn>user\n{}<turn|>\n<|turn>model\n",
            user.trim()
        ))
    }
}

/// Bytes read in order.
struct Reader<'a> {
    data: &'a [u8],
    at: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], String> {
        let s = self
            .data
            .get(self.at..self.at + n)
            .ok_or("a broken tokeniser")?;
        self.at += n;
        Ok(s)
    }

    fn u32(&mut self) -> Result<u32, String> {
        self.take(4)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
    }
}

/// The byte a byte piece (`<0x41>`) stands for.
fn byte_of(piece: &[u8]) -> Option<u8> {
    let hex = piece.strip_prefix(b"<0x")?.strip_suffix(b">")?;
    (hex.len() == 2)
        .then(|| u8::from_str_radix(std::str::from_utf8(hex).ok()?, 16).ok())
        .flatten()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A small vocabulary: bytes, letters, `▁`, a few merges, a special.
    fn small() -> Tokenizer {
        let mut pieces: Vec<Vec<u8>> = vec![b"<bos>".to_vec()];
        pieces.extend((0..=255u8).map(|b| format!("<0x{b:02X}>").into_bytes()));
        for p in ["a", "b", "c", SPACE, "ab", "abc", "\u{2581}a", "bc"] {
            pieces.push(p.as_bytes().to_vec());
        }
        let id = |s: &str| pieces.iter().position(|p| p == s.as_bytes()).unwrap() as u32;
        let merges = [
            (id("a"), id("b")),
            (id("ab"), id("c")),
            (id(SPACE), id("a")),
            (id("b"), id("c")),
        ];
        Tokenizer::new(pieces.clone(), &merges, &[0]).unwrap()
    }

    #[test]
    fn words_merge_by_rank_and_bytes_fall_back() {
        let t = small();
        let id = |s: &str| t.ids[s.as_bytes()];
        // "abc": ab first (rank 0), then abc (rank 1).
        assert_eq!(t.encode("abc"), vec![id("abc")]);
        // " abc": ▁a (rank 2) waits for ab (rank 0): ▁, abc.
        assert_eq!(t.encode(" abc"), vec![id(SPACE), id("abc")]);
        // "bc" alone merges; "é" is two byte pieces.
        assert_eq!(t.encode("bcé"), vec![id("bc"), 1 + 0xC3, 1 + 0xA9]);
        assert_eq!(t.encode("<bos>ab"), vec![0, id("ab")]);
        assert_eq!(t.decode(&t.encode("<bos> abc bcé")), " abc bcé");
    }

    #[test]
    fn it_survives_its_bytes() {
        let t = small();
        let back = Tokenizer::from_bytes(&t.to_bytes()).unwrap();
        for s in ["abc", " abc ab", "<bos>cba é"] {
            assert_eq!(back.encode(s), t.encode(s));
        }
    }
}
