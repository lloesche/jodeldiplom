//! Byte-pair encoding over a vocabulary's ids ([`Merges`]), as Hugging
//! Face's `tokenizers` does it: of a word's neighbouring pairs, the one
//! with the lowest merge rank is merged first (the leftmost of equals),
//! until no pair merges. The models' tokenisers each make a word's first
//! ids their own way (Gemma's characters with byte pieces as a fallback,
//! Qwen's bytes as GPT-2's characters) and merge them here. A text is cut
//! at its special pieces first ([`Specials`]: the longest that starts at
//! a byte, found by that byte), and only the text between them merged.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, HashSet};

/// A vocabulary's merges: each mergeable pair of ids, its rank and the id
/// it makes.
pub struct Merges {
    pairs: HashMap<(u32, u32), (u32, u32)>,
}

impl Merges {
    /// The merges of `pairs` (ids, in rank order) over `pieces` (each id's
    /// bytes, `ids` their ids): what each makes is its two pieces joined,
    /// which must be a piece; neither may be empty (so what a merge makes
    /// is longer than either side: [`Merges::word`] relies on it). A pair
    /// listed twice keeps its first rank.
    pub fn new(
        pieces: &[Vec<u8>],
        ids: &HashMap<Vec<u8>, u32>,
        pairs: &[(u32, u32)],
    ) -> Result<Merges, String> {
        let mut merged = HashMap::with_capacity(pairs.len());
        for (rank, &(a, b)) in (0..).zip(pairs) {
            let piece = |i: u32| pieces.get(i as usize).ok_or("a merge's id is out of range");
            let (left, right) = (piece(a)?, piece(b)?);
            if left.is_empty() || right.is_empty() {
                return Err("a merge with an empty piece".into());
            }
            let joined = [left.as_slice(), right].concat();
            let made = *ids
                .get(&joined)
                .ok_or("a merge that makes no piece of the vocabulary")?;
            merged.entry((a, b)).or_insert((rank, made));
        }
        Ok(Merges { pairs: merged })
    }

    /// The rank and the id of the pair `a`, `b`, where it merges.
    fn pair(&self, a: u32, b: u32) -> Option<(u32, u32)> {
        self.pairs.get(&(a, b)).copied()
    }

    /// The word of `ids` merged, its ids appended to `out`.
    pub fn word(&self, mut ids: Vec<u32>, out: &mut Vec<u32>) {
        let n = ids.len();
        // Each symbol's neighbours (merged-away ones are skipped through
        // these).
        let mut prev: Vec<isize> = (0..n as isize).map(|i| i - 1).collect();
        let mut next: Vec<usize> = (1..=n).collect();
        let mut alive = vec![true; n];
        // The lowest rank first, then the leftmost.
        let mut queue = BinaryHeap::new();
        for i in 0..n.saturating_sub(1) {
            if let Some((rank, made)) = self.pair(ids[i], ids[i + 1]) {
                queue.push(Reverse((rank, i, made)));
            }
        }
        while let Some(Reverse((_, i, made))) = queue.pop() {
            if !alive[i] || next[i] >= n {
                continue;
            }
            let j = next[i];
            // A stale entry: the pair is not what it was. The pair at `i`
            // only grows (a side merges with its right neighbour; no piece
            // is empty), and an id is its bytes, so the same id made means
            // the same pair.
            if self.pair(ids[i], ids[j]).map(|m| m.1) != Some(made) {
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
                if let Some((rank, m)) = self.pair(ids[p], ids[i]) {
                    queue.push(Reverse((rank, p, m)));
                }
            }
            if next[i] < n {
                if let Some((rank, m)) = self.pair(ids[i], ids[next[i]]) {
                    queue.push(Reverse((rank, i, m)));
                }
            }
        }
        out.extend((0..n).filter(|&i| alive[i]).map(|i| ids[i]));
    }
}

/// A stretch of text cut at its special tokens ([`Specials::split`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Part<'t> {
    /// Text between specials, to encode.
    Text(&'t str),
    /// A special token, its id as it is.
    Special(u32),
}

/// A vocabulary's special tokens: found in text where they stand (the
/// longest of those starting at the first place one does), and known by
/// id.
pub struct Specials {
    /// The specials by their first byte, the longest first.
    by_first: Vec<Vec<(Vec<u8>, u32)>>,
    /// Their ids (as many as there are specials, whatever the ids).
    is: HashSet<u32>,
}

impl Specials {
    /// The specials `list` (each its text and id).
    pub fn new(list: Vec<(Vec<u8>, u32)>) -> Specials {
        let mut by_first = vec![Vec::new(); 256];
        let mut is = HashSet::with_capacity(list.len());
        for (text, id) in list {
            is.insert(id);
            if let Some(&first) = text.first() {
                by_first[usize::from(first)].push((text, id));
            }
        }
        for list in &mut by_first {
            list.sort_by_key(|(t, _)| Reverse(t.len()));
        }
        Specials { by_first, is }
    }

    /// Whether `id` is a special.
    pub fn is_special(&self, id: u32) -> bool {
        self.is.contains(&id)
    }

    /// `text` cut at its specials, in order (no empty text between them).
    pub fn split<'t>(&self, text: &'t str) -> Vec<Part<'t>> {
        let bytes = text.as_bytes();
        let mut out = Vec::new();
        let (mut from, mut at) = (0, 0);
        while at < bytes.len() {
            let found = self.by_first[usize::from(bytes[at])]
                .iter()
                .find(|(t, _)| bytes[at..].starts_with(t));
            match found {
                Some((t, id)) => {
                    if from < at {
                        out.push(Part::Text(&text[from..at]));
                    }
                    out.push(Part::Special(*id));
                    at += t.len();
                    from = at;
                }
                None => at += 1,
            }
        }
        if from < bytes.len() {
            out.push(Part::Text(&text[from..]));
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn specials_cut_text_the_longest_first() {
        let s = Specials::new(vec![
            (b"<a>".to_vec(), 7),
            (b"<a><b>".to_vec(), 8),
            (b"<b>".to_vec(), 9),
        ]);
        assert_eq!(
            s.split("x<a><b>y<b><a>"),
            [
                Part::Text("x"),
                Part::Special(8),
                Part::Text("y"),
                Part::Special(9),
                Part::Special(7)
            ]
        );
        assert_eq!(s.split("é<"), [Part::Text("é<")]);
        assert!(s.is_special(9) && !s.is_special(3) && !s.is_special(1000));
    }

    #[test]
    fn the_lowest_rank_merges_first_the_leftmost_of_equals() {
        let pieces: Vec<Vec<u8>> = ["a", "b", "c", "ab", "bc", "abc", "aa"]
            .iter()
            .map(|p| p.as_bytes().to_vec())
            .collect();
        let ids: HashMap<Vec<u8>, u32> = (0..).zip(&pieces).map(|(i, p)| (p.clone(), i)).collect();
        // bc first, then a·bc; ab never (bc took its b).
        let m = Merges::new(&pieces, &ids, &[(1, 2), (0, 4), (0, 1)]).unwrap();
        let mut out = Vec::new();
        m.word(vec![0, 1, 2], &mut out);
        assert_eq!(out, [5]);
        // aa twice: the leftmost pair first.
        let m = Merges::new(&pieces, &ids, &[(0, 0)]).unwrap();
        out.clear();
        m.word(vec![0, 0, 0], &mut out);
        assert_eq!(out, [6, 0]);
        assert!(Merges::new(&pieces, &ids, &[(2, 2)]).is_err());
        // An empty piece: a merge with it would make what one side is.
        let mut with_empty = pieces.clone();
        with_empty.push(Vec::new());
        let err = Merges::new(&with_empty, &ids, &[(0, 7)]).err().unwrap();
        assert!(err.contains("empty"), "{err}");
    }
}
