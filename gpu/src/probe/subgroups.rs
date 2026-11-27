//! Subgroups on this GPU, where it has them ([`probe`]): their size as
//! measured in a workgroup of 256, and whether `subgroupAdd`,
//! `subgroupExclusiveAdd`, `subgroupShuffle`, `subgroupBroadcast`,
//! `subgroupBroadcastFirst` and `subgroupBallot` give the right results,
//! `subgroupAdd` in non-uniform control flow too (the even lanes only).

use std::collections::{BTreeMap, BTreeSet};

use super::gpu::{Bound, Gpu};
use super::text::check_lines;
use super::{Check, Miss};

/// Words written per invocation.
const RECORD: usize = 10;

/// Workgroups run.
const GROUPS: u32 = 64;

// naga takes subgroup operations with wgpu's SUBGROUP feature, without
// WGSL's `enable subgroups` (which it does not parse yet).
const SOURCE: &str = "@group(0) @binding(0) var<storage, read_write> out: array<u32>;
@compute @workgroup_size(256)
fn main(
    @builtin(local_invocation_index) i: u32,
    @builtin(workgroup_id) w: vec3<u32>,
    @builtin(subgroup_size) size: u32,
    @builtin(subgroup_invocation_id) id: u32,
) {
    let at = (w.x * 256u + i) * 10u;
    out[at] = size;
    out[at + 1u] = id;
    out[at + 2u] = subgroupBroadcastFirst(i);
    out[at + 3u] = subgroupAdd(1u);
    out[at + 4u] = subgroupAdd(id);
    out[at + 5u] = subgroupExclusiveAdd(1u);
    out[at + 6u] = subgroupShuffle(i, (id + 1u) % size);
    out[at + 7u] = subgroupBroadcast(i, 0u);
    let ballot = subgroupBallot(id % 3u == 0u);
    out[at + 8u] = countOneBits(ballot.x) + countOneBits(ballot.y) + countOneBits(ballot.z) + countOneBits(ballot.w);
    var even = 0u;
    if (id % 2u == 0u) {
        even = subgroupAdd(1u);
    }
    out[at + 9u] = even;
}
";

/// What the subgroups did.
#[derive(Clone, Debug)]
pub(super) struct Subgroups {
    /// The sizes seen, and the adapter's claim (its least and most).
    pub(super) sizes: BTreeSet<u32>,
    claimed: (u32, u32),
    /// Each operation's check over the invocations.
    pub(super) checks: Vec<Check>,
}

/// Subgroups on `gpu` (one with `SUBGROUP`), the adapter's claim
/// `claimed`.
pub(super) fn probe(gpu: &Gpu, claimed: (u32, u32)) -> Result<Subgroups, Miss> {
    let kernel = gpu.kernel(SOURCE, "main")?;
    let invocations = (GROUPS * 256) as usize;
    let out = gpu.zeros((invocations * RECORD * 4) as u64)?;
    gpu.run(&kernel, &[Bound::All(&out)], [GROUPS, 1, 1], 1)?;
    let words = gpu.read(&out)?;
    Ok(check(&words, claimed))
}

/// The records `words` (one of [`RECORD`] words an invocation, by
/// workgroup and local index) checked against what each subgroup should
/// give: its members found by the first one's local index.
pub(super) fn check(words: &[u32], claimed: (u32, u32)) -> Subgroups {
    let records: Vec<&[u32]> = words.chunks_exact(RECORD).collect();
    let mut sizes = BTreeSet::new();
    // (workgroup, the first's index) → members' (subgroup id, local index).
    let mut groups: BTreeMap<(usize, u32), Vec<(u32, u32)>> = BTreeMap::new();
    for (n, r) in records.iter().enumerate() {
        sizes.insert(r[0]);
        groups
            .entry((n / 256, r[2]))
            .or_default()
            .push((r[1], (n % 256) as u32));
    }
    let names = [
        "subgroupAdd(1)",
        "subgroupAdd(id)",
        "subgroupExclusiveAdd(1)",
        "subgroupShuffle",
        "subgroupBroadcast(x, 0)",
        "subgroupBallot",
        "subgroupAdd in a branch",
        "ids 0 to size - 1",
    ];
    let mut wrong = [0u64; 8];
    for (n, r) in records.iter().enumerate() {
        let members = &groups[&(n / 256, r[2])];
        let size = r[0];
        let count = members.len() as u32;
        let of = |id: u32| members.iter().find(|m| m.0 == id).map(|m| m.1);
        let fails = [
            r[3] != count,
            r[4] != members.iter().map(|m| m.0).sum::<u32>(),
            r[5] != r[1],
            Some(r[6]) != of((r[1] + 1) % size.max(1)),
            Some(r[7]) != of(0),
            r[8] != members.iter().filter(|m| m.0 % 3 == 0).count() as u32,
            r[1] % 2 == 0 && r[9] != members.iter().filter(|m| m.0 % 2 == 0).count() as u32,
            r[1] >= size || count != size,
        ];
        for (w, f) in wrong.iter_mut().zip(fails) {
            *w += u64::from(f);
        }
    }
    Subgroups {
        sizes,
        claimed,
        checks: names
            .into_iter()
            .zip(wrong)
            .map(|(name, wrong)| Check {
                name,
                cases: records.len() as u64,
                wrong,
            })
            .collect(),
    }
}

/// Whether every check held.
pub(super) fn correct(s: &Subgroups) -> bool {
    s.checks.iter().all(|c| c.wrong == 0)
}

/// The findings as report lines.
pub(super) fn lines(s: &Subgroups) -> Vec<String> {
    let sizes: Vec<String> = s.sizes.iter().map(u32::to_string).collect();
    let mut lines = vec![format!(
        "size: measured {} (the adapter claims {} to {})",
        sizes.join(", "),
        s.claimed.0,
        s.claimed.1
    )];
    lines.extend(check_lines(&s.checks));
    lines.push(if correct(s) {
        "verdict: every subgroup operation right".into()
    } else {
        "verdict: subgroup operations WRONG here".into()
    });
    lines
}
