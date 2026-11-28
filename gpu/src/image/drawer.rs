//! The transformer and the autoencoder together ([`Drawer`]): a picture
//! from a prompt's embeddings, a seed and its reference pictures, as one
//! plan: the references through the encoder into tokens, the steps'
//! modulations, four steps of the transformer and the sampler, the
//! latents through the decoder. The autoencoder's large buffers are the
//! transformer's, taken in turns (their uses never overlap), and the
//! picture's is the first reference's (its pixels taken in before the
//! picture is written).

use std::path::Path;
use std::sync::{Mutex, PoisonError};

use jodel_tensors::SafeTensors;

use super::encoder::{Embeds, WIDTH};
use super::gpu::{Loaded, Made, Tiles};
use super::plan::{At, Plan};
use super::sampling::{embedded_time, noise, schedule};
use super::transformer::{self as tr, time_sinusoids, Tokens, Transformer, CHANNELS};
use super::vae::{self, Vae};
use super::weights::{Packing, Source};
use crate::engine::{Input, Output};
use crate::{Footprint, Precision, Resident};

/// The files [`Drawer::load`] opens: the transformer's in BFL's FP8
/// release (`black-forest-labs/FLUX.2-klein-4b-fp8`), the autoencoder's in
/// the diffusers pipeline's (`black-forest-labs/FLUX.2-klein-4B`), each
/// within its repository (one each, listed as the encoder's are:
/// [`super::ENCODER_FILES`]).
const TRANSFORMER_FILE: &str = "flux-2-klein-4b-fp8.safetensors";
const AUTOENCODER_FILE: &str = "vae/diffusion_pytorch_model.safetensors";
pub const TRANSFORMER_FILES: &[&str] = &[TRANSFORMER_FILE];
pub const PIPELINE_FILES: &[&str] = &[AUTOENCODER_FILE];

/// The sampler's steps (klein is distilled for four).
pub const STEPS: usize = 4;

/// A picture: RGB, 8 bits a channel, row by row.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Picture {
    pub width: usize,
    pub height: usize,
    pub rgb: Vec<u8>,
}

/// A reference picture (a character's portrait, say): RGB, 8 bits a
/// channel, its sides multiples of 16, at most 1024 × 1024 pixels (the
/// pipeline's own limit; it takes larger ones shrunk, this does not).
pub type Reference = Picture;

/// What a plan makes (the picture; or, for checks, a part of the way).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Part {
    /// The whole picture (`h × w × 3` in -1..1).
    Picture,
    /// The references' tokens (128 each).
    References,
    /// The transformer's first prediction for the latents' tokens.
    FirstStep,
    /// The decoder alone over latent tokens given.
    Decode,
}

/// A plan's shape: the picture's size, the references' sizes, the part.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct Key {
    width: usize,
    height: usize,
    refs: Vec<(usize, usize)>,
    part: Part,
}

impl Key {
    fn tokens(&self) -> Tokens {
        Tokens {
            h: self.height / 16,
            w: self.width / 16,
            refs: self.refs.iter().map(|&(w, h)| (h / 16, w / 16)).collect(),
        }
    }
}

/// A plan's slots for what goes in and comes out: the transformer's, the
/// references' pixels, the picture.
struct Io {
    tr: tr::Slots,
    refs: Vec<u8>,
    picture: u8,
}

impl Io {
    /// The f32s `key`'s plan reads back.
    fn read(&self, key: &Key) -> Output<u8> {
        let t = key.tokens();
        let (buf, first, count) = match key.part {
            Part::Picture | Part::Decode => (self.picture, 0, key.width * key.height * 3),
            Part::References => (
                self.tr.lat,
                t.latent() * CHANNELS,
                (t.picture() - t.latent()) * CHANNELS,
            ),
            Part::FirstStep => (self.tr.pred, 0, t.latent() * CHANNELS),
        };
        Output { buf, first, count }
    }
}

/// Whether a picture of `width × height` can be made: sides multiples of
/// 16, its latent pixels in rows of 32 (the decoder's attention).
fn check_size(width: usize, height: usize, what: &str) -> Result<(), String> {
    if width == 0
        || height == 0
        || !width.is_multiple_of(16)
        || !height.is_multiple_of(16)
        || !((width / 16) * (height / 16)).is_multiple_of(8)
    {
        return Err(format!(
            "{what} of {width} × {height}: its sides must be multiples of 16, and (width/16)·(height/16) of 8"
        ));
    }
    Ok(())
}

/// The weights of the transformer (from `t`) and the autoencoder (from
/// `v`) into `pack`.
fn pack<'a>(
    (t, v): (Source<'a>, Source<'a>),
    pack: &mut Packing<'a>,
) -> Result<(Transformer, Vae), String> {
    Ok((Transformer::pack(t, pack)?, Vae::pack(v, pack)?))
}

/// FLUX.2 klein 4B's transformer and its autoencoder on the GPU.
pub struct Drawer {
    loaded: Loaded,
    transformer: Transformer,
    vae: Vae,
    precision: Precision,
    /// The plan last made (let go by [`Resident::release`]).
    made: Mutex<Option<(Key, Io, Made)>>,
    /// The bytes of the largest plan made.
    working: u64,
}

impl Drawer {
    /// What it takes on the GPU while loaded, and while it makes a picture
    /// of `width × height` with references of these sizes, before it is
    /// loaded: its weights packed as they would be, nothing read.
    pub fn planned(width: usize, height: usize, refs: &[(usize, usize)]) -> Footprint {
        let mut packing = Packing::new();
        let (transformer, vae) =
            pack((Source::Shapes, Source::Shapes), &mut packing).expect("shapes alone do not fail");
        let key = Key {
            width,
            height,
            refs: refs.to_vec(),
            part: Part::Picture,
        };
        Footprint {
            resident: packing.bytes(),
            working: plan(&transformer, &vae, &key, None).0.bytes(),
        }
    }

    /// The transformer from BFL's FP8 release (`transformer`: the folder
    /// of `black-forest-labs/FLUX.2-klein-4b-fp8`) and the autoencoder from
    /// the diffusers release (`klein`: the folder of
    /// `black-forest-labs/FLUX.2-klein-4B`, its `vae/`), each in its
    /// repository's own layout, on the GPU.
    pub fn load(transformer: &Path, klein: &Path, precision: Precision) -> Result<Drawer, String> {
        let t = SafeTensors::open(&transformer.join(TRANSFORMER_FILE))?;
        let v = SafeTensors::open(&klein.join(AUTOENCODER_FILE))?;
        let mut packing = Packing::new();
        let (transformer, vae) = pack(
            (Source::Checkpoint(&t), Source::Checkpoint(&v)),
            &mut packing,
        )?;
        let loaded = packing.upload("jodel-gpu drawer")?;
        Ok(Drawer {
            loaded,
            transformer,
            vae,
            precision,
            made: Mutex::new(None),
            working: 0,
        })
    }

    /// Whether the fast kernels run on tensor cores here.
    pub fn tensor_cores(&self) -> bool {
        self.loaded.fast(self.precision).is_some()
    }

    /// The bytes the GPU's allocator holds for every part now (where the
    /// backend tells).
    pub fn reserved(&self) -> Option<u64> {
        self.loaded.gpu.reserved()
    }

    /// A picture of `width × height` for `embeds`, its noise drawn from
    /// `seed` as the reference pipeline draws it, `refs` its reference
    /// pictures.
    pub fn paint(
        &mut self,
        embeds: &Embeds,
        seed: u64,
        (width, height): (usize, usize),
        refs: &[Reference],
    ) -> Result<Picture, String> {
        let start = noise(seed, CHANNELS * (width / 16) * (height / 16));
        let values = self.run(Part::Picture, embeds, &start, (width, height), refs)?;
        Ok(Picture {
            width,
            height,
            rgb: values
                .iter()
                .map(|v| ((v / 2.0 + 0.5).clamp(0.0, 1.0) * 255.0).round() as u8)
                .collect(),
        })
    }

    /// The part `part` of the way to a picture of `width × height` for
    /// `embeds` from the noise `start` (the latents as the pipeline draws
    /// them: 128 channels of `height/16 × width/16`; for [`Part::Decode`],
    /// the latent tokens to decode instead), with `refs`: its values, for
    /// checks against the reference.
    pub fn run(
        &mut self,
        part: Part,
        embeds: &Embeds,
        start: &[f32],
        (width, height): (usize, usize),
        refs: &[Reference],
    ) -> Result<Vec<f32>, String> {
        check_size(width, height, "a picture")?;
        for r in refs {
            check_size(r.width, r.height, "a reference")?;
            if r.width * r.height > 1024 * 1024 || r.rgb.len() != r.width * r.height * 3 {
                return Err(format!(
                    "a reference of {} × {}: at most 1024 × 1024 RGB",
                    r.width, r.height
                ));
            }
        }
        if embeds.values.len() != super::text::TOKENS * WIDTH {
            return Err("embeddings of the wrong size".into());
        }
        let key = Key {
            width,
            height,
            refs: refs.iter().map(|r| (r.width, r.height)).collect(),
            part,
        };
        let tokens = key.tokens();
        let latent = tokens.latent();
        if start.len() != latent * CHANNELS {
            return Err(format!(
                "{} starting values for {} latent tokens",
                start.len(),
                latent
            ));
        }
        // The latents as tokens (from channel-major noise; given as tokens
        // to decode).
        let lat: Vec<f32> = if part == Part::Decode {
            start.to_vec()
        } else {
            (0..latent)
                .flat_map(|t| (0..CHANNELS).map(move |c| start[c * latent + t]))
                .collect()
        };
        let times: Vec<f32> = schedule(latent, STEPS)
            .iter()
            .flat_map(|&(s, _)| time_sinusoids(embedded_time(s)))
            .collect();
        let rope = tokens.rope();
        let pictures: Vec<Vec<f32>> = refs.iter().map(pixels).collect();
        let slot = self.made.get_mut().unwrap_or_else(PoisonError::into_inner);
        if slot.as_ref().is_none_or(|(k, _, _)| *k != key) {
            *slot = None;
            let coop = self.loaded.fast(self.precision);
            let (p, io) = plan(&self.transformer, &self.vae, &key, coop);
            let made = Made::new(&self.loaded, &p, io.read(&key).count)?;
            self.working = self.working.max(made.bytes());
            *slot = Some((key.clone(), io, made));
        }
        let (_, io, made) = slot.as_mut().expect("made");
        fn whole(buf: u8, values: &[f32]) -> Input<'_, u8> {
            Input { buf, at: 0, values }
        }
        let mut inputs = vec![
            whole(io.tr.lat, &lat),
            whole(io.tr.txt, &embeds.values),
            whole(io.tr.times, &times),
            whole(io.tr.rope, &rope),
        ];
        for (&slot, p) in io.refs.iter().zip(&pictures) {
            inputs.push(whole(slot, p));
        }
        let read = io.read(&key);
        made.run(&self.loaded, &inputs, read)
    }
}

impl Drawer {
    /// What it holds on the GPU now: its weights, and its plan's buffers
    /// (kept for the next picture of the same size, until released).
    pub fn held(&self) -> Footprint {
        let made = self.made.lock().unwrap_or_else(PoisonError::into_inner);
        Footprint {
            resident: self.loaded.bytes(),
            working: made.as_ref().map_or(0, |(_, _, m)| m.bytes()),
        }
    }

    /// The GPU's name.
    pub fn gpu_name(&self) -> &str {
        &self.loaded.gpu.name
    }
}

impl crate::OnDevice for Drawer {
    fn lost(&self) -> bool {
        self.loaded.lost()
    }
}

impl Resident for Drawer {
    /// Its weights, and the buffers of the largest picture it has made
    /// (none before one; [`Drawer::planned`] says before it is loaded).
    fn footprint(&self) -> Footprint {
        Footprint {
            resident: self.loaded.bytes(),
            working: self.working,
        }
    }

    /// Lets the plan's buffers go (the weights stay); the next picture
    /// makes them again.
    fn release(&self) {
        *self.made.lock().unwrap_or_else(PoisonError::into_inner) = None;
    }
}

/// A reference's pixels as the encoder takes them: in -1..1, a fourth
/// channel of zeros.
fn pixels(r: &Reference) -> Vec<f32> {
    r.rgb
        .chunks_exact(3)
        .flat_map(|p| {
            let v = |c: u8| f32::from(c) / 255.0 * 2.0 - 1.0;
            [v(p[0]), v(p[1]), v(p[2]), 0.0]
        })
        .collect()
}

/// The plan for `key`, and its slots for what goes in and comes out.
fn plan(transformer: &Transformer, vae: &Vae, key: &Key, coop: Option<Tiles>) -> (Plan, Io) {
    let mut p = Plan::new(coop);
    let tokens = key.tokens();
    let slots = tr::Slots::new(&mut p, &tokens, STEPS);
    let vae_slots = vae::Slots {
        large: slots.large(),
        part: p.slot(0),
        stats: p.slot(0),
    };
    let latent = tokens.latent();
    let refs: Vec<u8> = key.refs.iter().map(|&(w, h)| p.slot(w * h * 4)).collect();
    // The picture is written over the first reference where there is one.
    let over_ref = !refs.is_empty();
    let picture = refs.first().copied().unwrap_or_else(|| p.slot(0));
    if key.part != Part::Decode {
        let mut at = latent;
        for (&(w, h), &image) in key.refs.iter().zip(&refs) {
            vae.plan_encode(
                &mut p,
                vae_slots,
                At::new(image, 0),
                (h, w),
                At::new(slots.lat, at * CHANNELS),
            );
            at += (h / 16) * (w / 16);
        }
    }
    if matches!(key.part, Part::Picture | Part::FirstStep) {
        // The transformer takes its large slots back from the autoencoder.
        for s in slots.large() {
            p.hand_over(s);
        }
        transformer.plan_mods(&mut p, &slots, STEPS);
        let sigmas = schedule(latent, STEPS);
        let steps = if key.part == Part::FirstStep {
            1
        } else {
            STEPS
        };
        for (i, &(s, next)) in sigmas.iter().take(steps).enumerate() {
            transformer.plan_step(&mut p, (&slots, &tokens), (i, STEPS), next - s);
        }
    }
    if matches!(key.part, Part::Picture | Part::Decode) {
        p.need(picture, key.width * key.height * 3);
        // The decoder takes the large slots (and the first reference's).
        for s in slots.large() {
            p.hand_over(s);
        }
        if over_ref {
            p.hand_over(picture);
        }
        vae.plan_decode(
            &mut p,
            vae_slots,
            At::new(slots.lat, 0),
            (key.height, key.width),
            At::new(picture, 0),
        );
    }
    let io = Io {
        tr: slots,
        refs,
        picture,
    };
    (p, io)
}

#[cfg(test)]
mod tests {
    use super::{pack, plan, Key, Packing, Part, Source, Tiles};
    use std::time::Duration;

    use crate::image::plan::{cut, flops, Op, Plan};
    use crate::image::Kernel;

    /// A picture of 1344 × 768 with a reference, on f32 and on either
    /// tiles (no GPU: the weights' shapes alone): every cut kernel's
    /// slices ([`cut`]) tile its dispatch's axis from its first workgroup
    /// to its last with none twice, their FLOPs add up to the whole's, and
    /// no slice takes more than its plan's budget (one workgroup along the
    /// axis at most twice it). The largest dispatch of each path, and the
    /// time it takes at its rate, are printed (`--nocapture`).
    #[test]
    fn no_dispatch_takes_more_than_its_budget() {
        let mut packing = Packing::new();
        let (transformer, vae) =
            pack((Source::Shapes, Source::Shapes), &mut packing).expect("shapes alone");
        let key = Key {
            width: 1344,
            height: 768,
            refs: vec![(1024, 1024)],
            part: Part::Picture,
        };
        // Whether `b` is the slice of the same dispatch right after `a`:
        // the same kernel, buffers and numbers but the first workgroup,
        // which is where `a`'s workgroups end.
        let continues = |a: &Op, b: &Op| {
            let Some(c) = cut(a.kernel) else {
                return false;
            };
            let (mut x, mut y) = (a.clone(), b.clone());
            (x.params[c.first], y.params[c.first]) = (0, 0);
            (x.groups[c.axis], y.groups[c.axis]) = (0, 0);
            a.kernel == b.kernel
                && (x.params, x.groups, x.buffers) == (y.params, y.groups, y.buffers)
                && b.params[c.first] == a.params[c.first] + a.groups[c.axis] as usize
        };
        for coop in [None, Some(Tiles::Sixteen), Some(Tiles::Eight)] {
            let (p, _) = plan(&transformer, &vae, &key, coop);
            let mut largest: Option<(f64, Kernel, [u32; 3])> = None;
            for slices in p.ops.chunk_by(continues) {
                let first = &slices[0];
                let Some(c) = cut(first.kernel) else {
                    continue;
                };
                let what = format!("{:?} on {coop:?}", first.kernel);
                assert_eq!(first.params[c.first], 0, "{what} starts at 0");
                let budget = p.budget(c);
                for s in slices {
                    let work = flops(s);
                    let one = s.groups[c.axis] == 1;
                    assert!(
                        work <= budget || (one && work <= 2.0 * budget),
                        "{what} {:?}: {} GFLOP, the budget {}",
                        s.groups,
                        work / 1e9,
                        budget / 1e9
                    );
                    if largest.is_none_or(|(w, _, _)| work / budget > w) {
                        largest = Some((work / budget, s.kernel, s.groups));
                    }
                }
                // The slices tile the whole axis, their work the whole's.
                let groups: usize = slices.iter().map(|s| s.groups[c.axis] as usize).sum();
                assert_eq!(groups, first.params[0].div_ceil(c.rows), "{what} tiled");
                let mut whole = first.clone();
                whole.groups[c.axis] = groups as u32;
                let (sum, whole): (f64, f64) = (slices.iter().map(flops).sum(), flops(&whole));
                assert!(
                    (sum - whole).abs() <= 1e-9 * whole,
                    "{what}: {sum} of {whole}"
                );
            }
            if let Some((share, kernel, groups)) = largest {
                let ms = share * p.piece * 1e3;
                eprintln!("{coop:?}: the largest dispatch {kernel:?} {groups:?}, about {ms:.1} ms");
            }
        }
    }

    /// A picture's submissions as a new background pace sizes them
    /// ([`crate::pace::Pieces`]: `end` for each, `learn` from each one's
    /// time once the next is made, as the run does), each dispatch timed at
    /// the rate its plan sizes its slices by (one not cut 20 µs, a
    /// submission 150 µs more): the first two take no more than two pieces,
    /// and none more than its path's bound, a little over what it takes
    /// now (10.6 pieces on f32, 3.3 on 16 × 16, 8.1 on 8 × 8: a few cheap
    /// dispatches, then twice as many costly ones, made before the one
    /// that would shrink them is settled). An eighth of the run, the first
    /// submission before, took some 3 s on the M4 Max. The first
    /// submissions' times and the longest are printed (`--nocapture`).
    #[test]
    fn a_pictures_submissions_are_short_from_the_first() {
        let mut packing = Packing::new();
        let (transformer, vae) =
            pack((Source::Shapes, Source::Shapes), &mut packing).expect("shapes alone");
        let key = Key {
            width: 1344,
            height: 768,
            refs: Vec::new(),
            part: Part::Picture,
        };
        let seconds =
            |p: &Plan, op: &Op| cut(op.kernel).map_or(20e-6, |c| flops(op) / p.budget(c) * p.piece);
        for coop in [None, Some(Tiles::Sixteen), Some(Tiles::Eight)] {
            let (p, _) = plan(&transformer, &vae, &key, coop);
            let all = p.ops.len();
            let mut pieces = crate::pace::Pieces::resting();
            let (mut at, mut times, mut running) = (0, Vec::new(), None);
            while at < all {
                let end = pieces.end(at, 1, all);
                let took = 150e-6 + p.ops[at..end].iter().map(|op| seconds(&p, op)).sum::<f64>();
                times.push(took);
                // As the run does: the one before is settled once this one
                // is made.
                if let Some((n, t)) = running.replace((end - at, took)) {
                    pieces.learn(n, all, Duration::from_secs_f64(t));
                }
                at = end;
            }
            let longest = times.iter().copied().fold(0.0, f64::max);
            let eighth: f64 = (p.ops[..all.div_ceil(8)].iter())
                .map(|op| seconds(&p, op))
                .sum();
            let ms: Vec<String> = times
                .iter()
                .take(8)
                .map(|t| format!("{:.1}", t * 1e3))
                .collect();
            eprintln!(
                "{coop:?}: {} submissions, the first {} ms, the longest {:.1} ms \
                 (an eighth of the run {:.0} ms)",
                times.len(),
                ms.join(" "),
                longest * 1e3,
                eighth * 1e3
            );
            let first = times[0].max(times[1]);
            assert!(first <= 2.0 * p.piece, "{coop:?}: the first {first} s");
            // The most pieces a submission takes on this path.
            let most = match coop {
                None => 11.5,
                Some(Tiles::Sixteen) => 3.6,
                Some(Tiles::Eight) => 9.0,
            };
            assert!(longest <= most * p.piece, "{coop:?}: {longest} s");
        }
    }

    /// Every part's plan, with and without references, on f32 and on
    /// either tiles, writes a slot it takes over before it reads it, and
    /// runs only kernels the program on those tiles has (no GPU: the
    /// weights' shapes alone).
    #[test]
    fn a_slot_taken_over_is_written_first() {
        let mut packing = Packing::new();
        let (transformer, vae) =
            pack((Source::Shapes, Source::Shapes), &mut packing).expect("shapes alone");
        for part in [
            Part::Picture,
            Part::References,
            Part::FirstStep,
            Part::Decode,
        ] {
            for refs in [vec![], vec![(256, 256)], vec![(128, 256), (256, 128)]] {
                let key = Key {
                    width: 256,
                    height: 128,
                    refs,
                    part,
                };
                for coop in [None, Some(Tiles::Sixteen), Some(Tiles::Eight)] {
                    let (p, _) = plan(&transformer, &vae, &key, coop);
                    assert_eq!(p.check(), None, "{key:?}");
                    let kernels = Kernel::on(coop);
                    assert!(
                        p.ops.iter().all(|op| kernels.contains(&op.kernel)),
                        "{key:?} on {coop:?}"
                    );
                }
            }
        }
    }
}
