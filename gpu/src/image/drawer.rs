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
use super::gpu::{Loaded, Made};
use super::plan::{At, Plan};
use super::sampling::{embedded_time, noise, schedule};
use super::transformer::{self as tr, time_sinusoids, Tokens, Transformer, CHANNELS};
use super::vae::{self, Vae};
use super::weights::{Packing, Source};
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
    fn read(&self, key: &Key) -> (u8, usize, usize) {
        let t = key.tokens();
        match key.part {
            Part::Picture | Part::Decode => (self.picture, 0, key.width * key.height * 3),
            Part::References => (
                self.tr.lat,
                t.latent() * CHANNELS,
                (t.picture() - t.latent()) * CHANNELS,
            ),
            Part::FirstStep => (self.tr.pred, 0, t.latent() * CHANNELS),
        }
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
            working: plan(&transformer, &vae, &key, false).0.bytes(),
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
        self.loaded.fast(self.precision)
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
            let fast = self.loaded.fast(self.precision);
            let (p, io) = plan(&self.transformer, &self.vae, &key, fast);
            let made = Made::new(&self.loaded, &p, io.read(&key).2)?;
            self.working = self.working.max(made.bytes());
            *slot = Some((key.clone(), io, made));
        }
        let (_, io, made) = slot.as_mut().expect("made");
        let mut inputs: Vec<(u8, usize, &[f32])> = vec![
            (io.tr.lat, 0, &lat),
            (io.tr.txt, 0, &embeds.values),
            (io.tr.times, 0, &times),
            (io.tr.rope, 0, &rope),
        ];
        for (&slot, p) in io.refs.iter().zip(&pictures) {
            inputs.push((slot, 0, p));
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

    /// Whether its device was lost (a driver reset, the GPU gone): it runs
    /// nothing again, and a new one is loaded instead.
    pub fn is_lost(&self) -> bool {
        self.loaded.lost()
    }

    /// The GPU's name.
    pub fn gpu_name(&self) -> &str {
        &self.loaded.gpu.name
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
fn plan(transformer: &Transformer, vae: &Vae, key: &Key, fast: bool) -> (Plan, Io) {
    let mut p = Plan::new(fast);
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
    use super::{pack, plan, Key, Packing, Part, Source};

    /// Every part's plan, with and without references, writes a slot it
    /// takes over before it reads it (no GPU: the weights' shapes alone).
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
                let (p, _) = plan(&transformer, &vae, &key, false);
                assert_eq!(p.check(), None, "{key:?}");
            }
        }
    }
}
