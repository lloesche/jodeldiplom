//! The transformer and the autoencoder together ([`Painter`]): a picture
//! from a prompt's embeddings, a seed and its reference pictures, as one
//! plan: the references through the encoder into tokens, the steps'
//! modulations, four steps of the transformer and the sampler, the
//! latents through the decoder. The autoencoder's buffers are the
//! transformer's, taken in turns (their uses never overlap).

use std::path::Path;

use jodel_tensors::SafeTensors;

use super::encoder::{Embeds, WIDTH};
use super::gpu::{Loaded, Made};
use super::plan::{At, Plan};
use super::sampling::{embedded_time, noise, schedule};
use super::transformer::{self as tr, time_sinusoids, Tokens, Transformer, CHANNELS};
use super::vae::{Slots, Vae};
use super::weights::Packing;
use super::{Footprint, Precision};

/// The transformer's file in its repository, and the autoencoder's in
/// the diffusers release's.
const TRANSFORMER_FILE: &str = "flux-2-klein-4b-fp8.safetensors";
const VAE_FILE: &str = "vae/diffusion_pytorch_model.safetensors";

/// The sampler's steps (klein is distilled for four).
pub const STEPS: usize = 4;

/// The autoencoder's slots: its large ones are the transformer's.
const VAE: Slots = Slots {
    large: [tr::CAT, tr::QKV, tr::HEADS_SLOT, tr::H, tr::X],
    part: 12,
    stats: 13,
};
/// The picture made, and the first reference's pixels (the others' in the
/// slots after it).
const IMAGE: u8 = 14;

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

    /// The f32s read back.
    fn read(&self) -> (u8, usize, usize) {
        let t = self.tokens();
        match self.part {
            Part::Picture | Part::Decode => (IMAGE, 0, self.width * self.height * 3),
            Part::References => (
                tr::LAT,
                t.latent() * CHANNELS,
                (t.picture() - t.latent()) * CHANNELS,
            ),
            Part::FirstStep => (tr::PRED, 0, t.latent() * CHANNELS),
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

/// FLUX.2 klein 4B's transformer and its autoencoder on the GPU.
pub struct Painter {
    loaded: Loaded,
    transformer: Transformer,
    vae: Vae,
    precision: Precision,
    made: Option<(Key, Made)>,
}

impl Painter {
    /// The bytes it takes on the GPU while loaded, and while it makes a
    /// picture of `width × height` with references of these sizes.
    pub fn footprint(width: usize, height: usize, refs: &[(usize, usize)]) -> Footprint {
        let fp8 = tr::FP8_VALUES as u64;
        let (vae, vae_bytes) = Vae::shapes();
        let resident = fp8 + fp8 / (64 * 128) * 4 + 2 * tr::BF16_VALUES as u64 + vae_bytes;
        let key = Key {
            width,
            height,
            refs: refs.to_vec(),
            part: Part::Picture,
        };
        Footprint {
            resident,
            working: plan(None, &vae, &key, false).bytes(),
        }
    }

    /// The transformer from BFL's FP8 release (`transformer`: the folder
    /// of `black-forest-labs/FLUX.2-klein-4b-fp8`) and the autoencoder from
    /// the diffusers release (`klein`: the folder of
    /// `black-forest-labs/FLUX.2-klein-4B`, its `vae/`), each in its
    /// repository's own layout, on the GPU.
    pub fn load(transformer: &Path, klein: &Path, precision: Precision) -> Result<Painter, String> {
        let t = SafeTensors::open(&transformer.join(TRANSFORMER_FILE))?;
        let v = SafeTensors::open(&klein.join(VAE_FILE))?;
        let mut pack = Packing::new();
        let transformer = Transformer::pack(&t, &mut pack)?;
        let vae = Vae::pack(&v, &mut pack)?;
        let loaded = pack.upload("jodel-gpu painter")?;
        Ok(Painter {
            loaded,
            transformer,
            vae,
            precision,
            made: None,
        })
    }

    /// The GPU's name.
    pub fn gpu_name(&self) -> &str {
        &self.loaded.gpu.name
    }

    /// Whether the fast kernels run on tensor cores here.
    pub fn tensor_cores(&self) -> bool {
        self.loaded.gpu.coop && self.precision == Precision::Fast
    }

    /// Whether its device was lost (a driver reset, the GPU gone): it runs
    /// nothing again, and a new one is loaded instead.
    pub fn is_lost(&self) -> bool {
        self.loaded.lost()
    }

    /// What it takes on the GPU now: its weights, and its plan's buffers
    /// (kept for the next picture of the same size, until
    /// [`Painter::release`]).
    pub fn bytes(&self) -> Footprint {
        Footprint {
            resident: self.loaded.bytes(),
            working: self.made.as_ref().map_or(0, |(_, m)| m.bytes()),
        }
    }

    /// The bytes the GPU's allocator holds for every part now (where the
    /// backend tells).
    pub fn reserved(&self) -> Option<u64> {
        self.loaded.gpu.reserved()
    }

    /// Lets the plan's buffers go (the weights stay).
    pub fn release(&mut self) {
        self.made = None;
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
        if self.made.as_ref().is_none_or(|(k, _)| *k != key) {
            self.made = None;
            let fast = self.precision == Precision::Fast && self.loaded.gpu.coop;
            let p = plan(Some(&self.transformer), &self.vae, &key, fast);
            let made = Made::new(&self.loaded, &p, key.read().2)?;
            self.made = Some((key.clone(), made));
        }
        let (_, made) = self.made.as_mut().expect("made");
        let mut inputs: Vec<(u8, usize, &[f32])> = vec![
            (tr::LAT, 0, &lat),
            (tr::TXT, 0, &embeds.values),
            (tr::TIMES, 0, &times),
            (tr::ROPE, 0, &rope),
        ];
        for (k, p) in pictures.iter().enumerate() {
            inputs.push((IMAGE + k as u8, 0, p));
        }
        made.run(&self.loaded, &inputs, key.read())
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

/// The plan for `key` (without the transformer: its slots only, for a
/// footprint).
fn plan(transformer: Option<&Transformer>, vae: &Vae, key: &Key, fast: bool) -> Plan {
    let mut p = Plan::new(fast);
    let tokens = key.tokens();
    tr::need(&mut p, &tokens, STEPS);
    let latent = tokens.latent();
    if key.part != Part::Decode {
        let mut at = latent;
        for (k, &(w, h)) in key.refs.iter().enumerate() {
            let image = IMAGE + k as u8;
            p.need(image, w * h * 4);
            vae.plan_encode(
                &mut p,
                VAE,
                At::new(image, 0),
                (h, w),
                At::new(tr::LAT, at * CHANNELS),
            );
            at += (h / 16) * (w / 16);
        }
    }
    if let Some(t) = transformer.filter(|_| matches!(key.part, Part::Picture | Part::FirstStep)) {
        t.plan_mods(&mut p, STEPS);
        let sigmas = schedule(latent, STEPS);
        let steps = if key.part == Part::FirstStep {
            1
        } else {
            STEPS
        };
        for (i, &(s, next)) in sigmas.iter().take(steps).enumerate() {
            t.plan_step(&mut p, &tokens, (i, STEPS), next - s);
        }
    }
    if matches!(key.part, Part::Picture | Part::Decode) {
        p.need(IMAGE, key.width * key.height * 3);
        vae.plan_decode(
            &mut p,
            VAE,
            At::new(tr::LAT, 0),
            (key.height, key.width),
            At::new(IMAGE, 0),
        );
    }
    p
}
