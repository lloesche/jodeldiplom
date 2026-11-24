//! Karaoke mode's stage on the GPU: the note lane, the singer's arrow, its
//! trail and sparks, the lyrics at the bottom and the song's progress,
//! drawn over the pixmap each frame as instances of a few shapes, each a
//! signed distance in the fragment shader (a pill, a capsule, a star, a
//! disc, an arrow, a tube and its liquid) or a run of text from a glyph
//! atlas. One pipeline, premultiplied: a shape's light that is to add up
//! (a glow, a spark) comes with no coverage, so it adds; the rest goes
//! over. The bubbles' liquid, strain and overfilling are drawn here for
//! the stage's bubbles (tubes, [`super::snake`]) and the picture's alike
//! ([`Instance::liquid`], [`super::liquid`]).
//!
//! [`StageFrame`] is what the window gets each frame: the instances, and
//! the text the atlas did not hold yet ([`AtlasUpload`]).

use super::liquid::{amplitude, GROW};

/// The atlas's width and height, in pixels.
pub const ATLAS: u32 = 2048;

/// What a shape is ([`Instance::params`]`[0]`).
pub mod kind {
    /// A rounded box: `shape` its centre and half size, `params[3]` its
    /// corner radius (0: round ends), `params[1]` its glow, `params[2]` its
    /// outline. Over in `color`, outlined and glowing in `color2`.
    pub const PILL: f32 = 0.0;
    /// The part of a rounded box (as [`PILL`], round ends) between `uv[0]`
    /// and `uv[1]` across: `color` at its top, `color2` at its bottom,
    /// glowing in `color2` by `params[1]`.
    pub const FILL: f32 = 1.0;
    /// A capsule from (`shape[0]`, `shape[1]`) to (`shape[2]`,
    /// `shape[3]`), `params[1]` thick each way, glowing `params[2]` in
    /// `color2`; its core in `color`.
    pub const SEGMENT: f32 = 2.0;
    /// A four-pointed spark at (`shape[0]`, `shape[1]`), its rays
    /// `shape[2]` long, turned by `shape[3]`: light, added.
    pub const SPARK: f32 = 3.0;
    /// A disc at (`shape[0]`, `shape[1]`), `shape[2]` across each way, in
    /// `color`, glowing `shape[3]` in `color2`.
    pub const DISC: f32 = 4.0;
    /// An arrow pointing right, its tip at (`shape[0]`, `shape[1]`),
    /// `shape[2]` long: in `color`, glowing `params[1]` in `color2`.
    pub const ARROW: f32 = 5.0;
    /// Text from the atlas: `uv` its texels there, drawn over the quad;
    /// left of `params[1]` (pixels across the window) in `color2`, right of
    /// it in `color`. Where `shape[3]` is set, it pops round (`shape[0]`,
    /// `shape[1]`) from `shape[2]` and shows `shape[3]` seconds, as a
    /// [`BADGE`] does.
    pub const TEXT: f32 = 6.0;
    /// A word's bubble, a tube ([`crate::gui::snake`]): round its middle
    /// line (`params[3]` points from `shape[3]` in the path buffer),
    /// `shape[2]` thick each way, bulging round `shape[0..2]`; straining and
    /// bursting as its [`Instance::strain`] says; in `color`, outlined
    /// `params[2]` and glowing `params[1]` in `color2`.
    pub const TUBE: f32 = 7.0;
    /// A picture's bubble's liquid: a [`FILL`] (its ends `params[2]` round,
    /// 0: round) straining with its bubble, its front living where
    /// [`super::flags::FRONT`] (its fill jumped at `uv[2]`)
    /// ([`super::Instance::liquid`]).
    pub const LIQUID: f32 = 8.0;
    /// A streak's rainbow bands fanning out from (`shape[0]`, `shape[1]`)
    /// over a lane `shape[2]`, `shape[3]` half wide and high, within the
    /// lane (`uv`: its centre and half size); `params[1..]` its shine
    /// (level, from, since): light, added ([`crate::gui::rainbow::bands_of`]).
    pub const BANDS: f32 = 9.0;
    /// A streak's swirl round (`shape[0]`, `shape[1]`), `shape[2]` out; its
    /// shine as [`BANDS`]' ([`crate::gui::rainbow::swirl_of`]).
    pub const SWIRL: f32 = 10.0;
    /// A streak's rainbow edge round a bar at `shape` (centre, half size);
    /// its shine as [`BANDS`]' ([`crate::gui::rainbow::rim_of`]).
    pub const RIM: f32 = 11.0;
    /// The shine meter ([`crate::gui::glory::meter_at`]): `shape` its middle
    /// and half size, `params[1..]` its fill from, to and since.
    pub const METER: f32 = 12.0;
    /// The ring of the time left ([`crate::gui::glory::ring_at`]): `shape`
    /// its middle, radius and thickness, `params[1]` the share left,
    /// `strain[0..3]` the shine.
    pub const RING: f32 = 13.0;
    /// A phrase's mark on the progress bar: a disc at (`shape[0]`,
    /// `shape[1]`), `shape[2]` across, in `color`, popping from `params[1]`
    /// ([`crate::gui::glory::mark_pop`]).
    pub const MARK: f32 = 14.0;
    /// Great's glow round the lane (`shape` its middle and half size) in
    /// `color`, `params[1]` its reach, `params[2]` its strength: light.
    pub const GLOW: f32 = 15.0;
    /// Amazing's flames along an edge from `shape[0]` to `shape[2]` at
    /// `shape[1]`, outward `shape[3]` (1 down, -1 up), `params[1]` high,
    /// `params[2]` strong ([`crate::gui::glory::blaze`]): light.
    pub const BLAZE: f32 = 16.0;
    /// A badge: a round [`PILL`] (`shape` its middle and half size) in
    /// `color`, ringed `params[3]` wide in `color2`, popping round its
    /// middle from `params[1]` (the singing's time) and showing
    /// `params[2]` seconds ([`crate::gui::glory::label_pop`]).
    pub const BADGE: f32 = 17.0;
    /// A tube's liquid ([`TUBE`]'s shape) along it from `uv[0]` to `uv[1]`
    /// across, as [`LIQUID`]'s.
    pub const TUBE_LIQUID: f32 = 18.0;
    /// A burst of rays round (`shape[0]`, `shape[1]`), `shape[2]` their
    /// reach, `shape[3]` of them, turned `params[1]`, in `color`: light
    /// ([`crate::gui::trophy::rays_at`]).
    pub const RAYS: f32 = 19.0;
    /// A gem at (`shape[0]`, `shape[1]`), `shape[2]` its size, in `color`,
    /// glowing `params[1]` in `color2`, its glint at `params[2]`
    /// ([`crate::gui::trophy::gem_at`]).
    pub const GEM: f32 = 20.0;
    /// A medal at (`shape[0]`, `shape[1]`), `shape[2]` its radius, a star
    /// badge where `shape[3]` is 1: from `color2` (deep) to `color`
    /// (light), glowing `params[1]`, its sheen at `params[2]`
    /// ([`crate::gui::trophy::medal_at`]).
    pub const MEDAL: f32 = 21.0;
}

/// The flags of a bubble or its liquid, or of text ([`Instance::strain`]`[3]`).
pub mod flags {
    /// It is bursting: overfilling from `strain[1]`, gone once it tears.
    pub const POPPING: u32 = 1;
    /// Its liquid's front is a living surface (it is not full).
    pub const FRONT: u32 = 2;
    /// Text in a streak's rainbow, its shine (level, from, since) in
    /// `strain[0..3]` ([`crate::gui::rainbow::in_rainbow`]).
    pub const RAINBOW: u32 = 4;
}

/// One shape, as the shader takes it: the quad it covers (`x0, y0, x1,
/// y1` in window pixels, its glow included) and what [`kind`] says of the
/// rest. Colours are sRGB, 0 to 1, with their alpha.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Instance {
    /// The quad: left, top, right, bottom.
    pub quad: [f32; 4],
    /// The shape's own numbers.
    pub shape: [f32; 4],
    /// Its colour.
    pub color: [f32; 4],
    /// Its second colour.
    pub color2: [f32; 4],
    /// Its kind, then three numbers.
    pub params: [f32; 4],
    /// Atlas texels, or a fill's span.
    pub uv: [f32; 4],
    /// A bubble's strain (and its liquid's): its pressure, when it bursts,
    /// its wobble's phase, its [`flags`].
    pub strain: [f32; 4],
}

// SAFETY: plain f32s in a `repr(C)` struct, no padding: any bytes are one.
unsafe impl bytemuck::Zeroable for Instance {}
// SAFETY: as above.
unsafe impl bytemuck::Pod for Instance {}

/// How a bubble strains and bursts: how much it strains (0 to 1), its
/// wobble's phase (radians), and when it bursts (the singing's time), once
/// it has begun to.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Strain {
    /// How much it strains.
    pub pressure: f32,
    /// Its wobble's phase.
    pub phase: f32,
    /// When it burst, while it overfills.
    pub popping: Option<f32>,
}

/// A bubble's skin: its body, its edge (outline and glow), how far its
/// glow reaches and how wide its outline is.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Skin {
    /// Its body.
    pub body: [f32; 4],
    /// Its edge.
    pub edge: [f32; 4],
    /// Its glow's reach, pixels.
    pub glow: f32,
    /// Its outline's width, pixels.
    pub outline: f32,
}

/// A liquid's colours: at its top, at its bottom, and how far it glows.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Tint {
    /// At its top.
    pub top: [f32; 4],
    /// At its bottom.
    pub bottom: [f32; 4],
    /// Its glow's reach, pixels.
    pub glow: f32,
}

/// A bubble's liquid: from and to across (pixels), whether its front is
/// living (it is not full), when its fill jumped (the singing's time), and
/// its tint.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Pour {
    /// From and to across.
    pub span: (f32, f32),
    /// Whether its front is living.
    pub front: bool,
    /// When its fill jumped.
    pub jumped: f32,
    /// Its tint.
    pub tint: Tint,
}

impl Instance {
    /// The liquid of a bubble at `centre`, `half` its size, its ends
    /// `radius` round (0: round ends), poured as `pour` says, straining with
    /// it; none where too little shows.
    pub fn liquid(
        centre: (f32, f32),
        half: (f32, f32),
        radius: f32,
        pour: &Pour,
        strain: &Strain,
    ) -> Option<Instance> {
        if pour.span.1 - pour.span.0 < 0.5 {
            return None;
        }
        let ripple = if pour.front {
            amplitude(half.1) * 3.5 + 1.0
        } else {
            0.0
        };
        let (mx, my) = Self::room(half, pour.tint.glow, ripple);
        Some(Instance {
            quad: [centre.0 - mx, centre.1 - my, centre.0 + mx, centre.1 + my],
            shape: [centre.0, centre.1, half.0, half.1],
            color: pour.tint.top,
            color2: pour.tint.bottom,
            params: [kind::LIQUID, pour.tint.glow, radius, 0.0],
            uv: [pour.span.0, pour.span.1, pour.jumped, 0.0],
            strain: Self::strain_of(strain, pour.front),
        })
    }

    /// How far a bubble `half` its size reaches from its centre, glowing
    /// `glow` and its front's waves `ripple` further across: bulged to
    /// 1 + [`GROW`].
    fn room(half: (f32, f32), glow: f32, ripple: f32) -> (f32, f32) {
        let big = 1.0 + GROW;
        (
            half.0 * big + glow + ripple + 1.0,
            half.1 * big + glow + 1.0,
        )
    }

    /// A bubble's `s` strain as the shader takes it, its liquid's front
    /// living where `front`.
    pub(super) fn strain_of(s: &Strain, front: bool) -> [f32; 4] {
        let mut bits = 0;
        if s.popping.is_some() {
            bits |= flags::POPPING;
        }
        if front {
            bits |= flags::FRONT;
        }
        [s.pressure, s.popping.unwrap_or(0.0), s.phase, bits as f32]
    }

    /// The strain a bubble or its liquid was made with.
    pub fn strained(&self) -> Strain {
        let bits = self.strain[3] as u32;
        Strain {
            pressure: self.strain[0],
            phase: self.strain[2],
            popping: (bits & flags::POPPING != 0).then_some(self.strain[1]),
        }
    }

    /// Whether this is a liquid whose front is living.
    pub fn living(&self) -> bool {
        let liquid = [kind::LIQUID, kind::TUBE_LIQUID].contains(&self.params[0]);
        liquid && self.strain[3] as u32 & flags::FRONT != 0
    }
}

/// Text new to the atlas: its place there and its coverage, row by row.
#[derive(Clone, Debug, PartialEq)]
pub struct AtlasUpload {
    /// Left, top, width, height, in the atlas.
    pub rect: (u32, u32, u32, u32),
    /// The coverage, `width × height`.
    pub coverage: Vec<u8>,
}

/// A frame of the stage.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct StageFrame {
    /// The shapes, back to front.
    pub instances: Vec<Instance>,
    /// The shapes over everything, the light's bloom too (a tier's name
    /// on its badge), back to front.
    pub top: Vec<Instance>,
    /// Text to put into the atlas first.
    pub uploads: Vec<AtlasUpload>,
    /// The tubes' middle lines, point by point (`x, y`, two unused).
    pub paths: Vec<[f32; 4]>,
    /// The bursts showing ([`super::particles`]).
    pub sprays: Vec<super::particles::Spray>,
    /// The shapes that are light to bloom, drawn apart ([`super::bloom`]).
    pub lights: Vec<Instance>,
    /// The bursts that are light to bloom.
    pub light_sprays: Vec<super::particles::Spray>,
    /// The singing's time, in seconds: what the bubbles' strain and the
    /// bursts go by.
    pub clock: f32,
}

/// `stage` (none: none) with the picture's bubbles' liquid (`bubbles`)
/// and bursts (`sprays`) drawn over it too, at `clock` (the singing's
/// time) where there is no stage.
pub fn with_picture(
    stage: Option<StageFrame>,
    bubbles: &[Instance],
    sprays: &[super::particles::Spray],
    clock: f32,
) -> Option<StageFrame> {
    match stage {
        Some(mut f) => {
            f.instances.extend_from_slice(bubbles);
            f.sprays.extend_from_slice(sprays);
            Some(f)
        }
        None => (!bubbles.is_empty() || !sprays.is_empty()).then(|| StageFrame {
            instances: bubbles.to_vec(),
            sprays: sprays.to_vec(),
            clock,
            ..Default::default()
        }),
    }
}

/// The kinds and flags as the shader takes them.
fn kinds_wgsl() -> String {
    let k = |f: f32| f as u32;
    format!(
        "const K_PILL: u32 = {}u;
const K_FILL: u32 = {}u;
const K_SEGMENT: u32 = {}u;
const K_SPARK: u32 = {}u;
const K_DISC: u32 = {}u;
const K_ARROW: u32 = {}u;
const K_TUBE: u32 = {}u;
const K_LIQUID: u32 = {}u;
const K_BANDS: u32 = {}u;
const K_SWIRL: u32 = {}u;
const K_RIM: u32 = {}u;
const K_METER: u32 = {}u;
const K_RING: u32 = {}u;
const K_MARK: u32 = {}u;
const K_GLOW: u32 = {}u;
const K_BLAZE: u32 = {}u;
const K_TEXT: u32 = {}u;
const K_BADGE: u32 = {}u;
const K_TUBE_LIQUID: u32 = {}u;
const K_RAYS: u32 = {}u;
const K_GEM: u32 = {}u;
const K_MEDAL: u32 = {}u;
",
        k(kind::PILL),
        k(kind::FILL),
        k(kind::SEGMENT),
        k(kind::SPARK),
        k(kind::DISC),
        k(kind::ARROW),
        k(kind::TUBE),
        k(kind::LIQUID),
        k(kind::BANDS),
        k(kind::SWIRL),
        k(kind::RIM),
        k(kind::METER),
        k(kind::RING),
        k(kind::MARK),
        k(kind::GLOW),
        k(kind::BLAZE),
        k(kind::TEXT),
        k(kind::BADGE),
        k(kind::TUBE_LIQUID),
        k(kind::RAYS),
        k(kind::GEM),
        k(kind::MEDAL),
    )
}

/// The stage's shader: `stage.wgsl`, its kinds ([`kind`]), the liquid's
/// maths ([`super::liquid::wgsl`]), a tube's nearest point
/// ([`super::snake::wgsl`]), a streak's shine ([`super::rainbow::stage_wgsl`]),
/// the glory ([`super::glory::wgsl`]) and the result's shapes
/// ([`super::trophy::wgsl`]).
pub fn shader() -> String {
    format!(
        "{}{}{}{}{}{}{}",
        kinds_wgsl(),
        include_str!("stage.wgsl"),
        super::liquid::wgsl(),
        super::snake::wgsl(),
        super::rainbow::stage_wgsl(),
        super::glory::wgsl(),
        super::trophy::wgsl()
    )
}

/// How a pass's shapes go onto what is drawn: over it (premultiplied,
/// covering it as much as they cover), or added to it (light).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Blend {
    /// Over it.
    Over,
    /// Added.
    Add,
}

impl Blend {
    /// Its blend, premultiplied.
    pub fn state(self) -> wgpu::BlendState {
        let c = wgpu::BlendComponent {
            src_factor: wgpu::BlendFactor::One,
            dst_factor: match self {
                Blend::Over => wgpu::BlendFactor::OneMinusSrcAlpha,
                Blend::Add => wgpu::BlendFactor::One,
            },
            operation: wgpu::BlendOperation::Add,
        };
        wgpu::BlendState { color: c, alpha: c }
    }
}

/// Whether `format` takes linear light from the shaders: an sRGB one
/// (encoded as it is written) or a float one (light as it is).
pub fn takes_linear(format: wgpu::TextureFormat) -> bool {
    format.is_srgb()
        || matches!(
            format,
            wgpu::TextureFormat::Rgba16Float | wgpu::TextureFormat::Rgba32Float
        )
}

/// What a stage pass draws next: its shapes (back to front), those over
/// everything (drawn apart: [`StagePass::draw_top`]), the text to put into
/// the atlas first, and the singing's time.
#[derive(Clone, Copy, Debug, Default)]
pub struct Feed<'a> {
    /// The shapes.
    pub instances: &'a [Instance],
    /// The shapes over everything.
    pub top: &'a [Instance],
    /// The text.
    pub uploads: &'a [AtlasUpload],
    /// The tubes' middle lines.
    pub paths: &'a [[f32; 4]],
    /// The time.
    pub clock: f32,
}

impl StageFrame {
    /// Its stage, as the stage pass draws it.
    pub fn feed(&self) -> Feed<'_> {
        Feed {
            instances: &self.instances,
            top: &self.top,
            uploads: &self.uploads,
            paths: &self.paths,
            clock: self.clock,
        }
    }

    /// Its light, as the light's pass draws it (no text: the light has
    /// none).
    pub fn light_feed(&self) -> Feed<'_> {
        // Nor tubes: their points are not sent with the light.
        let tube = |i: &Instance| [kind::TUBE, kind::TUBE_LIQUID].contains(&i.params[0]);
        debug_assert!(!self.lights.iter().any(tube), "a tube among the lights");
        Feed {
            instances: &self.lights,
            clock: self.clock,
            ..Feed::default()
        }
    }
}

/// The stage's pipeline and what it reads.
pub struct StagePass {
    pipeline: wgpu::RenderPipeline,
    layout: wgpu::BindGroupLayout,
    bind: wgpu::BindGroup,
    uniforms: wgpu::Buffer,
    sampler: wgpu::Sampler,
    /// The text's atlas, made with the first text put into it (a pass
    /// never given any has a texel of it), and its view.
    atlas: Option<wgpu::Texture>,
    atlas_view: wgpu::TextureView,
    /// The tubes' middle lines (a point at least).
    paths: wgpu::Buffer,
    instances: Option<wgpu::Buffer>,
    /// The instances in the buffer for the next frame: the shapes, then
    /// those over everything.
    count: (u32, u32),
    linear: bool,
}

/// A texture of the atlas's format, `side` square, and its view.
fn atlas_texture(device: &wgpu::Device, side: u32) -> (wgpu::Texture, wgpu::TextureView) {
    let atlas = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("jodeldiplom stage atlas"),
        size: wgpu::Extent3d {
            width: side,
            height: side,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::R8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let view = atlas.create_view(&wgpu::TextureViewDescriptor::default());
    (atlas, view)
}

/// A buffer of tubes' points, `size` bytes.
fn paths_buffer(device: &wgpu::Device, size: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("jodeldiplom stage paths"),
        size: size.max(16),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

impl StagePass {
    /// For frames of `format`, its shapes going on as `blend` says.
    pub fn new(device: &wgpu::Device, format: wgpu::TextureFormat, blend: Blend) -> StagePass {
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("jodeldiplom stage"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(32),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(16),
                    },
                    count: None,
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("jodeldiplom stage"),
            bind_group_layouts: &[Some(&layout)],
            ..Default::default()
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("jodeldiplom stage"),
            source: wgpu::ShaderSource::Wgsl(shader().into()),
        });
        let attributes = wgpu::vertex_attr_array![
            0 => Float32x4, 1 => Float32x4, 2 => Float32x4,
            3 => Float32x4, 4 => Float32x4, 5 => Float32x4, 6 => Float32x4
        ];
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("jodeldiplom stage"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs"),
                compilation_options: Default::default(),
                buffers: &[Some(wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<Instance>() as u64,
                    step_mode: wgpu::VertexStepMode::Instance,
                    attributes: &attributes,
                })],
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fs"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(blend.state()),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });
        let uniforms = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("jodeldiplom stage"),
            size: 32,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("jodeldiplom stage"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let (_, atlas_view) = atlas_texture(device, 1);
        let paths = paths_buffer(device, 16);
        let bind = StagePass::bind(device, (&layout, &uniforms, &sampler), &atlas_view, &paths);
        StagePass {
            pipeline,
            layout,
            bind,
            uniforms,
            sampler,
            atlas: None,
            atlas_view,
            paths,
            instances: None,
            count: (0, 0),
            linear: takes_linear(format),
        }
    }

    /// The bind group of `uniforms`, `atlas`, `sampler` and `paths` in
    /// `layout`.
    fn bind(
        device: &wgpu::Device,
        (layout, uniforms, sampler): (&wgpu::BindGroupLayout, &wgpu::Buffer, &wgpu::Sampler),
        atlas: &wgpu::TextureView,
        paths: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("jodeldiplom stage"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniforms.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(atlas),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: paths.as_entire_binding(),
                },
            ],
        })
    }

    /// What it draws next (`None`: nothing), for a window of `size`: the
    /// text put into its atlas, its shapes sent over.
    pub fn set(
        &mut self,
        (device, queue): (&wgpu::Device, &wgpu::Queue),
        feed: Option<Feed>,
        size: (u32, u32),
    ) {
        self.count = (0, 0);
        let Some(feed) = feed else {
            return;
        };
        let mut rebind = false;
        if self.atlas.is_none() && !feed.uploads.is_empty() {
            let (atlas, view) = atlas_texture(device, ATLAS);
            (self.atlas, self.atlas_view, rebind) = (Some(atlas), view, true);
        }
        let paths: &[u8] = bytemuck::cast_slice(feed.paths);
        if self.paths.size() < paths.len() as u64 {
            self.paths = paths_buffer(device, (paths.len() as u64).next_power_of_two());
            rebind = true;
        }
        if !paths.is_empty() {
            queue.write_buffer(&self.paths, 0, paths);
        }
        if rebind {
            let with = (&self.layout, &self.uniforms, &self.sampler);
            self.bind = StagePass::bind(device, with, &self.atlas_view, &self.paths);
        }
        for up in feed.uploads {
            let (x, y, w, h) = up.rect;
            let Some(atlas) = self.atlas.as_ref().filter(|_| w > 0 && h > 0) else {
                continue;
            };
            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: atlas,
                    mip_level: 0,
                    origin: wgpu::Origin3d { x, y, z: 0 },
                    aspect: wgpu::TextureAspect::All,
                },
                &up.coverage,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(w),
                    rows_per_image: Some(h),
                },
                wgpu::Extent3d {
                    width: w,
                    height: h,
                    depth_or_array_layers: 1,
                },
            );
        }
        if feed.instances.is_empty() && feed.top.is_empty() {
            return;
        }
        let bytes: Vec<u8> = (bytemuck::cast_slice::<Instance, u8>(feed.instances).iter())
            .chain(bytemuck::cast_slice::<Instance, u8>(feed.top))
            .copied()
            .collect();
        let fits = self
            .instances
            .as_ref()
            .is_some_and(|b| b.size() >= bytes.len() as u64);
        if !fits {
            self.instances = Some(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("jodeldiplom stage"),
                size: (bytes.len() as u64).next_power_of_two().max(4096),
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
        }
        if let Some(buffer) = &self.instances {
            queue.write_buffer(buffer, 0, &bytes);
        }
        let mut uniforms = [0u8; 32];
        uniforms[0..4].copy_from_slice(&(size.0 as f32).to_le_bytes());
        uniforms[4..8].copy_from_slice(&(size.1 as f32).to_le_bytes());
        uniforms[8..12].copy_from_slice(&(if self.linear { 1f32 } else { 0.0 }).to_le_bytes());
        uniforms[12..16].copy_from_slice(&(ATLAS as f32).to_le_bytes());
        uniforms[16..20].copy_from_slice(&feed.clock.to_le_bytes());
        queue.write_buffer(&self.uniforms, 0, &uniforms);
        self.count = (feed.instances.len() as u32, feed.top.len() as u32);
    }

    /// Draws its shapes into `pass` (nothing without any).
    pub fn draw(&self, pass: &mut wgpu::RenderPass) {
        self.draw_range(pass, 0..self.count.0);
    }

    /// Draws its shapes over everything into `pass`.
    pub fn draw_top(&self, pass: &mut wgpu::RenderPass) {
        self.draw_range(pass, self.count.0..self.count.0 + self.count.1);
    }

    fn draw_range(&self, pass: &mut wgpu::RenderPass, range: std::ops::Range<u32>) {
        let Some(buffer) = self.instances.as_ref().filter(|_| !range.is_empty()) else {
            return;
        };
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind, &[]);
        pass.set_vertex_buffer(0, buffer.slice(..));
        pass.draw(0..4, range);
    }
}
