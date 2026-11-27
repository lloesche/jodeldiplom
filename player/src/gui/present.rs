//! Karaoke mode's frames on the GPU ([`Painter`], inside iced's frame
//! through [`super::stage_view`]): its backdrop ([`super::backdrop`]) is
//! the ground, a triangle over the whole window reading it; its stage and
//! its bursts are drawn over that, the light that blooms over them, the
//! effects' screen passes over that ([`super::fx`]), and the shapes over
//! everything last ([`Painter::PARTS`]).

mod offscreen;
mod stamps;

pub(crate) use offscreen::read_back;
pub use offscreen::Offscreen;

use super::backdrop::BackdropFrame;
use super::fx::Music;
use crate::gui::stage::gpu::StageFrame;

/// Karaoke mode's ground: its backdrop over the whole window, read
/// smoothly (it is drawn at a share of the window's size).
const GROUND: &str = r"
@group(0) @binding(0) var backdrop: texture_2d<f32>;
@group(0) @binding(1) var soft: sampler;

struct Out {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

// One triangle that covers the window, the backdrop's corners on its.
@vertex
fn vs(@builtin(vertex_index) i: u32) -> Out {
    let x = f32((i << 1u) & 2u) * 2.0 - 1.0;
    let y = f32(i & 2u) * 2.0 - 1.0;
    var o: Out;
    o.pos = vec4<f32>(x, y, 0.0, 1.0);
    o.uv = vec2<f32>(x * 0.5 + 0.5, 0.5 - y * 0.5);
    return o;
}

@fragment
fn fs(o: Out) -> @location(0) vec4<f32> {
    return vec4<f32>(textureSampleLevel(backdrop, soft, o.uv, 0.0).rgb, 1.0);
}
";

/// Karaoke mode's passes: its ground, its stage over it and all that goes
/// over that.
pub(super) struct Painter {
    device: wgpu::Device,
    queue: wgpu::Queue,
    /// The ground ([`GROUND`]) and what it reads (made again with the
    /// backdrop's texture), in a frame of `size`.
    ground: wgpu::RenderPipeline,
    ground_layout: wgpu::BindGroupLayout,
    ground_bind: Option<wgpu::BindGroup>,
    size: (u32, u32),
    /// The stage, drawn over the ground, and its bursts over that.
    stage: crate::gui::stage::gpu::StagePass,
    particles: super::particles::ParticlePass,
    /// The light that blooms ([`super::bloom`]).
    light: super::bloom::Light,
    /// The backdrop, how it is read, and which of its textures the ground
    /// reads ([`BackdropPass::generation`]).
    ///
    /// [`BackdropPass::generation`]: super::backdrop::gpu::BackdropPass::generation
    backdrop: super::backdrop::gpu::BackdropPass,
    /// The song's look over the backdrop, where one draws: the ground
    /// reads its output instead.
    looks: super::backdrop::looks::LookPass,
    soft: wgpu::Sampler,
    bound: (u64, u64),
    /// The GPU's time after each step of a frame, where it is measured
    /// ([`Offscreen::timed`]).
    stamps: Option<crate::gui::present::stamps::Stamps>,
    /// The window's pixels to a point (its scale).
    density: f32,
}

/// A step of the window's pass: what it is called and what it draws.
type Part = (&'static str, fn(&Painter, &mut wgpu::RenderPass<'_>));

/// An adapter and a device opened on it.
type Opened = (wgpu::Adapter, wgpu::Device, wgpu::Queue);

/// A device on `adapter` (not a software renderer), with limits for a
/// texture the size of any window, and `features`.
pub(super) fn open(adapter: wgpu::Adapter, features: wgpu::Features) -> Result<Opened, String> {
    let info = adapter.get_info();
    if info.device_type == wgpu::DeviceType::Cpu {
        return Err(format!("{}: a software renderer", info.name));
    }
    let limits = wgpu::Limits::downlevel_defaults().using_resolution(adapter.limits());
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("jodeldiplom window"),
        required_features: features,
        required_limits: limits,
        ..Default::default()
    }))
    .map_err(|e| format!("{}: {e}", info.name))?;
    Ok((adapter, device, queue))
}

impl Painter {
    /// For frames of `format`.
    pub(super) fn new(
        device: wgpu::Device,
        queue: wgpu::Queue,
        format: wgpu::TextureFormat,
    ) -> Painter {
        let (ground, ground_layout) = ground_pipeline(&device, format);
        let stage = crate::gui::stage::gpu::StagePass::new(
            &device,
            format,
            crate::gui::stage::gpu::Blend::Over,
        );
        let particles = super::particles::ParticlePass::new(
            &device,
            format,
            crate::gui::stage::gpu::Blend::Over,
        );
        let light = super::bloom::Light::new(&device, format);
        // The backdrop made in linear light where the frame takes it so.
        let backdrop = super::backdrop::gpu::BackdropPass::new(&device, &queue, format.is_srgb());
        let looks = super::backdrop::looks::LookPass::new(&device, format.is_srgb());
        let soft = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("jodeldiplom ground"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        Painter {
            device,
            queue,
            ground,
            ground_layout,
            ground_bind: None,
            size: (1, 1),
            stage,
            particles,
            light,
            backdrop,
            looks,
            soft,
            bound: (0, 0),
            stamps: None,
            density: 1.0,
        }
    }

    /// Marks the end of step `name` where a frame's steps are timed.
    fn mark(&mut self, name: &'static str) {
        if let Some(stamps) = &mut self.stamps {
            stamps.mark((&self.device, &self.queue), name);
        }
    }

    /// The backdrop for the next frame (`None`: none, its textures
    /// dropped), in a frame of `size`: drawn now, at its share of the
    /// window ([`backdrop_share`] of [`BackdropFrame::pictures`]).
    fn set_backdrop(&mut self, frame: Option<&BackdropFrame>, size: (u32, u32)) {
        let share = frame.map_or(1.0, |f| backdrop_share(f.pictures(size).into_iter()));
        self.size = size;
        let drawn = |side: u32| ((side as f32 * share).round() as u32).max(1);
        let size = (drawn(size.0), drawn(size.1));
        self.backdrop.set((&self.device, &self.queue), frame, size);
        self.mark("backdrop");
        let look = frame.and_then(|f| f.look.as_ref());
        let gpu = (&self.device, &self.queue);
        self.looks.set(gpu, look, self.backdrop.under(), self.size);
        self.mark("look");
    }

    /// The stage for the next frame (`None`: none), in a frame of `size`.
    fn set_stage(&mut self, frame: Option<&StageFrame>, size: (u32, u32)) {
        let gpu = (&self.device, &self.queue);
        self.stage.set(gpu, frame.map(StageFrame::feed), size);
        self.mark("uploads");
        let gpu = (&self.device, &self.queue);
        let (sprays, music, shake, time) = frame
            .map_or((&[][..], Music::NEUTRAL, [0.0; 2], 0.0), |f| {
                (&f.sprays[..], f.music, f.shake, f.clock)
            });
        self.particles.set(gpu, (sprays, &music, shake), time, size);
        self.mark("sprays");
        let gpu = (&self.device, &self.queue);
        self.light.set(gpu, frame, (size, self.density));
        self.mark("light");
        self.light.blur((&self.device, &self.queue));
        self.mark("bloom");
    }

    /// The next frame, of `size` pixels (`density` to a point): `backdrop`
    /// drawn as its ground, and `stage` over that.
    pub(super) fn prepare(
        &mut self,
        (size, density): ((u32, u32), f32),
        stage: Option<&StageFrame>,
        backdrop: Option<&BackdropFrame>,
    ) {
        let size = (size.0.max(1), size.1.max(1));
        self.density = density;
        self.set_backdrop(backdrop, size);
        self.set_stage(stage, size);
        self.bind();
    }

    /// The frame prepared, into `pass` (its viewport the window's): the
    /// ground, the stage and its bursts, the light and its bloom added over
    /// them (drawn and blurred as the frame was prepared), the effects'
    /// screen passes, and the stage's shapes over everything
    /// ([`Self::PARTS`]).
    pub(super) fn draw(&self, pass: &mut wgpu::RenderPass<'_>) {
        for (_, part) in Self::PARTS {
            part(self, pass);
        }
    }

    /// The window's pass, step by step: the ground first (the backdrop,
    /// where there is one), then what goes over it.
    const PARTS: [Part; 6] = [
        ("ground", |p, pass| {
            if let (Some(bind), true) = (&p.ground_bind, p.backdrop.shows()) {
                pass.set_pipeline(&p.ground);
                pass.set_bind_group(0, bind, &[]);
                pass.draw(0..3, 0..1);
            }
        }),
        ("stage", |p, pass| p.stage.draw(pass)),
        ("bursts", |p, pass| p.particles.draw(pass)),
        ("composite", |p, pass| p.light.composite(pass)),
        ("fx", |p, pass| p.stage.draw_screen(pass)),
        ("top", |p, pass| p.stage.draw_top(pass)),
    ];

    /// The device's queue it draws with.
    pub(super) fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    /// What the ground reads, made where the backdrop's texture or the
    /// look's output is new: the look's, while one draws, else the
    /// backdrop's.
    fn bind(&mut self) {
        let now = (self.backdrop.generation(), self.looks.generation());
        if now != self.bound {
            self.ground_bind = None;
            self.bound = now;
        }
        if self.ground_bind.is_none() {
            self.ground_bind = Some(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("jodeldiplom ground"),
                layout: &self.ground_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(
                            self.looks.out().unwrap_or(self.backdrop.under()),
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&self.soft),
                    },
                ],
            }));
        }
    }

    /// The frame onto `target` in a pass of its own (a pass a step where
    /// the steps are timed), with whatever else `then` adds to the
    /// commands; submitted.
    fn paint(&mut self, target: &wgpu::Texture, then: impl FnOnce(&mut wgpu::CommandEncoder)) {
        self.bind();
        let view = target.create_view(&wgpu::TextureViewDescriptor::default());
        let device = self.device.clone();
        let encoder = || {
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("jodeldiplom window"),
            })
        };
        let mut commands = encoder();
        if self.stamps.is_some() {
            for (k, (name, part)) in Self::PARTS.into_iter().enumerate() {
                part(self, &mut begin(&mut commands, &view, k == 0));
                self.queue
                    .submit([std::mem::replace(&mut commands, encoder()).finish()]);
                self.mark(name);
            }
        } else {
            self.draw(&mut begin(&mut commands, &view, true));
        }
        then(&mut commands);
        self.queue.submit([commands.finish()]);
    }
}

/// A pass into `view`, cleared to black first (`clear`) or over what is
/// there.
fn begin<'e>(
    encoder: &'e mut wgpu::CommandEncoder,
    view: &wgpu::TextureView,
    clear: bool,
) -> wgpu::RenderPass<'e> {
    let load = if clear {
        wgpu::LoadOp::Clear(wgpu::Color::BLACK)
    } else {
        wgpu::LoadOp::Load
    };
    encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("jodeldiplom window"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view,
            depth_slice: None,
            resolve_target: None,
            ops: wgpu::Operations {
                load,
                store: wgpu::StoreOp::Store,
            },
        })],
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    })
}

/// Karaoke mode's ground's pipeline ([`GROUND`]) for frames of `format`,
/// and the layout of what it reads.
fn ground_pipeline(
    device: &wgpu::Device,
    format: wgpu::TextureFormat,
) -> (wgpu::RenderPipeline, wgpu::BindGroupLayout) {
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("jodeldiplom ground"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
        ],
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("jodeldiplom ground"),
        bind_group_layouts: &[Some(&layout)],
        ..Default::default()
    });
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("jodeldiplom ground"),
        source: wgpu::ShaderSource::Wgsl(GROUND.into()),
    });
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("jodeldiplom ground"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &module,
            entry_point: Some("vs"),
            compilation_options: Default::default(),
            buffers: &[],
        },
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        fragment: Some(wgpu::FragmentState {
            module: &module,
            entry_point: Some("fs"),
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        multiview_mask: None,
        cache: None,
    });
    (pipeline, layout)
}

/// The least share of the window's width and height the backdrop is drawn
/// at.
const LEAST_SHARE: f32 = 0.5;
/// The steps the share goes in: its texture is made again only when the
/// card's size or the cover changes by one.
const SHARE_STEP: f32 = 1.0 / 16.0;

/// The share of the window's width and height Karaoke mode's backdrop is
/// drawn at (the ground reads it smoothly over the whole window), its
/// covers' cards (each its size in window pixels, and the texels of its
/// pictures, the larger of the layers' and the cover's): as few pixels as
/// still give each texel on a card one of its own, so a card enlarging
/// its cover (a 1000-pixel one on a Retina screen) is drawn with every
/// detail it has, a quarter to a half of the pixels; the rest of the
/// backdrop is blur and soft light. The whole window where a card shows
/// its pictures at their size or smaller.
fn backdrop_share(cards: impl Iterator<Item = ((f32, f32), (u32, u32))>) -> f32 {
    let needed = cards
        .map(|((w, h), (tw, th))| (tw as f32 / w.max(1.0)).max(th as f32 / h.max(1.0)))
        .fold(LEAST_SHARE, f32::max);
    ((needed / SHARE_STEP).ceil() * SHARE_STEP).min(1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_card_enlarging_its_cover_is_drawn_at_the_cover_s_texels() {
        // A 1024 cover on a 16-inch MacBook Pro's card (1650 pixels).
        let card = (1650.0, 1650.0);
        let share = backdrop_share([(card, (1024, 1024))].into_iter());
        assert_eq!(share, 0.625);
        assert!(card.0 * share >= 1024.0);
        // Its 3000-pixel picture: every pixel.
        assert_eq!(backdrop_share([(card, (3000, 3000))].into_iter()), 1.0);
        // A 5K screen's card, a small cover: half, no less.
        assert_eq!(
            backdrop_share([((2400.0, 2400.0), (600, 600))].into_iter()),
            0.5
        );
        // Two covers crossing: the one with more texels.
        let two = [(card, (1024, 1024)), (card, (1400, 1400))];
        assert_eq!(backdrop_share(two.into_iter()), 0.875);
        // No cover: the ground, blurred.
        assert_eq!(backdrop_share(std::iter::empty()), LEAST_SHARE);
    }
}
