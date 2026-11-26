//! The song's look over the backdrop on the GPU ([`LookPass`]): after the
//! backdrop is drawn (at its share of the window), each pass of the look
//! in force ([`jodel_looks`], resolved for the frame: [`LookFrame`]) draws
//! the whole window once, a pipeline a pass (its own shader, made the
//! first time it draws), each reading what the one before drew, into
//! textures the window's size (half floats: the feedback's echoes fade
//! smoothly); the last into one of two outputs, the other the frame
//! before, which a pass that feeds back reads. The window's ground reads
//! the output ([`LookPass::out`]) where the backdrop's was: the stage, the
//! lyrics and the effects are drawn over the look, never under it. With no
//! look, nothing is drawn and nothing held: the ground reads the backdrop
//! as before.

use std::sync::Arc;

use jodel_looks::{Ctx, Driver, Pass, Signals};

/// A look's passes for a frame, each with its numbers
/// ([`jodel_looks::Checked::frame`]).
#[derive(Clone, Debug, Default)]
pub struct LookFrame {
    /// The passes, in order.
    pub passes: Vec<(&'static Pass, Ctx)>,
}

/// The textures' format.
const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
/// A pass's numbers' place in the uniform buffer (its slots aligned for
/// any GPU).
const SLOT: u64 = 512;

/// A texture the window's size and its view.
struct Target {
    _texture: wgpu::Texture,
    view: wgpu::TextureView,
}

/// The passes' pipelines, the textures they draw into, their numbers, and
/// which output is this frame's.
pub struct LookPass {
    linear: bool,
    layout: wgpu::BindGroupLayout,
    pipeline_layout: wgpu::PipelineLayout,
    sampler: wgpu::Sampler,
    pipelines: Vec<(&'static str, wgpu::RenderPipeline)>,
    uniforms: wgpu::Buffer,
    /// Between passes (two, in turn), and the two outputs.
    chain: Vec<Target>,
    outs: Vec<Target>,
    size: (u32, u32),
    /// Which output is this frame's (none: no look).
    now: Option<usize>,
    /// Changes whenever [`LookPass::out`] does.
    generation: u64,
}

impl LookPass {
    /// For a window whose textures are read as linear light (`linear`)
    /// or as they are.
    pub fn new(device: &wgpu::Device, linear: bool) -> LookPass {
        let entry = |binding, ty| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty,
            count: None,
        };
        let texture = wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("jodeldiplom look"),
            entries: &[
                entry(
                    0,
                    wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(std::mem::size_of::<Ctx>() as u64),
                    },
                ),
                entry(1, texture),
                entry(2, texture),
                entry(
                    3,
                    wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                ),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("jodeldiplom look"),
            bind_group_layouts: &[Some(&layout)],
            ..Default::default()
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("jodeldiplom look"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let uniforms = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("jodeldiplom look"),
            size: SLOT * jodel_looks::guard::MOST_LAYERS as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        LookPass {
            linear,
            layout,
            pipeline_layout,
            sampler,
            pipelines: Vec::new(),
            uniforms,
            chain: Vec::new(),
            outs: Vec::new(),
            size: (0, 0),
            now: None,
            generation: 0,
        }
    }

    /// What the ground reads instead of the backdrop: this frame's
    /// output, while a look draws.
    pub fn out(&self) -> Option<&wgpu::TextureView> {
        self.now.map(|i| &self.outs[i].view)
    }

    /// Changes whenever [`LookPass::out`] does.
    pub fn generation(&self) -> u64 {
        self.generation
    }

    /// `pass`'s pipeline, made the first time.
    fn pipeline(&mut self, device: &wgpu::Device, pass: &'static Pass) -> usize {
        if let Some(i) = self.pipelines.iter().position(|(n, _)| *n == pass.name) {
            return i;
        }
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(pass.name),
            source: wgpu::ShaderSource::Wgsl(
                jodel_looks::wgsl::pass_shader(pass, self.linear).into(),
            ),
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some(pass.name),
            layout: Some(&self.pipeline_layout),
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
                    format: FORMAT,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });
        self.pipelines.push((pass.name, pipeline));
        self.pipelines.len() - 1
    }

    /// A texture of `size` to draw into and read.
    fn target(device: &wgpu::Device, size: (u32, u32)) -> Target {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("jodeldiplom look"),
            size: wgpu::Extent3d {
                width: size.0,
                height: size.1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        Target {
            _texture: texture,
            view,
        }
    }

    /// The look of the next frame (`frame`; none: none, its textures let
    /// go) over `under` (the backdrop as drawn), in a window of `size`:
    /// drawn now.
    pub fn set(
        &mut self,
        (device, queue): (&wgpu::Device, &wgpu::Queue),
        frame: Option<&LookFrame>,
        under: &wgpu::TextureView,
        size: (u32, u32),
    ) {
        let Some(frame) = frame.filter(|f| !f.passes.is_empty()) else {
            if self.now.take().is_some() {
                self.chain.clear();
                self.outs.clear();
                self.size = (0, 0);
                self.generation += 1;
            }
            return;
        };
        let size = (size.0.max(1), size.1.max(1));
        if self.size != size {
            self.chain = (0..2).map(|_| Self::target(device, size)).collect();
            self.outs = (0..2).map(|_| Self::target(device, size)).collect();
            self.size = size;
        }
        let now = self.now.map_or(0, |i| 1 - i);
        let pipelines: Vec<usize> = (frame.passes.iter())
            .map(|(p, _)| self.pipeline(device, p))
            .collect();
        for (i, (_, ctx)) in frame.passes.iter().enumerate() {
            queue.write_buffer(&self.uniforms, i as u64 * SLOT, bytemuck::bytes_of(ctx));
        }
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("jodeldiplom look"),
        });
        let prev = &self.outs[1 - now].view;
        let last = frame.passes.len() - 1;
        for (i, &k) in pipelines.iter().enumerate() {
            let input = if i == 0 {
                under
            } else {
                &self.chain[(i - 1) % 2].view
            };
            let into = if i == last {
                &self.outs[now].view
            } else {
                &self.chain[i % 2].view
            };
            let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("jodeldiplom look"),
                layout: &self.layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: &self.uniforms,
                            offset: i as u64 * SLOT,
                            size: wgpu::BufferSize::new(std::mem::size_of::<Ctx>() as u64),
                        }),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(input),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(prev),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::Sampler(&self.sampler),
                    },
                ],
            });
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("jodeldiplom look"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: into,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.pipelines[k].1);
            pass.set_bind_group(0, &bind, &[]);
            pass.draw(0..3, 0..1);
        }
        queue.submit([encoder.finish()]);
        self.now = Some(now);
        self.generation += 1;
    }
}

/// The Shader tab's picture's size on the processor, for its guards'
/// measures.
const MEASURED: (usize, usize) = (48, 27);
/// How long the flashes are counted over, seconds.
const COUNTED: f64 = 3.0;

/// What the guards measure on the Shader tab's picture, frame by frame:
/// the look drawn on the processor over the cover made small
/// ([`jodel_looks::render`]), with the frame's own numbers; the light of
/// each frame of the last few seconds (the flashes counted in it), the
/// lyric band's contrast, the stack's cost.
#[derive(Default)]
pub struct Readout {
    small: Option<(u64, jodel_looks::render::Image)>,
    lights: std::collections::VecDeque<(std::time::Instant, f32)>,
    shown: Option<crate::core::shader_tab::Readout>,
}

impl Readout {
    /// The measures after the frame `frame` of `app` drawn at `now`: new
    /// ones where they changed.
    pub fn step(
        &mut self,
        app: &crate::app::App,
        frame: Option<&super::BackdropFrame>,
        now: std::time::Instant,
    ) -> Option<crate::core::shader_tab::Readout> {
        use jodel_looks::{guard, render::Image};
        let checked = crate::core::shader_tab::look(&app.state).check();
        let art = app.art();
        let id = art.map_or(0, |a| a.id);
        if self.small.as_ref().is_none_or(|(i, _)| *i != id) {
            let img = art.map_or_else(
                || Image::flat(MEASURED.0, MEASURED.1, jodel_looks::math::s3(0.3)),
                |a| {
                    let (w, h) = a.image.dimensions();
                    Image::from_rgb(w as usize, h as usize, a.image.as_raw())
                        .resized(MEASURED.0, MEASURED.1)
                },
            );
            self.small = Some((id, img));
        }
        let small = &self.small.as_ref()?.1;
        let (w, h) = (MEASURED.0 as f32, MEASURED.1 as f32);
        let mut contrast = guard::band_contrast(&small.px, small.w, (0.68, 0.86));
        if let Some(look) = frame.and_then(|f| f.look.as_ref()) {
            let passes: Vec<(&'static Pass, Ctx)> = (look.passes.iter())
                .map(|&(p, c)| {
                    let mut c = c;
                    c.size = [w, h, 1.0 / w, 1.0 / h];
                    (p, c)
                })
                .collect();
            let out = jodel_looks::render::draw(&passes, small, None);
            self.lights.push_back((now, guard::light(&out.px)));
            if let Some((_, c)) = passes.first() {
                contrast =
                    guard::band_contrast(&out.px, out.w, (c.band[0] + 0.01, c.band[1] - 0.01));
            }
        }
        while (self.lights.front())
            .is_some_and(|(t, _)| now.saturating_duration_since(*t).as_secs_f64() > COUNTED)
        {
            self.lights.pop_front();
        }
        let span = (self.lights.front()).map_or(0.0, |(t, _)| {
            now.saturating_duration_since(*t).as_secs_f64()
        });
        let lights: Vec<f32> = self.lights.iter().map(|&(_, l)| l).collect();
        let fps = if span > 0.1 {
            (lights.len() as f64 / span) as f32
        } else {
            60.0
        };
        let r = crate::core::shader_tab::Readout {
            flashes: guard::flashes(&lights, fps),
            contrast: (contrast * 10.0).round() / 10.0,
            cost: checked.cost(),
            dropped: checked.dropped,
        };
        (self.shown.as_ref() != Some(&r)).then(|| {
            self.shown = Some(r.clone());
            r
        })
    }
}

/// The cover's colours a look takes where there is no cover: warm,
/// cool, dark.
pub const PALETTE: [[f32; 3]; 3] = [[1.0, 0.85, 0.65], [0.45, 0.55, 0.8], [0.12, 0.12, 0.14]];

/// The cover's colours as a look takes them (encoded, 0 to 1): its most
/// vivid, its next most vivid, and its largest; [`PALETTE`]'s for each it
/// does not have.
pub fn palette_of(swatches: &[crate::work::cover::Swatch]) -> [[f32; 3]; 3] {
    let vivid = |s: &&crate::work::cover::Swatch| {
        let c = s.rgb.map(f32::from);
        let (max, min) = (c[0].max(c[1]).max(c[2]), c[0].min(c[1]).min(c[2]));
        (max - min) * (0.25 + s.share).sqrt()
    };
    let mut by: Vec<&crate::work::cover::Swatch> = swatches.iter().collect();
    by.sort_by(|a, b| vivid(b).total_cmp(&vivid(a)));
    let largest = swatches.iter().max_by(|a, b| a.share.total_cmp(&b.share));
    let of = |s: Option<&crate::work::cover::Swatch>, or: [f32; 3]| {
        s.map_or(or, |s| s.rgb.map(|v| f32::from(v) / 255.0))
    };
    [
        of(by.first().copied(), PALETTE[0]),
        of(by.get(1).copied(), PALETTE[1]),
        of(largest, PALETTE[2]),
    ]
}

/// What the look is given for a frame: the look made safe to draw, the
/// song's time as heard, the drivers then (the song's curves read there,
/// the singing's set live over them), the curves themselves (for the
/// passes that draw the music), the seed (the look's and the song's) and
/// the lyric band (top and bottom, shares of the height).
#[derive(Clone, Debug)]
pub struct LookCue {
    /// The look.
    pub look: Arc<jodel_looks::Checked>,
    /// The time.
    pub t: f64,
    /// The drivers.
    pub signals: Signals,
    /// The song's curves (none: no song analysed).
    pub curves: Option<Arc<jodel_looks::Curves>>,
    /// The seed.
    pub seed: u32,
    /// The lyric band.
    pub band: Option<(f32, f32)>,
}

/// What the look keeps between frames: the look last made safe (and what
/// from), the song's curves (and what from: the song heard and its sung
/// lines), and the look and time of the last frame (a frame far from it,
/// or another look, forgets the frame before).
#[derive(Default)]
pub struct Looks {
    checked: Option<(jodel_looks::Look, Arc<jodel_looks::Checked>)>,
    curves: Option<(CurvesOf, Arc<jodel_looks::Curves>)>,
    last: Option<(String, f64)>,
}

/// What a song's curves are made from: the song heard (by its place) and
/// its sung lines.
type CurvesOf = (usize, Vec<(f64, f64)>);

/// How far a frame's time may be from the last one's and its step and
/// still be the next (further: a seek, a jump).
const NEXT: f64 = 0.25;

/// What only the singing knows, live, on its own clock (the song's time
/// less how late the voice comes: [`crate::core::lag`]) at `heard`: the
/// pitch sung (none for a tenth of a second: none), how many of the last
/// two seconds' notes hit, the game's tier and its streak's rung.
pub fn live(state: &crate::core::State, heard: f64, signals: &mut Signals) {
    let singing = &state.karaoke.singing;
    let sung = state.karaoke.lag.sung_clock(heard);
    let pitch = (singing.heard.iter().rev())
        .find(|h| h.t <= sung)
        .filter(|h| sung - h.t <= 0.1)
        .map_or(0.0, |h| {
            ((h.cents / 100.0 - 36.0) / 48.0).clamp(0.0, 1.0) as f32
        });
    signals.set(Driver::Pitch, pitch);
    let recent: Vec<bool> = (singing.heard.iter().rev())
        .skip_while(|h| h.t > sung)
        .take_while(|h| sung - h.t <= 2.0)
        .filter(|h| h.off.is_some())
        .map(|h| h.hit)
        .collect();
    let hits = recent.iter().filter(|&&h| h).count();
    signals.set(Driver::Accuracy, hits as f32 / recent.len().max(1) as f32);
    let game = state.config.karaoke_game;
    signals.set(Driver::Tier, crate::gui::streak::level(singing, game));
    let top = crate::tiers::Tier::TOP.spec().streak.max(1);
    signals.set(
        Driver::Streak,
        (singing.tally.streak as f32 / top as f32).min(1.0),
    );
}

impl Looks {
    /// The look in force for `state` ([`crate::core::State::look`]) at
    /// `heard` seconds into the song `song` heard (none: none analysed),
    /// for a window of `size`: its cue (none: no look).
    pub fn cue(
        &mut self,
        state: &crate::core::State,
        song: Option<&super::music::Song>,
        heard: f64,
        size: (u32, u32),
    ) -> Option<LookCue> {
        let look = state.look()?;
        if self.checked.as_ref().is_none_or(|(l, _)| l != look) {
            self.checked = Some((look.clone(), Arc::new(look.check())));
        }
        let checked = Arc::clone(&self.checked.as_ref()?.1);
        let singing = &state.karaoke.singing;
        let lines: Vec<(f64, f64)> = (singing.notes.lines.iter())
            .filter_map(|l| Some((l.notes.first()?.start, l.notes.last()?.end)))
            .collect();
        let curves = song.map(|s| self.curves_of(s, lines));
        let mut signals = curves.as_ref().map_or(Signals::QUIET, |c| c.at(heard));
        live(state, heard, &mut signals);
        let (w, h) = (size.0.max(1) as f32, size.1.max(1) as f32);
        let band = crate::gui::stage::lines::band((0.0, 0.0, w, h));
        let song_id = singing
            .song()
            .map_or(String::new(), |p| p.to_string_lossy().into_owned());
        Some(LookCue {
            seed: jodel_looks::seed(&checked.id, &song_id),
            look: checked,
            t: heard,
            signals,
            curves,
            band: Some((band.0 / h, band.1 / h)),
        })
    }

    /// The curves of `song` with its sung `lines`, made once for them.
    fn curves_of(
        &mut self,
        song: &super::music::Song,
        lines: Vec<(f64, f64)>,
    ) -> Arc<jodel_looks::Curves> {
        let of: CurvesOf = (std::ptr::from_ref(song) as usize, lines);
        if let Some((was, c)) = &self.curves {
            if *was == of {
                return Arc::clone(c);
            }
        }
        let c = Arc::new(jodel_looks::Curves::new(
            song.analysis(),
            song.waveform(),
            &song.sections,
            &of.1,
        ));
        self.curves = Some((of, Arc::clone(&c)));
        c
    }

    /// The look's passes for a frame of `cue` in a window of `size`, `dt`
    /// seconds after the last, over a cover of `palette`'s colours.
    pub fn frame(
        &mut self,
        cue: &LookCue,
        size: (u32, u32),
        (dt, palette): (f32, [[f32; 3]; 3]),
    ) -> LookFrame {
        let forget = self
            .last
            .as_ref()
            .is_none_or(|(id, t)| *id != cue.look.id || (cue.t - t - f64::from(dt)).abs() > NEXT);
        self.last = Some((cue.look.id.clone(), cue.t));
        let moment = jodel_looks::Moment {
            size,
            t: cue.t,
            seed: cue.seed,
            dt,
            forget,
            band: cue.band,
            palette,
        };
        LookFrame {
            passes: cue.look.frame(&moment, &cue.signals, cue.curves.as_deref()),
        }
    }
}
