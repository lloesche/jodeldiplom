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

use jodel_looks::{Ctx, Curves, Pass, Signals};

/// A look's frame: the cue it is drawn from, and its passes, each with
/// its numbers ([`jodel_looks::Checked::frame`]).
#[derive(Clone, Debug)]
pub struct LookFrame {
    /// The cue.
    pub cue: LookCue,
    /// The passes, in order.
    pub passes: Vec<(&'static Pass, Ctx)>,
}

/// The textures' format.
const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
/// A pass's numbers' place in the uniform buffer (its slots aligned for
/// any GPU).
const SLOT: u64 = 512;
const _: () = assert!(
    std::mem::size_of::<Ctx>() as u64 <= SLOT,
    "a pass's numbers fit its slot"
);

/// A texture the window's size and its view.
struct Target {
    _texture: wgpu::Texture,
    view: wgpu::TextureView,
}

/// What a look draws into while one draws: textures of the window's
/// `size`, two between passes (in turn) and two outputs, `now` this
/// frame's (the other the frame before).
struct Targets {
    size: (u32, u32),
    chain: [Target; 2],
    outs: [Target; 2],
    now: usize,
}

/// The passes' pipelines, their numbers, and what they draw into (none:
/// no look).
pub struct LookPass {
    linear: bool,
    layout: wgpu::BindGroupLayout,
    pipeline_layout: wgpu::PipelineLayout,
    sampler: wgpu::Sampler,
    pipelines: Vec<(&'static str, wgpu::RenderPipeline)>,
    uniforms: wgpu::Buffer,
    targets: Option<Targets>,
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
            targets: None,
            generation: 0,
        }
    }

    /// What the ground reads instead of the backdrop: this frame's
    /// output, while a look draws.
    pub fn out(&self) -> Option<&wgpu::TextureView> {
        (self.targets.as_ref()).map(|t| &t.outs[t.now].view)
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
            if self.targets.take().is_some() {
                self.generation += 1;
            }
            return;
        };
        let size = (size.0.max(1), size.1.max(1));
        let mut targets = match self.targets.take() {
            Some(t) if t.size == size => Targets {
                now: 1 - t.now,
                ..t
            },
            _ => Targets {
                size,
                chain: [(); 2].map(|_| Self::target(device, size)),
                outs: [(); 2].map(|_| Self::target(device, size)),
                now: 0,
            },
        };
        let now = targets.now;
        let pipelines: Vec<usize> = (frame.passes.iter())
            .map(|(p, _)| self.pipeline(device, p))
            .collect();
        for (i, (_, ctx)) in frame.passes.iter().enumerate() {
            queue.write_buffer(&self.uniforms, i as u64 * SLOT, bytemuck::bytes_of(ctx));
        }
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("jodeldiplom look"),
        });
        let (chain, outs) = (&targets.chain, &targets.outs);
        let prev = &outs[1 - now].view;
        let last = frame.passes.len() - 1;
        for (i, &k) in pipelines.iter().enumerate() {
            let input = if i == 0 {
                under
            } else {
                &chain[(i - 1) % 2].view
            };
            let into = if i == last {
                &outs[now].view
            } else {
                &chain[i % 2].view
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
        targets.now = now;
        self.targets = Some(targets);
        self.generation += 1;
    }
}

/// The size of the song's picture made small ([`Looks::picture`]): what
/// a look's strength is found over and the Shader tab's guards measure
/// on (what washes out or blacks out does so at any size).
const MEASURED: (usize, usize) = (48, 27);

/// The cover's colours as a look takes them (encoded, 0 to 1): its most
/// vivid, its next most vivid, and its largest; [`jodel_looks::PALETTE`]'s
/// for each it does not have.
pub fn palette_of(swatches: &[crate::work::picture::Swatch]) -> [[f32; 3]; 3] {
    let vivid = |s: &&crate::work::picture::Swatch| {
        let c = s.rgb.map(f32::from);
        let (max, min) = (c[0].max(c[1]).max(c[2]), c[0].min(c[1]).min(c[2]));
        (max - min) * (0.25 + s.share).sqrt()
    };
    let mut by: Vec<&crate::work::picture::Swatch> = swatches.iter().collect();
    by.sort_by(|a, b| vivid(b).total_cmp(&vivid(a)));
    let largest = swatches.iter().max_by(|a, b| a.share.total_cmp(&b.share));
    let of = |s: Option<&crate::work::picture::Swatch>, or: [f32; 3]| {
        s.map_or(or, |s| s.rgb.map(|v| f32::from(v) / 255.0))
    };
    [
        of(by.first().copied(), jodel_looks::PALETTE[0]),
        of(by.get(1).copied(), jodel_looks::PALETTE[1]),
        of(largest, jodel_looks::PALETTE[2]),
    ]
}

/// What the look is given for a frame: the look made safe to draw, the
/// strength it draws at ([`jodel_looks::guard::strength`]), the song's
/// time as heard, the drivers then (the song's curves read there, the
/// singing's set live over them), the curves themselves (for the passes
/// that draw the music), the seed (the look's and the song's) and the
/// lyric band (top and bottom, shares of the height).
#[derive(Clone, Debug)]
pub struct LookCue {
    /// The look.
    pub look: Arc<jodel_looks::Checked>,
    /// The strength.
    pub strength: f32,
    /// The time.
    pub t: f64,
    /// The drivers.
    pub signals: Signals,
    /// The song's curves (none: no song analysed).
    pub curves: Option<Arc<Curves>>,
    /// The seed.
    pub seed: u32,
    /// The lyric band.
    pub band: Option<(f32, f32)>,
}

/// What the look keeps between frames: the song's curves (made once for
/// the song heard and its sung lines: [`SongCurves`]), what the look in
/// force is made into for them and the song's picture ([`Made`], for its
/// [`Key`]), how many have been made (each one's stamp), the strength
/// drawn (eased toward the made look's: [`RISE`], [`FALL`]), the finding
/// under way, the song's picture made small (and its id), and the look
/// and time of the last frame (a frame far from it, or another look,
/// forgets the frame before).
#[derive(Default)]
pub struct Looks {
    curves: Option<Arc<SongCurves>>,
    made: Option<Made>,
    makes: u64,
    drawn: f32,
    finding: Finding,
    picture: Option<(u64, Arc<jodel_looks::render::Image>)>,
    last: Option<(String, f64)>,
}

/// How long the strength drawn takes to rise from none to full, seconds
/// (a look found comes in softly).
const RISE: f32 = 0.5;
/// How long it takes to fall from full to none, seconds (a look not yet
/// measured, or found weaker, goes at once: never a frame too bright for
/// long).
const FALL: f32 = 0.1;

/// The strength drawn `drawn`, `dt` seconds on toward `target`: up in
/// [`RISE`] from none to full, down in [`FALL`], never past it.
fn eased(drawn: f32, target: f32, dt: f32) -> f32 {
    let dt = dt.clamp(0.0, 0.1);
    if target > drawn {
        (drawn + dt / RISE).min(target)
    } else {
        (drawn - dt / FALL).max(target)
    }
}

/// What a song's curves are made from: the song heard
/// ([`super::music::Song::id`]) and its sung lines. A value: the same, the
/// same curves.
#[derive(Clone, Debug, PartialEq)]
struct CurvesOf {
    song: u64,
    lines: Vec<(f64, f64)>,
}

/// A song's curves ([`jodel_looks::Curves`]) and what they are made from.
#[derive(Debug)]
struct SongCurves {
    of: CurvesOf,
    curves: Arc<Curves>,
}

/// The song as heard at a frame ([`Looks::hear`]): every driver then, and
/// the song's curves they were read from (none: none analysed), what the
/// look is cued for ([`Looks::cue`]).
#[derive(Clone, Debug)]
pub struct SongHeard {
    /// Every driver.
    pub signals: Signals,
    curves: Option<Arc<SongCurves>>,
}

/// What a look made for a song is made from: the look, the song's curves
/// (none: no song analysed), and its picture's id (0: none). Values all:
/// the same key, the same look, curves and strength.
#[derive(Debug, PartialEq)]
struct Key {
    look: jodel_looks::Look,
    curves: Option<CurvesOf>,
    art: u64,
}

impl Key {
    /// Whether it is the key of `look` over the curves of `curves` and the
    /// picture `art` (its id), without making one.
    fn is(&self, look: &jodel_looks::Look, curves: Option<&CurvesOf>, art: u64) -> bool {
        self.look == *look && self.curves.as_ref() == curves && self.art == art
    }
}

/// The singing's sung lines in `state`: each one's first note's start and
/// last note's end.
fn sung_lines(state: &crate::core::State) -> impl Iterator<Item = (f64, f64)> + '_ {
    (state.karaoke.singing.notes.lines.iter())
        .filter_map(|l| Some((l.notes.first()?.start, l.notes.last()?.end)))
}

/// A look made for a song: what for, its stamp (which finding is its),
/// the look made safe, the song's curves, its strength once it is found,
/// and what it aims at until then: the strength last found for the same
/// look (by its id: an edit of it on the Shader tab too) over the same
/// picture (its curves new: its waveform come, its grid fixed by hand,
/// another song without a cover), else none (the picture as it is).
struct Made {
    key: Key,
    stamp: u64,
    look: Arc<jodel_looks::Checked>,
    curves: Option<Arc<Curves>>,
    strength: Option<f32>,
    interim: Option<f32>,
}

impl Made {
    /// The strength it aims at: its own, else its interim, else none.
    fn aim(&self) -> f32 {
        self.strength.or(self.interim).unwrap_or(0.0)
    }
}

/// A look's strength being found on a thread of its own (the drawing
/// goes on meanwhile), and the next to find once it is (only the newest
/// kept: a look edited on the Shader tab asks for one each frame), each
/// with the stamp of the look it is for.
#[derive(Default)]
struct Finding {
    running: Option<(u64, std::thread::JoinHandle<f32>)>,
    next: Option<(u64, Job)>,
}

/// What a strength is found from: the look, the picture, the curves.
type Job = (
    Arc<jodel_looks::Checked>,
    Arc<jodel_looks::render::Image>,
    Option<Arc<Curves>>,
);

impl Finding {
    /// `job` to do for the look stamped `stamp` once what runs is done (it
    /// in place of any other waiting).
    fn ask(&mut self, stamp: u64, job: Job) {
        self.next = Some((stamp, job));
    }

    /// The strength found since the last call and the stamp of the look it
    /// is for; the next started where nothing runs.
    fn poll(&mut self) -> Option<(u64, f32)> {
        let done = self.running.take_if(|(_, h)| h.is_finished());
        if self.running.is_none() {
            self.running = self.next.take().map(|(stamp, (look, picture, curves))| {
                let h = std::thread::spawn(move || {
                    jodel_looks::guard::strength(&look, &picture, curves.as_deref())
                });
                (stamp, h)
            });
        }
        // A finding that failed draws the look not at all.
        done.map(|(stamp, h)| (stamp, h.join().unwrap_or(0.0)))
    }
}

/// The song's picture made small ([`MEASURED`]), encoded colours (none: a
/// mid grey).
fn small_picture(art: Option<&crate::work::art::Art>) -> jodel_looks::render::Image {
    use jodel_looks::render::Image;
    let (w, h) = MEASURED;
    art.map_or_else(
        || Image::flat(w, h, jodel_looks::math::s3(0.3)),
        |a| {
            let (aw, ah) = a.image.dimensions();
            Image::from_rgb(aw as usize, ah as usize, a.image.as_raw()).resized(w, h)
        },
    )
}

/// How far a frame's time may be from the last one's and its step and
/// still be the next (further: a seek, a jump).
const NEXT: f64 = 0.25;

impl Looks {
    /// The song `song` heard (none: none analysed) at `heard` seconds in,
    /// for `state`: every driver then, the song's curves read there (made
    /// once for the song and its sung lines, kept), the singing's set live
    /// over them ([`crate::core::karaoke::live`]); what the look and the
    /// effects alike move by.
    pub fn hear(
        &mut self,
        state: &crate::core::State,
        song: Option<&super::music::Song>,
        heard: f64,
    ) -> SongHeard {
        self.follow(state, song);
        let mut signals = (self.curves.as_ref()).map_or(Signals::QUIET, |c| c.curves.at(heard));
        crate::core::karaoke::live(state, heard, &mut signals);
        SongHeard {
            signals,
            curves: self.curves.clone(),
        }
    }

    /// The song's curves kept as those of `song` (none: none) with
    /// `state`'s sung lines, made again only where either is new.
    fn follow(&mut self, state: &crate::core::State, song: Option<&super::music::Song>) {
        let Some(song) = song else {
            self.curves = None;
            return;
        };
        let same = |c: &Arc<SongCurves>| {
            c.of.song == song.id() && c.of.lines.iter().copied().eq(sung_lines(state))
        };
        if !self.curves.as_ref().is_some_and(same) {
            let of = CurvesOf {
                song: song.id(),
                lines: sung_lines(state).collect(),
            };
            let curves = Curves::new(song.analysis(), song.waveform(), &song.sections, &of.lines);
            self.curves = Some(Arc::new(SongCurves {
                of,
                curves: Arc::new(curves),
            }));
        }
    }

    /// The look in force for `state` ([`crate::core::State::look`]) at
    /// `heard` seconds into the song as `song` hears it ([`Looks::hear`]:
    /// every driver then, and the curves they are of), over its picture
    /// `art`, for a window of `size`, `dt` seconds after the last
    /// frame: its cue (none: no look). Its strength is found away from
    /// here, once for the look, the song and the picture; until it is, the
    /// look aims at none (the picture as it is) or, where the same look (or
    /// an edit of it) is over the same picture, at the one found for it
    /// there ([`Made::aim`]), and the strength drawn eases toward it.
    pub fn cue(
        &mut self,
        state: &crate::core::State,
        (art, song): (Option<&crate::work::art::Art>, &SongHeard),
        (heard, dt): (f64, f32),
        size: (u32, u32),
    ) -> Option<LookCue> {
        self.found();
        let Some(look) = state.look() else {
            self.made = None;
            self.finding.next = None;
            self.drawn = 0.0;
            return None;
        };
        let of = song.curves.as_ref().map(|c| &c.of);
        let art_id = art.map_or(0, |a| a.id);
        let made = match self.made.take() {
            Some(m) if m.key.is(look, of, art_id) => m,
            was => {
                let key = Key {
                    look: look.clone(),
                    curves: of.cloned(),
                    art: art_id,
                };
                self.make(was, key, (song.curves.as_deref(), art))
            }
        };
        self.drawn = eased(self.drawn, made.aim(), dt);
        let made = self.made.insert(made);
        let (w, h) = (size.0.max(1) as f32, size.1.max(1) as f32);
        let band = crate::gui::stage::lines::band((0.0, 0.0, w, h));
        let song_id = (state.karaoke.singing.song())
            .map_or(String::new(), |p| p.to_string_lossy().into_owned());
        Some(LookCue {
            seed: jodel_looks::seed(&made.look.id, &song_id),
            look: Arc::clone(&made.look),
            strength: self.drawn,
            t: heard,
            signals: song.signals,
            curves: made.curves.clone(),
            band: Some((band.0 / h, band.1 / h)),
        })
    }

    /// The mode and the tab left: a look whose strength is found is kept at
    /// it (coming back finds nothing again), one not measured yet let go,
    /// so nothing changes by itself while away; the frame before
    /// forgotten.
    pub fn left(&mut self) {
        match self.made.as_ref().and_then(|m| m.strength) {
            Some(s) => self.drawn = s,
            None => {
                self.made = None;
                self.drawn = 0.0;
            }
        }
        self.finding.next = None;
        self.last = None;
    }

    /// The strength found since the last frame, kept only for the look it
    /// was found for (one found for a look since gone is let go).
    fn found(&mut self) {
        if let Some((stamp, found)) = self.finding.poll() {
            if let Some(m) = self.made.as_mut().filter(|m| m.stamp == stamp) {
                m.strength = Some(found);
            }
        }
    }

    /// Whether the strength drawn is still to change by itself: the look
    /// in force not measured yet, or the strength drawn still easing to
    /// its own.
    pub fn settling(&self) -> bool {
        self.made
            .as_ref()
            .is_some_and(|m| m.strength != Some(self.drawn))
    }

    /// The look of `key` made over the song's `curves` (none: no song) and
    /// its picture `art`, in place of `was` (none: none): made safe (kept
    /// from `was` where it is the same look), its strength asked for;
    /// `was`'s aimed at meanwhile where it is the same look (by its id)
    /// over the same picture.
    fn make(
        &mut self,
        was: Option<Made>,
        key: Key,
        (curves, art): (Option<&SongCurves>, Option<&crate::work::art::Art>),
    ) -> Made {
        let look = match &was {
            Some(m) if m.key.look == key.look => Arc::clone(&m.look),
            _ => Arc::new(key.look.check()),
        };
        let curves = curves.map(|c| Arc::clone(&c.curves));
        let interim = (was.as_ref())
            .filter(|m| (&m.key.look.id, m.key.art) == (&key.look.id, key.art))
            .and_then(|m| m.strength.or(m.interim));
        self.makes += 1;
        let job = (Arc::clone(&look), self.picture(art), curves.clone());
        self.finding.ask(self.makes, job);
        Made {
            key,
            stamp: self.makes,
            look,
            curves,
            strength: None,
            interim,
        }
    }

    /// The song's picture `art` made small ([`MEASURED`]), made once for
    /// it.
    pub fn picture(
        &mut self,
        art: Option<&crate::work::art::Art>,
    ) -> Arc<jodel_looks::render::Image> {
        let id = art.map_or(0, |a| a.id);
        match &self.picture {
            Some((was, p)) if *was == id => Arc::clone(p),
            _ => {
                let p = Arc::new(small_picture(art));
                self.picture = Some((id, Arc::clone(&p)));
                p
            }
        }
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
            strength: cue.strength,
        };
        LookFrame {
            passes: cue.look.frame(&moment, &cue.signals, cue.curves.as_deref()),
            cue: cue.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A look made for no song, stamped `stamp`, its strength not found.
    fn made(stamp: u64) -> Made {
        let look = jodel_looks::presets::all()[0].clone();
        Made {
            look: Arc::new(look.check()),
            key: Key {
                look,
                curves: None,
                art: 0,
            },
            stamp,
            curves: None,
            strength: None,
            interim: None,
        }
    }

    /// `looks`' strength drawn `dt` seconds on, as a frame eases it.
    fn ease(looks: &mut Looks, dt: f32) {
        let aim = looks.made.as_ref().map_or(0.0, Made::aim);
        looks.drawn = eased(looks.drawn, aim, dt);
    }

    /// The same look (by its id) over the same picture, its curves new or
    /// one of its layers edited: the strength found for it is kept while
    /// its own is found, the drawn strength not dipping; another picture or
    /// another look aims at none again.
    #[test]
    fn new_curves_for_the_same_look_and_picture_keep_its_strength_meanwhile() {
        let mut looks = Looks {
            drawn: 0.6,
            ..Looks::default()
        };
        let found = Made {
            strength: Some(0.6),
            ..made(1)
        };
        let key = |song: Option<u64>, art: u64| Key {
            curves: song.map(|song| CurvesOf {
                song,
                lines: Vec::new(),
            }),
            art,
            ..made(0).key
        };
        let remade = looks.make(Some(found), key(Some(7), 0), (None, None));
        assert_eq!((remade.strength, remade.interim), (None, Some(0.6)));
        looks.made = Some(remade);
        for _ in 0..10 {
            ease(&mut looks, 0.05);
        }
        assert_eq!(looks.drawn, 0.6, "no dip while it is found again");
        assert!(looks.settling(), "its own still to come");
        // A layer's mix edited, the id kept (the Shader tab's knobs).
        let mut edited = key(Some(7), 0);
        edited.look.layers[0].mix = 0.5;
        let was = looks.made.take();
        let edit = looks.make(was, edited, (None, None));
        assert_eq!(edit.interim, Some(0.6), "an edit of the same look");
        let mut other_look = key(Some(7), 0);
        other_look.look.id = "another".into();
        assert_eq!(
            looks.make(Some(edit), other_look, (None, None)).interim,
            None
        );
        let found = Made {
            strength: Some(0.6),
            ..made(2)
        };
        let other = looks.make(Some(found), key(Some(7), 3), (None, None));
        assert_eq!(other.interim, None, "another picture");
        looks.made = Some(other);
        ease(&mut looks, FALL);
        assert_eq!(looks.drawn, 0.0);
    }

    /// Waits for what runs to be done (a minute at most).
    fn done(looks: &Looks) {
        let until = std::time::Instant::now() + std::time::Duration::from_secs(60);
        while (looks.finding.running.as_ref()).is_some_and(|(_, h)| !h.is_finished()) {
            assert!(std::time::Instant::now() < until, "the strength found");
            std::thread::yield_now();
        }
    }

    /// A strength found for a look since gone is let go, never drawn for
    /// the look now in force; that look's own is kept once it comes.
    #[test]
    fn a_strength_is_kept_only_for_the_look_it_was_found_for() {
        let mut looks = Looks::default();
        let job = |m: &Made| (Arc::clone(&m.look), Arc::new(small_picture(None)), None);
        let gone = made(1);
        looks.finding.ask(1, job(&gone));
        looks.found();
        let now = made(2);
        looks.finding.ask(2, job(&now));
        looks.made = Some(now);
        done(&looks);
        looks.found();
        assert_eq!(
            looks.made.as_ref().unwrap().strength,
            None,
            "the gone look's"
        );
        assert!(looks.settling());
        done(&looks);
        looks.found();
        assert!(looks.made.as_ref().unwrap().strength.is_some(), "its own");
    }

    /// The strength drawn: none while the look is not measured, rising to
    /// the one found over [`RISE`], falling over [`FALL`]; still once it is
    /// there.
    #[test]
    fn the_strength_drawn_eases_to_the_one_found() {
        let mut looks = Looks {
            made: Some(made(1)),
            drawn: 1.0,
            ..Looks::default()
        };
        ease(&mut looks, FALL / 2.0);
        assert!((looks.drawn - 0.5).abs() < 1e-4, "{}", looks.drawn);
        ease(&mut looks, FALL);
        assert_eq!(looks.drawn, 0.0, "none while not measured");
        assert!(looks.settling());
        looks.made.as_mut().unwrap().strength = Some(0.8);
        ease(&mut looks, 0.05);
        assert!((looks.drawn - 0.05 / RISE).abs() < 1e-4, "{}", looks.drawn);
        for _ in 0..20 {
            ease(&mut looks, 0.05);
        }
        assert_eq!(looks.drawn, 0.8);
        assert!(!looks.settling());
    }

    /// Leaving the mode or the tab keeps a look whose strength is found
    /// (back again, it is not found again), at that strength and still
    /// (nothing drawn meanwhile); one not measured yet is let go.
    #[test]
    fn a_look_measured_is_kept_across_leaving_and_coming_back() {
        let mut looks = Looks {
            made: Some(Made {
                strength: Some(0.7),
                ..made(1)
            }),
            drawn: 0.3,
            ..Looks::default()
        };
        looks.left();
        let kept = looks.made.as_ref().expect("kept");
        assert_eq!((kept.stamp, kept.strength), (1, Some(0.7)));
        assert_eq!(looks.drawn, 0.7);
        assert!(!looks.settling(), "still while away");
        looks.made = Some(made(2));
        looks.left();
        assert!(
            looks.made.is_none() && !looks.settling(),
            "not measured: let go"
        );
    }
}
