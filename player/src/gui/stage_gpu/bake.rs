//! The diploma's paper baked ([`Bake`]): the dearest shape the stage draws
//! (half the screen of procedural paper, its fibres, its aged edges and its
//! border: [`crate::gui::paper::paper_at`]), yet it does not change once it
//! has unrolled. So once a frame's paper has been the same a few frames in
//! a row ([`STILL`]), it is drawn into a texture of its own, and from
//! then on read back from it, pixel for pixel ([`kind::BAKED`]), while it
//! stays that paper (moved by whole pixels too: the sheet shaken); while it
//! changes (unrolling, the window resized) it is drawn as it is. Nothing of
//! it goes with time (the paper's shader has no clock), and what moves on
//! it (the stamps, the signature, the stars, the words) are shapes of their
//! own, drawn over it each frame.
//!
//! It is baked in [`STRIPS`] strips across it, top to bottom, one a frame
//! (the whole sheet at once took the M4 Max 2.25 ms on its frame): while
//! it is baked, the strips already in are read from the bake and the rest
//! drawn as it is, so each of those frames draws the paper about once.
//! Changed, moved or the window resized before the last strip is in, it
//! starts over.
//!
//! The texture is the window's size, made as a stage first shows (entering
//! Karaoke mode), kept while one shows and let go when none does; made
//! again once a new size has held [`STILL`] frames. Making it costs the
//! system a few milliseconds (its memory made resident), so the reveal
//! pays only for drawing the paper into it. What to do with the texture
//! and a frame's paper is decided apart from the GPU ([`Planner`]).

use super::pass::{make, Reads};
use super::{kind, Blend, Group, Instance, Screen, SCREEN_BYTES};

/// What a bake holds: half floats, as the shader gave them (premultiplied,
/// linear where the frame takes linear light), read back pixel for pixel.
const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
/// How many frames in a row a paper (or the window's size) is the same
/// before it is baked (the texture made again): a window being resized
/// changes it every frame, and is not baked each time.
const STILL: u32 = 3;
/// How many strips a paper is baked in, one a frame.
const STRIPS: u32 = 4;

/// What to do with the texture this frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Texture {
    /// Keep it as it is (or none, as there is none).
    Keep,
    /// Make it, of this size (what it held is gone).
    Make((u32, u32)),
    /// Let it go.
    LetGo,
}

/// What to do with a frame's paper.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) enum Paper {
    /// Draw it as it is (or nothing, where there is none).
    Draw,
    /// Bake its `strip` (of [`STRIPS`], top to bottom: [`rows`]), then
    /// read the strips baked from the bake and draw the rest as it is.
    Bake { strip: u32 },
    /// Read it from the bake: the bake's corner at `corner` (whole pixels,
    /// in the window's), `size` of its pixels.
    Read {
        corner: (f32, f32),
        size: (u32, u32),
    },
}

/// The plan for a frame: its texture, then its paper.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct Plan {
    pub(super) texture: Texture,
    pub(super) paper: Paper,
}

/// What decides the plan, frame by frame: the texture's size (none
/// without one), a new window size and how many frames in a row it held,
/// the paper baked and how many of its strips are in, the paper last
/// drawn and how many frames in a row it was the same, and the largest
/// texture the GPU takes (a side, pixels).
#[derive(Clone, Debug)]
pub(super) struct Planner {
    held: Option<(u32, u32)>,
    resized: ((u32, u32), u32),
    baked: Option<(Instance, u32)>,
    last: Option<Instance>,
    same: u32,
    largest: u32,
}

/// How far `now` is `then` moved, if it is moved by whole pixels and no
/// other: the same sheet, unrolled as far, in the same inks.
fn moved(then: &Instance, now: &Instance) -> Option<(f32, f32)> {
    let d = (now.shape[0] - then.shape[0], now.shape[1] - then.shape[1]);
    let whole = (d.0.round(), d.1.round());
    let near = |a: f32, b: f32| (a - b).abs() < 1e-2;
    let by = [whole.0, whole.1, whole.0, whole.1];
    let placed = near(d.0, whole.0)
        && near(d.1, whole.1)
        && (0..4).all(|k| near(now.quad[k] - then.quad[k], by[k]));
    let same = now.shape[2..] == then.shape[2..]
        && (now.color, now.color2, now.params) == (then.color, then.color2, then.params)
        && (now.uv, now.strain) == (then.uv, then.strain);
    (placed && same).then_some(whole)
}

/// The pixel `paper`'s bake starts at, in the window's pixels: its quad's
/// corner, on a whole pixel.
fn corner(paper: &Instance) -> (f32, f32) {
    (paper.quad[0].floor(), paper.quad[1].floor())
}

/// The size of `paper`'s bake: its quad, out to whole pixels.
fn extent(paper: &Instance) -> (u32, u32) {
    let (x, y) = corner(paper);
    (
        ((paper.quad[2].ceil() - x) as u32).max(1),
        ((paper.quad[3].ceil() - y) as u32).max(1),
    )
}

/// The rows of a bake `height` pixels tall its `strip` covers (of
/// [`STRIPS`], top to bottom): from, to.
fn rows(height: u32, strip: u32) -> (u32, u32) {
    let at = |k: u32| (u64::from(height) * u64::from(k) / u64::from(STRIPS)) as u32;
    (at(strip), at(strip + 1))
}

/// What draws `paper` as `plan` says, where it is read from its bake: the
/// shape that reads it ([`kind::BAKED`]: the bake's pixels, where its
/// corner is now; while it is baked, only the strips in), and the rest of
/// the paper drawn as it is, the rows below them (none once all are in).
fn read(plan: Paper, paper: &Instance) -> Option<(Instance, Option<Instance>)> {
    let ((x, y), (w, h)) = match plan {
        Paper::Draw => return None,
        Paper::Read { corner, size } => (corner, size),
        Paper::Bake { strip } => {
            let (w, h) = extent(paper);
            (corner(paper), (w, rows(h, strip).1))
        }
    };
    let (w, h) = (w as f32, h as f32);
    let shape = Instance {
        quad: [x, y, x + w, y + h],
        shape: [x, y, 0.0, 0.0],
        params: [kind::BAKED, 0.0, 0.0, 0.0],
        ..Instance::default()
    };
    let [left, top, right, bottom] = paper.quad;
    let rest = (matches!(plan, Paper::Bake { .. }) && y + h < bottom).then(|| Instance {
        quad: [left, top.max(y + h), right, bottom],
        ..*paper
    });
    Some((shape, rest))
}

impl Planner {
    /// No texture yet, textures up to `largest` pixels a side.
    pub(super) fn new(largest: u32) -> Planner {
        Planner {
            held: None,
            resized: ((0, 0), 0),
            baked: None,
            last: None,
            same: 0,
            largest,
        }
    }

    /// The plan for a frame of a stage in a window of `size`, with `paper`
    /// (or none).
    pub(super) fn frame(&mut self, paper: Option<&Instance>, size: (u32, u32)) -> Plan {
        let texture = self.texture(size);
        Plan {
            texture,
            paper: paper.map_or(Paper::Draw, |paper| self.paper(paper)),
        }
    }

    /// The plan for a frame without a stage: the texture let go.
    pub(super) fn no_stage(&mut self) -> Plan {
        let texture = match self.held {
            Some(_) => Texture::LetGo,
            None => Texture::Keep,
        };
        *self = Planner::new(self.largest);
        Plan {
            texture,
            paper: Paper::Draw,
        }
    }

    /// The texture for a window of `size`: made where there is none (a
    /// stage showing first), made again once another size has held. A
    /// window of another size than the texture's: a bake under way starts
    /// over, its paper to hold still again.
    fn texture(&mut self, size: (u32, u32)) -> Texture {
        if self.held != Some(size) && self.baked.is_some_and(|(_, strips)| strips < STRIPS) {
            (self.baked, self.same) = (None, 0);
        }
        let fits = size.0.max(size.1) <= self.largest && size.0.min(size.1) > 0;
        if self.held == Some(size) || !fits {
            self.resized = (size, 0);
            return Texture::Keep;
        }
        let held = self.resized.0 == size;
        self.resized = (size, if held { self.resized.1 + 1 } else { 1 });
        if self.held.is_some() && self.resized.1 < STILL {
            return Texture::Keep;
        }
        (self.held, self.baked) = (Some(size), None);
        Texture::Make(size)
    }

    /// What to do with `paper`: read where it is the paper baked, all its
    /// strips in (moved by whole pixels or not); its next strip baked
    /// where it is the paper being baked, where it is; baked from its
    /// first strip where it has held still and fits the texture, else
    /// drawn. Any other paper while one is baked: that bake starts over.
    fn paper(&mut self, paper: &Instance) -> Paper {
        if let Some((baked, strips)) = self.baked {
            match moved(&baked, paper) {
                Some(d) if strips == STRIPS => {
                    let (x, y) = corner(&baked);
                    return Paper::Read {
                        corner: (x + d.0, y + d.1),
                        size: extent(&baked),
                    };
                }
                Some((0.0, 0.0)) => {
                    self.baked = Some((baked, strips + 1));
                    return Paper::Bake { strip: strips };
                }
                _ if strips < STRIPS => self.baked = None,
                _ => {}
            }
        }
        let held = self
            .last
            .is_some_and(|last| moved(&last, paper) == Some((0.0, 0.0)));
        self.same = if held { self.same + 1 } else { 1 };
        self.last = Some(*paper);
        let (w, h) = extent(paper);
        let fits = self.held.is_some_and(|(tw, th)| w <= tw && h <= th);
        if self.same >= STILL && fits {
            self.baked = Some((*paper, 1));
            return Paper::Bake { strip: 0 };
        }
        Paper::Draw
    }
}

/// The paper read from its bake: its place among the frame's shapes, the
/// shape that reads it and the rest of it drawn as it is ([`read`]).
pub(super) struct Baked {
    pub(super) at: usize,
    pub(super) shape: Instance,
    pub(super) rest: Option<Instance>,
}

/// The paper's bake: what decides, the pipeline that draws it (the
/// paper's, into [`FORMAT`]) with numbers and an instance of its own, the
/// texture (none while no stage shows) and its generation (a new one for
/// each texture made or let go: what the stage's pass reads changed).
pub(super) struct Bake {
    planner: Planner,
    pipeline: wgpu::RenderPipeline,
    uniforms: wgpu::Buffer,
    instance: wgpu::Buffer,
    texture: Option<(wgpu::Texture, wgpu::TextureView)>,
    generation: u64,
}

impl Bake {
    /// Its pipeline made now (in the pass's `layout`), not the first time
    /// a diploma shows.
    pub(super) fn new(device: &wgpu::Device, layout: &wgpu::PipelineLayout) -> Bake {
        let buffer = |size, usage| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("jodeldiplom paper"),
                size,
                usage: usage | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let one = std::mem::size_of::<Instance>() as u64;
        Bake {
            planner: Planner::new(device.limits().max_texture_dimension_2d),
            pipeline: make(device, layout, (FORMAT, Blend::Over), Group::Paper),
            uniforms: buffer(SCREEN_BYTES, wgpu::BufferUsages::UNIFORM),
            instance: buffer(one, wgpu::BufferUsages::VERTEX),
            texture: None,
            generation: 0,
        }
    }

    /// The bake the stage's pass reads, if there is one.
    pub(super) fn view(&self) -> Option<&wgpu::TextureView> {
        self.texture.as_ref().map(|(_, view)| view)
    }

    /// Which texture it is ([`Bake`]'s generation).
    pub(super) fn generation(&self) -> u64 {
        self.generation
    }

    /// No stage shows: the texture let go.
    pub(super) fn no_stage(&mut self, gpu: (&wgpu::Device, &wgpu::Queue)) {
        let plan = self.planner.no_stage();
        self.texture_as(gpu, plan.texture);
    }

    /// A frame's `instances` in a window of `size` (`screen` its numbers,
    /// the rest of what is read `reads`): the texture made or kept as
    /// planned ([`Planner::frame`]), and its paper read from its bake
    /// where it is planned so, a strip of it baked first where one is to
    /// be (in a submission of its own); none where it is drawn as it is.
    pub(super) fn frame(
        &mut self,
        gpu: (&wgpu::Device, &wgpu::Queue),
        (instances, screen, size): (&[Instance], &Screen, (u32, u32)),
        reads: &Reads,
    ) -> Option<Baked> {
        let at = instances.iter().position(|i| i.params[0] == kind::PAPER);
        let paper = at.map(|at| &instances[at]);
        let plan = self.planner.frame(paper, size);
        self.texture_as(gpu, plan.texture);
        let (at, paper) = (at?, paper?);
        if let Paper::Bake { strip } = plan.paper {
            self.draw(gpu, paper, screen, reads, strip);
        }
        let (shape, rest) = read(plan.paper, paper)?;
        Some(Baked { at, shape, rest })
    }

    /// The texture made or let go as `step` says: made, cleared at once
    /// (in a submission of its own), so its memory is the GPU's from this
    /// frame on, not the first frame that reads it.
    fn texture_as(&mut self, (device, queue): (&wgpu::Device, &wgpu::Queue), step: Texture) {
        match step {
            Texture::Keep => return,
            Texture::LetGo => self.texture = None,
            Texture::Make((width, height)) => {
                let texture = device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("jodeldiplom paper"),
                    size: wgpu::Extent3d {
                        width,
                        height,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: FORMAT,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING
                        | wgpu::TextureUsages::RENDER_ATTACHMENT,
                    view_formats: &[],
                });
                let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
                let mut encoder = device.create_command_encoder(&Default::default());
                pass_into(&mut encoder, &view, CLEAR);
                queue.submit([encoder.finish()]);
                self.texture = Some((texture, view));
            }
        }
        self.generation += 1;
    }

    /// `paper`'s `strip` drawn into the bake (its quad's pixels from the
    /// texture's corner, the rest cleared with the first strip, kept with
    /// the others), at the frame's `screen` numbers.
    fn draw(
        &mut self,
        (device, queue): (&wgpu::Device, &wgpu::Queue),
        paper: &Instance,
        screen: &Screen,
        reads: &Reads,
        strip: u32,
    ) {
        let Some((texture, view)) = &self.texture else {
            return;
        };
        // The frame's numbers, but the texture's size, its corner where the
        // bake's is, and not shaken: the paper read from it is, as it is
        // drawn.
        let (x, y) = corner(paper);
        let numbers = Screen {
            size: [texture.width() as f32, texture.height() as f32],
            origin: [x, y],
            shake: [0.0; 2],
            ..*screen
        };
        queue.write_buffer(&self.uniforms, 0, bytemuck::bytes_of(&numbers));
        queue.write_buffer(&self.instance, 0, bytemuck::bytes_of(paper));
        // The bake itself not read while it is drawn: the atlas stands for
        // it.
        let bind = reads.bind(device, &self.uniforms, None);
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("jodeldiplom paper"),
        });
        let (w, h) = extent(paper);
        let (from, to) = rows(h, strip);
        {
            let load = if strip == 0 {
                CLEAR
            } else {
                wgpu::LoadOp::Load
            };
            let mut pass = pass_into(&mut encoder, view, load);
            if to > from {
                pass.set_scissor_rect(0, from, w, to - from);
                pass.set_pipeline(&self.pipeline);
                pass.set_bind_group(0, &bind, &[]);
                pass.set_vertex_buffer(0, self.instance.slice(..));
                pass.draw(0..4, 0..1);
            }
        }
        queue.submit([encoder.finish()]);
    }
}

/// What a pass clearing its texture loads: nothing.
const CLEAR: wgpu::LoadOp<wgpu::Color> = wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT);

/// A pass into `view`, loading it as `load` says.
fn pass_into<'e>(
    encoder: &'e mut wgpu::CommandEncoder,
    view: &wgpu::TextureView,
    load: wgpu::LoadOp<wgpu::Color>,
) -> wgpu::RenderPass<'e> {
    encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("jodeldiplom paper"),
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

#[cfg(test)]
mod tests {
    use super::*;

    /// A sheet at `(x, y)` (its quad's corner), `w` wide, unrolled to
    /// `front`.
    fn sheet((x, y): (f32, f32), w: f32, front: f32) -> Instance {
        Instance {
            quad: [x, y, x + w, y + 1220.0],
            shape: [x + w / 2.0, y + 610.0, w / 2.0 - 40.0, 570.0],
            color: [0.99, 0.96, 0.87, 1.0],
            params: [kind::PAPER, front, 12.0, 0.0],
            ..Instance::default()
        }
    }

    const AT: (f32, f32) = (100.25, 80.5);
    const WINDOW: (u32, u32) = (3024, 1964);

    /// The paper part of `p`'s plan for `paper` in [`WINDOW`], the texture
    /// kept.
    fn paper_of(p: &mut Planner, paper: Option<&Instance>) -> Paper {
        let plan = p.frame(paper, WINDOW);
        assert_eq!(plan.texture, Texture::Keep);
        plan.paper
    }

    /// The papers of `p`'s plans for `n` frames of `paper` in [`WINDOW`].
    fn papers_of(p: &mut Planner, paper: &Instance, n: u32) -> Vec<Paper> {
        (0..n).map(|_| paper_of(p, Some(paper))).collect()
    }

    /// The plans of a paper held still from its first frame on: drawn
    /// until it has held [`STILL`] frames, then baked strip by strip.
    fn baking() -> Vec<Paper> {
        let drawn = (1..STILL).map(|_| Paper::Draw);
        drawn
            .chain((0..STRIPS).map(|strip| Paper::Bake { strip }))
            .collect()
    }

    /// A planner with its texture made as the stage first shows.
    fn ready() -> Planner {
        let mut p = Planner::new(16384);
        let first = p.frame(None, WINDOW);
        assert_eq!(first.texture, Texture::Make(WINDOW));
        p
    }

    /// A planner with `paper` baked: its texture made as the stage first
    /// shows, the sheet unrolled, then held still.
    fn baked(paper: &Instance) -> Planner {
        let mut p = ready();
        for front in [-500.0, 0.0, 400.0] {
            let unrolling = sheet(AT, 1800.0, front);
            assert_eq!(paper_of(&mut p, Some(&unrolling)), Paper::Draw);
        }
        let n = STILL - 1 + STRIPS;
        assert_eq!(papers_of(&mut p, paper, n), baking());
        p
    }

    /// Unrolling it is drawn; held still for [`STILL`] frames, baked a
    /// strip a frame, top to bottom; read only once all are in, where its
    /// corner is, a whole pixel.
    #[test]
    fn a_paper_held_still_is_baked_strip_by_strip_then_read() {
        let open = sheet(AT, 1800.0, 610.0);
        let mut p = baked(&open);
        let read = Paper::Read {
            corner: (100.0, 80.0),
            size: (1801, 1221),
        };
        assert_eq!(paper_of(&mut p, Some(&open)), read);
        assert_eq!(paper_of(&mut p, None), Paper::Draw);
        assert_eq!(paper_of(&mut p, Some(&open)), read);
    }

    /// The texture is made as a stage first shows, kept while one shows,
    /// let go when none does, made again when one shows again.
    #[test]
    fn the_texture_lives_while_a_stage_shows() {
        let open = sheet(AT, 1800.0, 610.0);
        let mut p = baked(&open);
        for _ in 0..100 {
            assert_eq!(paper_of(&mut p, None), Paper::Draw);
        }
        assert!(matches!(paper_of(&mut p, Some(&open)), Paper::Read { .. }));
        assert_eq!(p.no_stage().texture, Texture::LetGo);
        assert_eq!(p.no_stage().texture, Texture::Keep);
        let back = p.frame(Some(&open), WINDOW);
        assert_eq!(back.texture, Texture::Make(WINDOW));
        assert_eq!(back.paper, Paper::Draw);
    }

    /// Resized frame by frame the texture is kept and the paper drawn;
    /// once the new size has held [`STILL`] frames, the texture is made
    /// again and the paper (held as long) baked into it, then read.
    #[test]
    fn a_resized_window_has_its_texture_made_again_once_still() {
        let mut p = baked(&sheet(AT, 1800.0, 610.0));
        for (w, sw) in [(3000, 1790.0), (2980, 1780.0), (2960, 1770.0)] {
            let plan = p.frame(Some(&sheet(AT, sw, 610.0)), (w, 1964));
            assert_eq!((plan.texture, plan.paper), (Texture::Keep, Paper::Draw));
        }
        let held = sheet(AT, 1770.0, 610.0);
        let n = 2 + STRIPS as usize;
        let plans: Vec<Plan> = (0..n).map(|_| p.frame(Some(&held), (2960, 1964))).collect();
        assert_eq!(
            (plans[0].texture, plans[0].paper),
            (Texture::Keep, Paper::Draw)
        );
        assert_eq!(plans[1].texture, Texture::Make((2960, 1964)));
        for (strip, plan) in (0..).zip(&plans[1..n - 1]) {
            assert_eq!(plan.paper, Paper::Bake { strip });
        }
        assert_eq!(plans[n - 1].texture, Texture::Keep);
        assert!(matches!(plans[n - 1].paper, Paper::Read { .. }));
        // Too large for the GPU: no texture, never baked.
        let mut small = Planner::new(1024);
        for _ in 0..10 {
            let plan = small.frame(Some(&held), WINDOW);
            assert_eq!((plan.texture, plan.paper), (Texture::Keep, Paper::Draw));
        }
    }

    /// Shaken by whole pixels it is read moved; by part of a pixel it is
    /// another paper, drawn, and baked again once it holds there.
    #[test]
    fn a_shaken_paper_is_read_moved_by_whole_pixels() {
        let open = sheet(AT, 1800.0, 610.0);
        let mut p = baked(&open);
        let shaken = sheet((AT.0 + 3.0, AT.1 - 2.0), 1800.0, 610.0);
        let read = Paper::Read {
            corner: (103.0, 78.0),
            size: (1801, 1221),
        };
        assert_eq!(paper_of(&mut p, Some(&shaken)), read);
        let part = sheet((AT.0 + 0.5, AT.1), 1800.0, 610.0);
        assert_eq!(papers_of(&mut p, &part, STILL), baking()[..STILL as usize]);
    }

    /// A new diploma of the same size: unrolling drawn; open, the same
    /// paper as the bake is read at once; in other inks, baked again.
    #[test]
    fn a_new_diploma_of_the_same_size() {
        let open = sheet(AT, 1800.0, 610.0);
        let mut p = baked(&open);
        for front in [-600.0, -100.0, 300.0] {
            let unrolling = sheet(AT, 1800.0, front);
            assert_eq!(paper_of(&mut p, Some(&unrolling)), Paper::Draw);
        }
        assert!(matches!(paper_of(&mut p, Some(&open)), Paper::Read { .. }));
        let inked = Instance {
            color: [0.9, 0.96, 0.87, 1.0],
            ..open
        };
        assert_eq!(papers_of(&mut p, &inked, STILL), baking()[..STILL as usize]);
    }

    /// Another paper, the same moved (by whole pixels even) or the window
    /// resized before its last strip is in: the bake starts over, from
    /// its first strip once the paper has held still again; never read
    /// before all its strips are in.
    #[test]
    fn a_paper_changed_moved_or_resized_mid_bake_starts_over() {
        let open = sheet(AT, 1800.0, 610.0);
        let inked = Instance {
            color: [0.9, 0.96, 0.87, 1.0],
            ..open
        };
        let shaken = sheet((AT.0 + 3.0, AT.1 - 2.0), 1800.0, 610.0);
        let whole = STILL - 1 + STRIPS;
        for other in [inked, shaken] {
            let mut p = ready();
            let n = STILL + 1;
            assert_eq!(papers_of(&mut p, &open, n), baking()[..n as usize]);
            assert_eq!(papers_of(&mut p, &other, whole), baking());
            assert!(matches!(paper_of(&mut p, Some(&other)), Paper::Read { .. }));
        }
        // Resized, the paper kept: the bake under way starts over, the
        // resized frame the first of the paper held still again.
        let mut p = ready();
        assert_eq!(papers_of(&mut p, &open, STILL), baking()[..STILL as usize]);
        let resized = p.frame(Some(&open), (3000, 1964));
        assert_eq!(
            (resized.texture, resized.paper),
            (Texture::Keep, baking()[0])
        );
        assert_eq!(papers_of(&mut p, &open, whole - 1), baking()[1..]);
        assert!(matches!(paper_of(&mut p, Some(&open)), Paper::Read { .. }));
    }

    /// The strips cover the bake's rows, top to bottom, each about a
    /// quarter; while it is baked, the strips in are read and the rows
    /// below them drawn as they are, none of a row twice, all once.
    #[test]
    fn the_strips_cover_the_paper_once() {
        for h in [1, 3, 4, 7, 1221, 1964] {
            let all: Vec<(u32, u32)> = (0..STRIPS).map(|k| rows(h, k)).collect();
            assert_eq!(all[0].0, 0);
            assert_eq!(all[STRIPS as usize - 1].1, h);
            assert!(all.windows(2).all(|w| w[0].1 == w[1].0));
            assert!(all.iter().all(|r| (r.1 - r.0).abs_diff(h / STRIPS) <= 1));
        }
        let open = sheet(AT, 1800.0, 610.0);
        assert_eq!(read(Paper::Draw, &open), None);
        for strip in 0..STRIPS {
            let (shape, rest) = read(Paper::Bake { strip }, &open).unwrap();
            assert_eq!(shape.params[0], kind::BAKED);
            assert_eq!(shape.shape[..2], [100.0, 80.0]);
            let done = 80.0 + rows(1221, strip).1 as f32;
            assert_eq!(shape.quad, [100.0, 80.0, 1901.0, done]);
            match rest {
                Some(rest) => {
                    assert_eq!(rest.quad, [AT.0, done, open.quad[2], open.quad[3]]);
                    assert_eq!(rest.shape, open.shape);
                    assert_eq!(rest.params, open.params);
                }
                None => assert_eq!(strip, STRIPS - 1),
            }
        }
        let read_all = Paper::Read {
            corner: (103.0, 78.0),
            size: (1801, 1221),
        };
        let (shape, rest) = read(read_all, &open).unwrap();
        assert_eq!(shape.quad, [103.0, 78.0, 1904.0, 1299.0]);
        assert!(rest.is_none());
    }

    /// A paper larger than the window's texture is never baked.
    #[test]
    fn a_paper_larger_than_the_window_is_drawn() {
        let mut p = Planner::new(16384);
        let wide = sheet(AT, 1800.0, 610.0);
        for _ in 0..5 {
            assert_eq!(p.frame(Some(&wide), (1600, 1300)).paper, Paper::Draw);
        }
    }

    /// Moved by whole pixels it is the same paper; by part of one, grown,
    /// unrolled further or in another ink, another.
    #[test]
    fn a_paper_moved_by_whole_pixels_is_the_same_paper() {
        let open = sheet(AT, 1800.0, 610.0);
        assert_eq!(moved(&open, &open), Some((0.0, 0.0)));
        let shaken = sheet((AT.0 + 4.0, AT.1 - 1.0), 1800.0, 610.0);
        assert_eq!(moved(&open, &shaken), Some((4.0, -1.0)));
        assert_eq!(
            moved(&open, &sheet((AT.0 + 0.3, AT.1), 1800.0, 610.0)),
            None
        );
        assert_eq!(moved(&open, &sheet(AT, 1801.0, 610.0)), None);
        assert_eq!(moved(&open, &sheet(AT, 1800.0, 500.0)), None);
        assert_eq!(corner(&open), (100.0, 80.0));
        assert_eq!(extent(&open), (1801, 1221));
    }
}
