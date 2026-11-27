//! Karaoke mode's stage as a widget of its own, filling the window
//! ([`StageView`]; on the Shader tab, the tab's picture: the stage drawn
//! at the widget's place and size), on iced's GPU through its shader
//! widget, its backdrop
//! the ground and the stage over it ([`super::present::Painter`]): the
//! lyrics, the lane, the count-in, the score, Next, the result. The
//! frames step the stage ([`StageView::frame`]); iced draws the last one
//! stepped. The buttons the stage draws in pixels (Next's button) take the
//! main button's press here and show the hand
//! ([`super::stage::Stage::targets`]).
//!
//! With iced's CPU renderer (no GPU) it draws nothing and says so
//! ([`StageView::gpu`]): karaoke needs a GPU, and the window says that
//! instead.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use iced::advanced::layout::{self, Layout};
use iced::advanced::renderer;
use iced::advanced::widget::{self, tree, Tree, Widget};
use iced::advanced::{mouse, Shell};
use iced::widget::shader::{self, Shader, Viewport};
use iced::{wgpu, Element, Event, Fill, Point, Rectangle, Size, Theme, Widget as _};

use super::backdrop::music::Song;
use super::backdrop::BackdropFrame;
use super::present::Painter;
use super::stage::{Stage, Tint};
use super::window::Message;
use crate::core::keymap::Op;
use crate::core::{Action, State};
use crate::gui::stage::gpu::StageFrame;
use crate::view::karaoke::KaraokePicture;

/// What the stage keeps between frames, and what waits for the next one
/// on screen.
struct Shared {
    stage: Stage,
    /// The stage as it is now; its uploads not sent yet.
    frame: Option<StageFrame>,
    /// The backdrop as it is now (its textures sent over as it first
    /// shows).
    backdrop: Option<BackdropFrame>,
    /// The mode was drawn in the last frame.
    shown: bool,
    /// Whether iced draws on the GPU (its shader widget works); `None`
    /// until it has drawn.
    gpu: Option<bool>,
    /// The window's scale factor (the stage is in its pixels), and its
    /// size in points.
    scale: f32,
    size: Size,
    /// When the frame before the last on the GPU had not got through by
    /// the time the last began: background GPU work gives way then.
    behind: bool,
    /// How long the last frame took to hand over to the GPU.
    handed: Duration,
    /// Where the widget was last drawn, in the window's pixels (none: not
    /// yet).
    bounds: Option<(u32, u32, u32, u32)>,
}

/// Karaoke mode's stage, shared with the frames iced draws.
#[derive(Clone)]
pub struct StageView(Arc<Mutex<Shared>>);

impl std::fmt::Debug for StageView {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StageView")
            .field("gpu", &self.lock().gpu)
            .finish_non_exhaustive()
    }
}

impl Default for StageView {
    fn default() -> StageView {
        StageView(Arc::new(Mutex::new(Shared {
            stage: Stage::default(),
            frame: None,
            backdrop: None,
            shown: false,
            gpu: None,
            scale: 1.0,
            size: Size::ZERO,
            behind: false,
            handed: Duration::ZERO,
            bounds: None,
        })))
    }
}

impl StageView {
    fn lock(&self) -> MutexGuard<'_, Shared> {
        self.0.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Whether iced draws on the GPU (until it has drawn: as if it did).
    pub fn gpu(&self) -> bool {
        self.lock().gpu.unwrap_or(true)
    }

    /// A frame of the mode: `picture` in a window of `size` pixels at
    /// `scale` (its points), `heard` seconds into the song (as `song` has
    /// it heard, once analysed), `dt` seconds after the last; the result
    /// screen in `tint` while one shows. With a GPU the stage is stepped
    /// and waits for the next frame on screen, over `backdrop`; without
    /// one nothing is. The mode just come, its score shows at once.
    pub fn frame(
        &self,
        (state, song): (&State, Option<&Song>),
        picture: &KaraokePicture,
        (size, scale): ((u32, u32), f32),
        (heard, dt): (f64, f32),
        tint: &Tint,
        backdrop: Option<BackdropFrame>,
    ) {
        let mut shared = self.lock();
        let shared = &mut *shared;
        shared.scale = scale;
        shared.size = Size::new(size.0 as f32 / scale, size.1 as f32 / scale);
        if !shared.shown {
            shared.stage.entered();
            shared.shown = true;
        }
        shared.backdrop = backdrop;
        if shared.gpu.unwrap_or(true) {
            let area = (0.0, 0.0, size.0 as f32, size.1 as f32);
            let song = (
                &super::stage::Song::of(state, song),
                super::stage::wanted(state),
            );
            let frame = shared.stage.scene(song, picture, area, (heard, dt), tint);
            shared.frame = merge(shared.frame.take(), frame);
        }
    }

    /// A frame without the mode: what the stage put into the atlas is
    /// still to be sent over (the stage counts on it), the backdrop goes.
    pub fn left(&self) {
        let mut shared = self.lock();
        shared.shown = false;
        shared.backdrop = None;
        shared.frame = merge(shared.frame.take(), None);
    }

    /// The window's size in points, as the last frame of the mode had it.
    pub fn size(&self) -> Size {
        self.lock().size
    }

    /// The widget's size in pixels where it was last drawn (none: not
    /// drawn yet): the Shader tab's picture steps the stage at it.
    pub fn drawn_size(&self) -> Option<(u32, u32)> {
        self.lock().bounds.map(|(_, _, w, h)| (w, h))
    }

    /// Whether what the stage draws moves by itself (its score rolling
    /// up), the mode showing on the GPU: frames must keep coming then.
    pub fn moving(&self) -> bool {
        let shared = self.lock();
        shared.shown && shared.gpu.unwrap_or(true) && shared.stage.moving()
    }

    /// Whether the frame before the last was late on the GPU, and how
    /// long the last took to hand over.
    pub fn late(&self) -> (bool, Duration) {
        let shared = self.lock();
        (shared.behind, shared.handed)
    }

    /// What a press of the main button at `at` (the window's points) does:
    /// the button the stage drew there in the frame shown, if there is
    /// one (Next's button).
    pub fn pressed(&self, at: Point) -> Option<Action> {
        let shared = self.lock();
        if !shared.shown || !shared.gpu.unwrap_or(true) {
            return None;
        }
        let (x, y) = (at.x * shared.scale, at.y * shared.scale);
        let inside = |[x0, y0, x1, y1]: [f32; 4]| (x0..=x1).contains(&x) && (y0..=y1).contains(&y);
        let mut targets = shared.stage.targets().iter();
        targets
            .find(|(at, _)| inside(*at))
            .map(|&(_, op): &(_, Op)| Action::Op(op))
    }

    /// The widget showing it, filling the window.
    pub fn view(&self) -> Element<'static, Message> {
        Staged {
            shader: iced::widget::shader(Input(self.clone()))
                .width(Fill)
                .height(Fill),
            view: self.clone(),
        }
        .boxed()
    }
}

/// `new` stage after `old`'s, not yet on the GPU: the text `old` put into
/// the atlas is still to be sent over first (the stage counts on it), even
/// where no stage shows now.
fn merge(old: Option<StageFrame>, new: Option<StageFrame>) -> Option<StageFrame> {
    let pending = old.map(|o| o.uploads).unwrap_or_default();
    if pending.is_empty() {
        return new;
    }
    let mut new = new.unwrap_or_default();
    let mut uploads = pending;
    uploads.append(&mut new.uploads);
    new.uploads = uploads;
    Some(new)
}

/// The widget: iced's shader widget, which tells whether iced draws on
/// the GPU as it draws.
struct Staged {
    shader: Shader<Message, Input>,
    view: StageView,
}

impl widget::Meta for Staged {}

impl Staged {
    fn inner(&self) -> &dyn Widget<Message, Theme, iced::Renderer> {
        &self.shader
    }

    fn inner_mut(&mut self) -> &mut dyn Widget<Message, Theme, iced::Renderer> {
        &mut self.shader
    }
}

impl Widget<Message, Theme, iced::Renderer> for Staged {
    fn size(&self) -> Size<iced::Length> {
        self.inner().size()
    }

    fn tag(&self) -> tree::Tag {
        self.inner().tag()
    }

    fn state(&self) -> tree::State {
        self.inner().state()
    }

    fn layout(&mut self, tree: &mut Tree, renderer: &iced::Renderer, limits: &layout::Limits) {
        self.inner_mut().layout(tree, renderer, limits);
    }

    fn update(
        &mut self,
        tree: &mut Tree,
        event: &Event,
        layout: Layout,
        cursor: mouse::Cursor,
        renderer: &iced::Renderer,
        shell: &mut Shell<'_, Message>,
        viewport: &Rectangle,
    ) {
        self.inner_mut()
            .update(tree, event, layout, cursor, renderer, shell, viewport);
    }

    fn draw(
        &self,
        tree: &Tree,
        renderer: &mut iced::Renderer,
        theme: &Theme,
        style: &renderer::Style,
        layout: Layout,
        cursor: mouse::Cursor,
        viewport: &Rectangle,
    ) {
        let gpu = matches!(renderer, iced::Renderer::Primary(_));
        self.view.lock().gpu = Some(gpu);
        if gpu {
            self.inner()
                .draw(tree, renderer, theme, style, layout, cursor, viewport);
        }
    }

    fn mouse_interaction(
        &self,
        tree: &Tree,
        layout: Layout,
        cursor: mouse::Cursor,
        viewport: &Rectangle,
        renderer: &iced::Renderer,
    ) -> mouse::Interaction {
        self.inner()
            .mouse_interaction(tree, layout, cursor, viewport, renderer)
    }
}

/// The shader widget's program: the stage's buttons under the pointer,
/// and the frame for the GPU.
struct Input(StageView);

impl shader::Program<Message> for Input {
    type State = ();
    type Primitive = Frame;

    fn update(
        &self,
        _state: &mut (),
        event: &Event,
        bounds: Rectangle,
        cursor: mouse::Cursor,
    ) -> Option<shader::Action<Message>> {
        let Event::Mouse(mouse::Event::ButtonPressed(mouse::Button::Left)) = event else {
            return None;
        };
        let at = cursor.position_in(bounds)?;
        let action = self.0.pressed(at)?;
        Some(shader::Action::publish(Message::Act(action)).and_capture())
    }

    fn draw(&self, _state: &(), _cursor: mouse::Cursor, _bounds: Rectangle) -> Frame {
        Frame(self.0.clone())
    }

    /// The hand over a button the stage drew (Next's button).
    fn mouse_interaction(
        &self,
        _state: &(),
        bounds: Rectangle,
        cursor: mouse::Cursor,
    ) -> mouse::Interaction {
        match cursor.position_in(bounds).and_then(|at| self.0.pressed(at)) {
            Some(_) => mouse::Interaction::Pointer,
            None => mouse::Interaction::default(),
        }
    }
}

/// A frame of the stage, for the GPU.
#[derive(Debug)]
pub struct Frame(StageView);

/// The painter on iced's device, and how far the GPU has got.
pub struct Pipeline {
    painter: Painter,
    /// Frames handed over so far.
    handed: u64,
    /// The last frame whose work before it the GPU has done (set from
    /// wgpu's callback).
    done: Arc<AtomicU64>,
}

impl shader::Pipeline for Pipeline {
    fn new(device: &wgpu::Device, queue: &wgpu::Queue, format: wgpu::TextureFormat) -> Pipeline {
        Pipeline {
            painter: Painter::new(device.clone(), queue.clone(), format),
            handed: 0,
            done: Arc::new(AtomicU64::new(0)),
        }
    }
}

impl shader::Primitive for Frame {
    type Pipeline = Pipeline;

    fn prepare(
        &self,
        pipeline: &mut Pipeline,
        device: &wgpu::Device,
        _queue: &wgpu::Queue,
        bounds: &Rectangle,
        viewport: &Viewport,
    ) {
        let started = Instant::now();
        // The callbacks of work done run here, without waiting.
        let _ = device.poll(wgpu::PollType::Poll);
        // The widget's place in the window's pixels: the whole window in
        // Karaoke mode, the tab's picture on the Shader tab.
        let s = viewport.scale_factor();
        let px = |v: f32| (v * s).round().max(0.0) as u32;
        let at = (
            px(bounds.x),
            px(bounds.y),
            px(bounds.width).max(1),
            px(bounds.height).max(1),
        );
        let size = iced::Size::new(at.2, at.3);
        let mut shared = self.0.lock();
        let shared = &mut *shared;
        shared.bounds = Some(at);
        // Late: what was submitted before the last frame had not got
        // through the GPU by now, kept from it by other work.
        shared.behind = pipeline.done.load(Ordering::Acquire) < pipeline.handed;
        // The stage stays as it is until the next frame stepped; what it
        // had to send is sent once.
        pipeline.painter.prepare(
            ((size.width, size.height), viewport.scale_factor()),
            shared.frame.as_ref(),
            shared.backdrop.as_ref(),
        );
        if let Some(frame) = &mut shared.frame {
            frame.uploads.clear();
        }
        // Called once what was submitted before now is done.
        pipeline.handed += 1;
        let (frame, done) = (pipeline.handed, Arc::clone(&pipeline.done));
        pipeline
            .painter
            .queue()
            .on_submitted_work_done(move || done.store(frame, Ordering::Release));
        shared.handed = started.elapsed();
    }

    fn draw(&self, pipeline: &Pipeline, pass: &mut wgpu::RenderPass<'_>) -> bool {
        if let Some((x, y, w, h)) = self.0.lock().bounds {
            pass.set_viewport(x as f32, y as f32, w as f32, h as f32, 0.0, 1.0);
        }
        pipeline.painter.draw(pass);
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::view::karaoke::Score;

    /// The picture with the score `points` and nothing else.
    fn scored(points: u32) -> KaraokePicture {
        KaraokePicture {
            bar: 0.1,
            lines: Vec::new(),
            count: None,
            cue: None,
            controls: None,
            start: None,
            before: Vec::new(),
            moved: 1.0,
            score: Some(Score { points, streak: 0 }),
            result: None,
        }
    }

    /// One frame of `points`, 1/60 s after the last, in a window of 640 ×
    /// 384 pixels at scale 1.
    fn frame(view: &StageView, points: u32) {
        let state = State::new("/m".into(), crate::config::Config::default());
        let size = ((640, 384), 1.0);
        let at = (0.0, 1.0 / 60.0);
        view.frame(
            (&state, None),
            &scored(points),
            size,
            at,
            &Tint::STAGE,
            None,
        );
    }

    /// The stage asks for frames while its score rolls up, and no more
    /// once it is at rest.
    #[test]
    fn the_stage_asks_for_frames_while_its_score_rolls() {
        let view = StageView::default();
        frame(&view, 12_400);
        assert!(view
            .lock()
            .frame
            .as_ref()
            .is_some_and(|f| !f.instances.is_empty()));
        frame(&view, 12_400);
        assert!(!view.moving());
        frame(&view, 13_200);
        let mut frames = 1;
        while view.moving() {
            frame(&view, 13_200);
            frames += 1;
            assert!(frames < 120, "still moving");
        }
        // Rolled in and its glow gone: about 0.4 s and 0.3 s of frames.
        assert!(frames > 30, "{frames}");
    }

    /// Points that came while the mode was left (20 000, then 30 000 on
    /// coming back) show at once: nothing rolls up.
    #[test]
    fn a_score_come_while_the_mode_was_left_shows_at_once() {
        let view = StageView::default();
        frame(&view, 20_000);
        view.left();
        assert!(!view.moving());
        frame(&view, 30_000);
        assert!(!view.moving());
    }

    /// Without a GPU nothing is stepped and nothing is pressed.
    #[test]
    fn without_a_gpu_nothing_is_stepped_or_pressed() {
        let view = StageView::default();
        view.lock().gpu = Some(false);
        frame(&view, 12_400);
        assert!(view.lock().frame.is_none() && !view.moving());
        assert!(view.pressed(Point::new(320.0, 300.0)).is_none());
    }
}
