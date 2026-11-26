//! The old canvas on iced's GPU ([`super::embed`]): a primitive of iced's
//! shader widget that hands each frame's changed pixels to the
//! [`Pixmap`], and draws it into iced's render pass, under what iced draws
//! over it. The pass draws in window pixels, so the widget is at the
//! window's top left corner.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use iced::widget::shader::{self, Viewport};
use iced::{wgpu, Rectangle};

use super::embed::Embed;
use super::present::pixmap::Pixmap;

/// A frame of the old canvas, for the GPU.
#[derive(Debug)]
pub struct Frame(Embed);

impl Frame {
    pub(super) fn new(embed: Embed) -> Frame {
        Frame(embed)
    }
}

/// The pixmap on iced's device, and how far the GPU has got.
pub struct Pipeline {
    pixmap: Pixmap,
    /// Frames handed over so far.
    handed: u64,
    /// The last frame whose work before it the GPU has done (set from
    /// wgpu's callback).
    done: Arc<AtomicU64>,
}

impl shader::Pipeline for Pipeline {
    fn new(device: &wgpu::Device, queue: &wgpu::Queue, format: wgpu::TextureFormat) -> Pipeline {
        Pipeline {
            pixmap: Pixmap::new(device.clone(), queue.clone(), format),
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
        debug_assert!(
            bounds.x == 0.0 && bounds.y == 0.0,
            "the old canvas fills the window from its top left: {bounds:?}"
        );
        let started = Instant::now();
        // The callbacks of work done run here, without waiting.
        let _ = device.poll(wgpu::PollType::Poll);
        let size = viewport.physical_size();
        let mut shared = self.0.lock();
        let shared = &mut *shared;
        // Late: what was submitted before the last frame had not got
        // through the GPU by now, kept from it by other work.
        shared.behind = pipeline.done.load(Ordering::Acquire) < pipeline.handed;
        let changes = std::mem::take(&mut shared.changes);
        pipeline
            .pixmap
            .upload(&shared.canvas, &changes, size.width, size.height);
        // Called once what was submitted before now is done.
        pipeline.handed += 1;
        let (frame, done) = (pipeline.handed, Arc::clone(&pipeline.done));
        pipeline
            .pixmap
            .queue()
            .on_submitted_work_done(move || done.store(frame, Ordering::Release));
        shared.handed = started.elapsed();
    }

    fn draw(&self, pipeline: &Pipeline, pass: &mut wgpu::RenderPass<'_>) -> bool {
        pipeline.pixmap.draw(pass)
    }
}
