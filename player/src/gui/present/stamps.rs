//! The GPU's time between the steps of a frame (timestamp queries), to
//! measure what each pass
//! costs ([`crate::gui::present::Offscreen::timed`]): once the GPU has done the
//! step (waited for: Apple's GPUs start a pass that needs nothing before
//! it early, alongside what is still being drawn), a timestamp written at
//! the start of a pass of its own (clearing one pixel). A pass's
//! timestamps are all a device needs (Apple's GPUs write none between
//! passes). Each step is the time since the mark before it, read back
//! together: the GPU's own time and what submitting it took.

/// The most marks read back at once.
const MOST: u32 = 2048;
/// The mark a frame starts with.
pub(super) const START: &str = "start";

/// The timestamps of the frames marked, and what each step is called.
pub(super) struct Stamps {
    set: wgpu::QuerySet,
    resolved: wgpu::Buffer,
    read: wgpu::Buffer,
    /// The marks so far.
    marked: Vec<&'static str>,
    /// Nanoseconds a tick.
    period: f64,
    /// The pixel each mark's pass clears.
    pixel: wgpu::TextureView,
}

/// What a device needs to write them.
pub(super) const FEATURES: wgpu::Features = wgpu::Features::TIMESTAMP_QUERY;

impl Stamps {
    pub(super) fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Stamps {
        let bytes = u64::from(MOST) * 8;
        let buffer = |usage| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("jodeldiplom stamps"),
                size: bytes,
                usage,
                mapped_at_creation: false,
            })
        };
        Stamps {
            set: device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("jodeldiplom stamps"),
                ty: wgpu::QueryType::Timestamp,
                count: MOST,
            }),
            resolved: buffer(wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC),
            read: buffer(wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ),
            marked: Vec::new(),
            period: f64::from(queue.get_timestamp_period()),
            pixel: device
                .create_texture(&wgpu::TextureDescriptor {
                    label: Some("jodeldiplom stamps"),
                    size: wgpu::Extent3d {
                        width: 1,
                        height: 1,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                    view_formats: &[],
                })
                .create_view(&wgpu::TextureViewDescriptor::default()),
        }
    }

    /// Marks the end of step `name` ([`START`]: a frame's start), waiting
    /// for everything submitted before to be done.
    pub(super) fn mark(
        &mut self,
        (device, queue): (&wgpu::Device, &wgpu::Queue),
        name: &'static str,
    ) {
        let at = self.marked.len() as u32;
        assert!(at < MOST, "more than {MOST} marks before they are read");
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("jodeldiplom stamps"),
        });
        encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("jodeldiplom stamps"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &self.pixel,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Discard,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: Some(wgpu::RenderPassTimestampWrites {
                query_set: &self.set,
                beginning_of_pass_write_index: Some(at),
                end_of_pass_write_index: None,
            }),
            occlusion_query_set: None,
            multiview_mask: None,
        });
        queue.submit([encoder.finish()]);
        self.marked.push(name);
    }

    /// The frames marked, read back: each its steps, with the
    /// milliseconds since the mark before each. Marking starts again.
    pub(super) fn take(
        &mut self,
        (device, queue): (&wgpu::Device, &wgpu::Queue),
    ) -> Result<Vec<Vec<(&'static str, f64)>>, String> {
        let n = self.marked.len() as u32;
        if n == 0 {
            return Ok(Vec::new());
        }
        let bytes = u64::from(n) * 8;
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("jodeldiplom stamps"),
        });
        encoder.resolve_query_set(&self.set, 0..n, &self.resolved, 0);
        encoder.copy_buffer_to_buffer(&self.resolved, 0, &self.read, 0, bytes);
        queue.submit([encoder.finish()]);
        let slice = self.read.slice(..bytes);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| e.to_string())?;
        rx.recv()
            .map_err(|e| e.to_string())?
            .map_err(|e| e.to_string())?;
        let ticks: Vec<u64> = {
            let mapped = slice.get_mapped_range().map_err(|e| e.to_string())?;
            let words = mapped.as_chunks::<8>().0;
            words.iter().map(|b| u64::from_le_bytes(*b)).collect()
        };
        self.read.unmap();
        let mut frames: Vec<Vec<(&'static str, f64)>> = Vec::new();
        for (k, &name) in self.marked.iter().enumerate() {
            if name == START {
                frames.push(Vec::new());
            } else if let Some(frame) = frames.last_mut() {
                let ms = ticks[k].saturating_sub(ticks[k - 1]) as f64 * self.period / 1e6;
                frame.push((name, ms));
            }
        }
        self.marked.clear();
        Ok(frames)
    }
}
