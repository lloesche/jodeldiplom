//! The window's frames off screen ([`Offscreen`]), for the tests and the
//! examples: drawn as the window draws them, and read back or timed.

use jodel_gpu::Turned;

use super::{open, Painter};
use crate::gui::backdrop::BackdropFrame;

/// The window's frames drawn into a texture off screen and read back:
/// for the tests, the same uploads and passes as the window's (Karaoke
/// mode's).
pub struct Offscreen {
    painter: Turned<Painter>,
    format: wgpu::TextureFormat,
    /// What draws: the adapter's name and backend.
    pub name: String,
    /// The timestamps of [`Offscreen::time_frame`], where it times, and
    /// the texture it draws into.
    stamps: Option<crate::gui::present::stamps::Stamps>,
    target: Option<wgpu::Texture>,
}

impl Offscreen {
    /// On the process's GPU ([`jodel_gpu::chosen`]), into frames that take
    /// the bytes as they are (`srgb`: that encode them as sRGB, as some
    /// windows do); an error where there is no GPU.
    pub fn new(srgb: bool) -> Result<Offscreen, String> {
        Offscreen::opened(srgb, wgpu::Features::empty())
    }

    /// As [`Offscreen::new`] (frames taking the bytes as they are), and
    /// timing each step of a frame ([`Offscreen::time_frame`]); an error where
    /// the GPU cannot.
    pub fn timed() -> Result<Offscreen, String> {
        let mut gpu = Offscreen::opened(false, crate::gui::present::stamps::FEATURES)?;
        let painter = &gpu.painter;
        gpu.stamps = Some(crate::gui::present::stamps::Stamps::new(
            &painter.device,
            &painter.queue,
        ));
        Ok(gpu)
    }

    fn opened(srgb: bool, features: wgpu::Features) -> Result<Offscreen, String> {
        let _turn = jodel_gpu::setting_up();
        let adapter = jodel_gpu::chosen().adapter.clone()?;
        if !adapter.features().contains(features) {
            return Err(format!("{}: no {features:?}", adapter.get_info().name));
        }
        let (adapter, device, queue) = open(adapter, features)?;
        let info = adapter.get_info();
        let format = if srgb {
            wgpu::TextureFormat::Bgra8UnormSrgb
        } else {
            wgpu::TextureFormat::Bgra8Unorm
        };
        Ok(Offscreen {
            painter: Turned::new(Painter::new(device, queue, format)),
            format,
            name: format!("{} ({:?})", info.name, info.backend),
            stamps: None,
            target: None,
        })
    }

    /// A frame of `width` × `height` drawn as the window draws Karaoke
    /// mode, with `backdrop` and `stage` set for it, timed step by step (a
    /// step of the window's pass drawn in a pass of its own), not read
    /// back: the times are read with [`Offscreen::costs`]. An error where
    /// it does not time ([`Offscreen::timed`]).
    pub fn time_frame(
        &mut self,
        backdrop: Option<&BackdropFrame>,
        stage: Option<&crate::gui::stage::gpu::StageFrame>,
        width: u32,
        height: u32,
    ) -> Result<(), String> {
        let stamps = self.stamps.take().ok_or("not timed")?;
        let painter = &mut *self.painter;
        painter.stamps = Some(stamps);
        painter.mark(crate::gui::present::stamps::START);
        painter.set_backdrop(backdrop, (width, height));
        painter.set_stage(stage, (width, height));
        if self
            .target
            .as_ref()
            .is_none_or(|t| (t.width(), t.height()) != (width, height))
        {
            self.target = Some(painter.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("jodeldiplom timed"),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: self.format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            }));
        }
        if let Some(target) = &self.target {
            painter.paint(target, |_| ());
        }
        self.stamps = painter.stamps.take();
        Ok(())
    }

    /// The frames timed since the last look ([`Offscreen::time_frame`]),
    /// each its steps with the GPU's milliseconds for each.
    pub fn costs(&mut self) -> Result<Vec<Vec<(&'static str, f64)>>, String> {
        let stamps = self.stamps.as_mut().ok_or("not timed")?;
        stamps.take((&self.painter.device, &self.painter.queue))
    }

    /// Karaoke mode's stage over the next frame, as the window's.
    pub fn set_stage(
        &mut self,
        frame: Option<&crate::gui::stage::gpu::StageFrame>,
        width: u32,
        height: u32,
    ) {
        self.painter.set_stage(frame, (width.max(1), height.max(1)));
    }

    /// Karaoke mode's backdrop under the next frame, as the window's
    /// (none: its textures dropped, the ground black).
    pub fn set_backdrop(&mut self, frame: Option<&BackdropFrame>, width: u32, height: u32) {
        self.painter
            .set_backdrop(frame, (width.max(1), height.max(1)));
    }

    /// Karaoke mode's plain backdrop (no story: the window's ground) under
    /// the frames from now on, `width` × `height`.
    pub fn set_plain_ground(&mut self, width: u32, height: u32) {
        let cue = crate::gui::backdrop::Cue::default();
        let ground = crate::gui::backdrop::Visualiser::default().step(&cue, (width, height), 0.0);
        self.set_backdrop(Some(&ground), width, height);
    }

    /// The frames' pixels to a point from now on, as a window's scale (1
    /// to begin with; a Retina screen's 2).
    pub fn set_density(&mut self, density: f32) {
        self.painter.density = density;
    }

    /// Draws a frame of `width` × `height` of what was set (Karaoke mode:
    /// its backdrop, black without one, and its stage) and reads it back,
    /// as `0x00RRGGBB` words row by row.
    pub fn show_frame(&mut self, width: u32, height: u32) -> Result<Vec<u32>, String> {
        let painter = &mut *self.painter;
        painter.bind();
        let gpu = (&painter.device, &painter.queue);
        read_back(
            gpu,
            self.format,
            (width, height),
            wgpu::Color::BLACK,
            |pass| painter.draw(pass),
        )
    }
}

/// A frame of `width` × `height` in `format`, cleared to `clear`, drawn by
/// `draw` and read back, as `0x00RRGGBB` words row by row: for the tests.
pub(crate) fn read_back(
    (device, queue): (&wgpu::Device, &wgpu::Queue),
    format: wgpu::TextureFormat,
    (width, height): (u32, u32),
    clear: wgpu::Color,
    draw: impl FnOnce(&mut wgpu::RenderPass<'_>),
) -> Result<Vec<u32>, String> {
    let frame = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("jodeldiplom offscreen"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    // Rows padded to the copy's alignment.
    let row = (width * 4).div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT)
        * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("jodeldiplom offscreen"),
        size: u64::from(row) * u64::from(height),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let view = frame.create_view(&wgpu::TextureViewDescriptor::default());
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("jodeldiplom offscreen"),
    });
    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("jodeldiplom offscreen"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(clear),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        draw(&mut pass);
    }
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: &frame,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(row),
                rows_per_image: Some(height),
            },
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
    queue.submit([encoder.finish()]);
    let slice = buffer.slice(..);
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
    let mapped = slice.get_mapped_range().map_err(|e| e.to_string())?;
    let mut words = Vec::with_capacity(width as usize * height as usize);
    for line in mapped.chunks_exact(row as usize) {
        words.extend(
            line[..width as usize * 4]
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| u32::from_le_bytes([b[0], b[1], b[2], 0])),
        );
    }
    Ok(words)
}
