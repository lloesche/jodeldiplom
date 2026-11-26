//! The window's frames off screen ([`Offscreen`]), for the tests and the
//! examples: drawn as the window draws them, and read back or timed.

use jodel_gpu::Turned;

use super::{open, Painter, TEXTURE_FORMAT};
use crate::gui::backdrop::BackdropFrame;
use crate::gui::canvas::{Canvas, Changes, LayerFrame};

/// The window's frames drawn into a texture off screen and read back:
/// for the tests, the same uploads and pass as the window's.
pub struct Offscreen {
    painter: Turned<Painter>,
    format: wgpu::TextureFormat,
    /// What draws: the adapter's name and backend.
    pub name: String,
    /// The timestamps of [`Offscreen::time_frame`], where it times, and
    /// the texture it draws into.
    stamps: Option<crate::gui::stamps::Stamps>,
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
        let mut gpu = Offscreen::opened(false, crate::gui::stamps::FEATURES)?;
        let painter = &gpu.painter;
        gpu.stamps = Some(crate::gui::stamps::Stamps::new(
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
            TEXTURE_FORMAT
        };
        Ok(Offscreen {
            painter: Turned::new(Painter::new(device, queue, format)),
            format,
            name: format!("{} ({:?})", info.name, info.backend),
            stamps: None,
            target: None,
        })
    }

    /// A frame of `width` × `height` drawn as the window draws it (over the
    /// pixmap [`Offscreen::show_pixels`] last sent, of that size), with
    /// `backdrop` and `stage` set for it, timed step by step (a step of the
    /// window's pass drawn in a pass of its own), not read back: the times
    /// are read with [`Offscreen::costs`]. An error where it does not time
    /// ([`Offscreen::timed`]) or nothing was sent.
    pub fn time_frame(
        &mut self,
        backdrop: Option<&BackdropFrame>,
        stage: Option<&crate::gui::stage_gpu::StageFrame>,
        width: u32,
        height: u32,
    ) -> Result<(), String> {
        let stamps = self.stamps.take().ok_or("not timed")?;
        let painter = &mut *self.painter;
        painter.stamps = Some(stamps);
        painter.mark(crate::gui::stamps::START);
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
        let painted = self
            .target
            .as_ref()
            .is_some_and(|t| painter.paint(t, |_| ()));
        self.stamps = painter.stamps.take();
        if painted {
            Ok(())
        } else {
            Err("nothing to draw".into())
        }
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
        frame: Option<&crate::gui::stage_gpu::StageFrame>,
        width: u32,
        height: u32,
    ) {
        self.painter.set_stage(frame, (width.max(1), height.max(1)));
    }

    /// Karaoke mode's backdrop under the next frame, as the window's
    /// (none: its textures dropped).
    pub fn set_backdrop(&mut self, frame: Option<&BackdropFrame>, width: u32, height: u32) {
        self.painter
            .set_backdrop(frame, (width.max(1), height.max(1)));
    }

    /// The frames' pixels to a point from now on, as a window's scale (1
    /// to begin with; a Retina screen's 2).
    pub fn set_density(&mut self, density: f32) {
        self.painter.density = density;
    }

    /// How many covers the backdrop holds on the GPU.
    pub fn backdrop_held(&self) -> usize {
        self.painter.backdrop.held()
    }

    /// Draws a frame of `pixels` (`width` × `height` pixmap words row by
    /// row, premultiplied over the backdrop: [`crate::gui::pixels::SEE_THROUGH`])
    /// with the backdrop and the stage set, and reads it back.
    pub fn show_pixels(
        &mut self,
        pixels: &[u32],
        width: u32,
        height: u32,
    ) -> Result<Vec<u32>, String> {
        if self
            .painter
            .kept
            .as_ref()
            .is_none_or(|k| k.window != (width, height))
        {
            self.painter.kept = Some(self.painter.texture(width, height));
            self.painter.bind = None;
        }
        let (w, h) = (width as usize, height as usize);
        self.painter.write(pixels, w, (0, 0, w, h));
        self.painter.set_layer(None);
        self.read_back(width, height)
    }

    /// Sends over `changes` and `layer` as the window does, draws a frame
    /// of `width` × `height` and reads it back, as `0x00RRGGBB` words row
    /// by row.
    pub fn show(
        &mut self,
        canvas: &Canvas,
        changes: &Changes,
        width: u32,
        height: u32,
        layer: Option<LayerFrame>,
    ) -> Result<Vec<u32>, String> {
        self.painter.upload(canvas, changes, width, height);
        self.painter.set_layer(layer);
        self.read_back(width, height)
    }

    /// The frame drawn from what was sent over, read back.
    fn read_back(&mut self, width: u32, height: u32) -> Result<Vec<u32>, String> {
        let device = self.painter.device.clone();
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
            format: self.format,
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
        let painted = self.painter.paint(&frame, |encoder| {
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
        });
        if !painted {
            return Err("nothing to draw".into());
        }
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
}
