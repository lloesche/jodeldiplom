//! The old canvas's frames on the GPU ([`Pixmap`], inside iced's frame
//! through [`crate::gui::embed`]'s widget): the pixmap is kept in a
//! texture the size of the window, each frame sends over only the pixels
//! that changed ([`Changes`]), and a triangle over the whole window copies
//! the texture onto it, pixel for pixel; a few rows a frame, not the whole
//! window. Karaoke mode never shows it: the stage's widget draws that
//! ([`crate::gui::stage_view`]).

use crate::gui::canvas::{Canvas, Changes};
use crate::gui::pixels::PixelRect;

const SHADER: &str = r"
@group(0) @binding(0) var pixels: texture_2d<f32>;

// One triangle that covers the window.
@vertex
fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let x = f32((i << 1u) & 2u) * 2.0 - 1.0;
    let y = f32(i & 2u) * 2.0 - 1.0;
    return vec4<f32>(x, y, 0.0, 1.0);
}

// Each window pixel is the pixmap's there.
@fragment
fn fs(@builtin(position) at: vec4<f32>) -> @location(0) vec4<f32> {
    return vec4<f32>(textureLoad(pixels, vec2<i32>(at.xy), 0).rgb, 1.0);
}
";

/// The pixmap's words (`0x00RRGGBB`) as they lie in memory, B G R x.
pub(super) const TEXTURE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Bgra8Unorm;

/// The pixmap's copy on the GPU, for one window size.
struct Kept {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    /// The window size it is for.
    window: (u32, u32),
    /// The pixmap size it last took.
    pixmap: (usize, usize),
}

/// The pixmap kept on the GPU and the pass that copies it onto a frame.
pub(crate) struct Pixmap {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::RenderPipeline,
    layout: wgpu::BindGroupLayout,
    /// The texture's format as the shader reads it: sRGB where the frame
    /// is, so the colours come out as they went in.
    view_format: wgpu::TextureFormat,
    kept: Option<Kept>,
    /// What the pass reads: made again when the texture is.
    bind: Option<wgpu::BindGroup>,
}

impl Pixmap {
    /// For frames of `format`.
    pub(crate) fn new(
        device: wgpu::Device,
        queue: wgpu::Queue,
        format: wgpu::TextureFormat,
    ) -> Pixmap {
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("jodeldiplom pixmap"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            }],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("jodeldiplom pixmap"),
            bind_group_layouts: &[Some(&layout)],
            ..Default::default()
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("jodeldiplom pixmap"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("jodeldiplom pixmap"),
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
        // The pixmap read as linear light (through an sRGB view) where the
        // frame takes it so.
        let view_format = if format.is_srgb() {
            wgpu::TextureFormat::Bgra8UnormSrgb
        } else {
            TEXTURE_FORMAT
        };
        Pixmap {
            device,
            queue,
            pipeline,
            layout,
            view_format,
            kept: None,
            bind: None,
        }
    }

    /// The device's queue it draws with.
    pub(crate) fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    /// The pixels of `canvas` in `changes` into the texture for a window
    /// of `width` × `height` (a new one, filled whole, for a new window or
    /// pixmap size).
    pub(crate) fn upload(&mut self, canvas: &Canvas, changes: &Changes, width: u32, height: u32) {
        let (width, height) = (width.max(1), height.max(1));
        let (pixels, pw, ph) = canvas.pixels();
        let same = self
            .kept
            .as_ref()
            .is_some_and(|k| k.window == (width, height) && k.pixmap == (pw, ph));
        if !same {
            if self
                .kept
                .as_ref()
                .is_none_or(|k| k.window != (width, height))
            {
                self.kept = Some(self.texture(width, height));
                self.bind = None;
            }
            // The whole window: the pixmap and the background beyond it.
            let (w, h) = (width as usize, height as usize);
            let mut whole = vec![0u32; w * h];
            canvas.write_xrgb(&mut whole, w, h);
            self.write(&whole, w, (0, 0, w, h));
            if let Some(kept) = &mut self.kept {
                kept.pixmap = (pw, ph);
            }
        } else {
            // Within the pixmap and the window.
            let (cw, chh) = (pw.min(width as usize), ph.min(height as usize));
            let all = [(0, 0, cw, chh)];
            let rects: &[PixelRect] = if changes.all { &all } else { &changes.rects };
            for &(x, y, w, h) in rects {
                let (w, h) = (w.min(cw.saturating_sub(x)), h.min(chh.saturating_sub(y)));
                if w > 0 && h > 0 {
                    self.write(pixels, pw, (x, y, w, h));
                }
            }
        }
        if let (None, Some(kept)) = (&self.bind, &self.kept) {
            self.bind = Some(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("jodeldiplom pixmap"),
                layout: &self.layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&kept.view),
                }],
            }));
        }
    }

    /// The pixmap over the whole frame, into `pass` (its viewport the
    /// window's); false where none was sent over yet.
    pub(crate) fn draw(&self, pass: &mut wgpu::RenderPass<'_>) -> bool {
        let Some(bind) = &self.bind else {
            return false;
        };
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, bind, &[]);
        pass.draw(0..3, 0..1);
        true
    }

    /// The rect `(x, y, w, h)` of `pixels` (rows of `stride` words) into
    /// the texture at the same place.
    fn write(&self, pixels: &[u32], stride: usize, (x, y, w, h): PixelRect) {
        let Some(kept) = &self.kept else {
            return;
        };
        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &kept.texture,
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x: x as u32,
                    y: y as u32,
                    z: 0,
                },
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(pixels),
            wgpu::TexelCopyBufferLayout {
                offset: ((y * stride + x) * 4) as u64,
                bytes_per_row: Some((stride * 4) as u32),
                rows_per_image: Some(h as u32),
            },
            wgpu::Extent3d {
                width: w as u32,
                height: h as u32,
                depth_or_array_layers: 1,
            },
        );
    }

    fn texture(&self, width: u32, height: u32) -> Kept {
        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("jodeldiplom pixmap"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: TEXTURE_FORMAT,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: if self.view_format == TEXTURE_FORMAT {
                &[]
            } else {
                &[wgpu::TextureFormat::Bgra8UnormSrgb]
            },
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor {
            format: Some(self.view_format),
            ..Default::default()
        });
        Kept {
            texture,
            view,
            window: (width, height),
            pixmap: (0, 0),
        }
    }
}
