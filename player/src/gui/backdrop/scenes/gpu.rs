//! The story scenes on the GPU ([`super`]): the scene drawn and the next
//! (sent over ahead of its cut, a texture a frame), each its picture,
//! plate (in layers) and shape in textures with their smaller levels (the
//! light behind a layer's edge and its blurred share read them), made from
//! the texels the scenes' job made ([`crate::work::scenes::Ready`]); the
//! rest dropped. A frame's scene is drawn by its shader (`scenes.wgsl`, its
//! maths in [`super::look`]) over the whole of the backdrop's texture, in
//! the backdrop's own pass.

use super::super::gpu::{
    full_screen, levels, sampler_entry, texture, texture_entry, Stretch, Upload,
};
use super::{SceneFrame, SceneLook};
use crate::work::scenes::{Moved, Ready, Story};

/// The bytes of a scene's numbers.
const LOOK_BYTES: u64 = std::mem::size_of::<SceneLook>() as u64;

/// The scenes' shader for a target that takes linear light (`linear`) or
/// not: the backdrop's full-screen triangle, the scenes' maths and how
/// their light is shown ([`super::look`]), the light's tone and
/// `scenes.wgsl`.
pub fn shader(linear: bool) -> String {
    format!(
        "const LINEAR: f32 = {};\n{}{}{}{}{}",
        if linear { "1.0" } else { "0.0" },
        super::super::look::SCREEN,
        super::look::wgsl(),
        super::look::FINISH,
        crate::gui::bloom::TONE,
        include_str!("scenes.wgsl")
    )
}

/// A scene's textures sent over so far.
#[derive(Default)]
struct Textures {
    picture: Option<wgpu::Texture>,
    plate: Option<wgpu::Texture>,
    shape: Option<wgpu::Texture>,
}

/// A scene on the GPU: which (its story's id, its place in it), its
/// textures, its numbers and what its shader reads (once all are there).
struct Held {
    story: u64,
    index: usize,
    textures: Textures,
    uniforms: wgpu::Buffer,
    bind: Option<wgpu::BindGroup>,
}

impl Held {
    /// `story`'s scene `index`, none of its textures sent yet.
    fn new(device: &wgpu::Device, story: &Story, index: usize) -> Held {
        Held {
            story: story.id,
            index,
            textures: Textures::default(),
            uniforms: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("jodeldiplom scene look"),
                size: LOOK_BYTES,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            bind: None,
        }
    }

    /// Whether it is `story`'s scene `index`.
    fn is(&self, story: &Story, index: usize) -> bool {
        self.story == story.id && self.index == index
    }
}

/// The scenes' pipeline and the scenes on the GPU.
pub struct ScenePass {
    pipeline: wgpu::RenderPipeline,
    layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    held: Vec<Held>,
}

impl ScenePass {
    /// For a target of `format` whose pass reads its texture as linear
    /// light (`linear`) or as it is.
    pub fn new(device: &wgpu::Device, format: wgpu::TextureFormat, linear: bool) -> ScenePass {
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("jodeldiplom scenes"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(LOOK_BYTES),
                    },
                    count: None,
                },
                texture_entry(1),
                texture_entry(2),
                texture_entry(3),
                sampler_entry(4),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("jodeldiplom scenes"),
            bind_group_layouts: &[Some(&layout)],
            ..Default::default()
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("jodeldiplom scenes"),
            source: wgpu::ShaderSource::Wgsl(shader(linear).into()),
        });
        let pipeline = full_screen(
            device,
            (&pipeline_layout, &module),
            (format, "fs"),
            None,
            "jodeldiplom scenes",
        );
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("jodeldiplom scenes"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        });
        ScenePass {
            pipeline,
            layout,
            sampler,
            held: Vec::new(),
        }
    }

    /// Every scene dropped.
    pub fn clear(&mut self) {
        self.held.clear();
    }

    /// The next frame's scene (none: none, every scene dropped): the one
    /// drawn sent over whole where it is new, the next a texture a frame
    /// (so no frame sends more than one scene's; their smaller levels made
    /// in `encoder` by `stretch`), the rest dropped, its numbers written.
    /// The scene drawn, where it is there to draw.
    pub(in crate::gui::backdrop) fn set(
        &mut self,
        (device, queue): (&wgpu::Device, &wgpu::Queue),
        (encoder, stretch): (&mut wgpu::CommandEncoder, &mut Stretch),
        frame: Option<&SceneFrame>,
    ) -> Option<Drawn> {
        let Some(frame) = frame else {
            self.clear();
            return None;
        };
        let story = &frame.story;
        let (drawn, look) = frame.drawn;
        let wanted: Vec<usize> = [Some(drawn), frame.next].into_iter().flatten().collect();
        self.held.retain(|h| wanted.iter().any(|&i| h.is(story, i)));
        for &index in &wanted {
            let at = match self.held.iter().position(|h| h.is(story, index)) {
                Some(at) => at,
                None => {
                    self.held.push(Held::new(device, story, index));
                    self.held.len() - 1
                }
            };
            let whole = index == drawn;
            let scene = &story.scenes[index];
            let held = &mut self.held[at];
            while held.bind.is_none() {
                if let Some(bind) = bind(device, (&self.layout, &self.sampler), held, scene) {
                    held.bind = Some(bind);
                    break;
                }
                send(
                    (device, queue),
                    (encoder, &mut *stretch),
                    &mut held.textures,
                    scene,
                );
                if !whole {
                    break;
                }
            }
        }
        let held = self
            .held
            .iter()
            .find(|h| h.is(story, drawn) && h.bind.is_some())?;
        queue.write_buffer(&held.uniforms, 0, bytemuck::bytes_of(&look));
        Some(Drawn {
            story: story.id,
            index: drawn,
            look,
        })
    }

    /// The scene `drawn` (as [`ScenePass::set`] gave it) drawn into
    /// `pass`.
    pub(in crate::gui::backdrop) fn draw(&self, pass: &mut wgpu::RenderPass<'_>, drawn: &Drawn) {
        let held = (self.held.iter()).find(|h| h.story == drawn.story && h.index == drawn.index);
        if let Some(bind) = held.and_then(|h| h.bind.as_ref()) {
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, bind, &[]);
            pass.draw(0..3, 0..1);
        }
    }
}

/// `scene`'s next texture not yet sent over (`textures`): its picture, its
/// plate (in layers), its shape; each with its levels.
fn send(
    (device, queue): (&wgpu::Device, &wgpu::Queue),
    (encoder, stretch): (&mut wgpu::CommandEncoder, &mut Stretch),
    textures: &mut Textures,
    scene: &Ready,
) {
    let srgb = |bytes| Upload {
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        texel: 4,
        size: scene.size,
        bytes,
    };
    let plate = match &scene.moved {
        Moved::Layers { plate, .. } => Some(plate),
        Moved::Warp => None,
    };
    let (slot, upload) = if textures.picture.is_none() {
        (&mut textures.picture, srgb(&scene.picture))
    } else if let (None, Some(plate)) = (&textures.plate, plate) {
        (&mut textures.plate, srgb(plate))
    } else {
        let shape = Upload {
            format: wgpu::TextureFormat::Rgba8Unorm,
            texel: 4,
            size: scene.shape_size,
            bytes: &scene.shape,
        };
        (&mut textures.shape, shape)
    };
    let levels = levels(upload.size);
    *slot = Some(texture(
        (device, queue),
        (encoder, stretch),
        &upload,
        levels,
    ));
}

/// What `held`'s shader reads (by `layout`, with `sampler`): its numbers,
/// its picture, its plate (the picture again for a warp), its shape; none
/// until all of `scene`'s are sent over.
fn bind(
    device: &wgpu::Device,
    (layout, sampler): (&wgpu::BindGroupLayout, &wgpu::Sampler),
    held: &Held,
    scene: &Ready,
) -> Option<wgpu::BindGroup> {
    let t = &held.textures;
    let (picture, shape) = (t.picture.as_ref()?, t.shape.as_ref()?);
    let plate = match scene.moved {
        Moved::Layers { .. } => t.plate.as_ref()?,
        Moved::Warp => picture,
    };
    let view = |t: &wgpu::Texture| t.create_view(&wgpu::TextureViewDescriptor::default());
    let (picture, plate, shape) = (view(picture), view(plate), view(shape));
    fn entry(binding: u32, view: &wgpu::TextureView) -> wgpu::BindGroupEntry<'_> {
        wgpu::BindGroupEntry {
            binding,
            resource: wgpu::BindingResource::TextureView(view),
        }
    }
    Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("jodeldiplom scenes"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: held.uniforms.as_entire_binding(),
            },
            entry(1, &picture),
            entry(2, &plate),
            entry(3, &shape),
            wgpu::BindGroupEntry {
                binding: 4,
                resource: wgpu::BindingResource::Sampler(sampler),
            },
        ],
    }))
}

/// Which scene a frame draws (its story's id, its place in it) and its
/// numbers: what tells one frame from the last.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(in crate::gui::backdrop) struct Drawn {
    story: u64,
    index: usize,
    look: SceneLook,
}
