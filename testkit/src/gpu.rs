//! The shader checks' small compute runner: a device on the window's GPU
//! ([`gpu`]), and a shader's own functions worked out at samples in a
//! compute pass ([`run`], [`run_reading`]), the samples read as an array
//! of their own WGSL type ([`Input`]), the results as vec4s; for holding a
//! shader to its processor's reference, near enough ([`near`]).

/// A sample a check's shader reads: plain numbers, laid out as its WGSL
/// type [`Input::WGSL`] (the shader declares the type when it is its own).
pub trait Input: bytemuck::Pod {
    /// Its WGSL type.
    const WGSL: &'static str;
}

impl Input for [f32; 4] {
    const WGSL: &'static str = "vec4<f32>";
}

/// A device on the window's GPU (the one the player chooses, set up in
/// its turn), none without one.
pub fn gpu() -> Option<(wgpu::Device, wgpu::Queue)> {
    let _turn = jodel_gpu::setting_up();
    let opened = jodel_gpu::chosen().adapter.clone().and_then(|adapter| {
        let limits = wgpu::Limits::downlevel_defaults().using_resolution(adapter.limits());
        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("jodeldiplom shader check"),
            required_limits: limits,
            ..Default::default()
        }))
        .map_err(|e| format!("{}: {e}", adapter.get_info().name))
    });
    match opened {
        Ok(opened) => Some(opened),
        Err(e) => {
            eprintln!("no GPU ({e}): skipped");
            None
        }
    }
}

/// The compute shader [`run`] makes: `source`, the samples as `ins` (an
/// array of `T`) and the results as `outs` (vec4s) at `group(1)`, and the
/// entry `check` running `body` over `i`, the sample.
pub fn code<T: Input>(source: &str, body: &str) -> String {
    format!(
        "{source}
@group(1) @binding(0) var<storage, read> ins: array<{}>;
@group(1) @binding(1) var<storage, read_write> outs: array<vec4<f32>>;
@compute @workgroup_size(1)
fn check(@builtin(global_invocation_id) id: vec3<u32>) {{
    let i = id.x;
{body}
}}
",
        T::WGSL
    )
}

/// `body` (WGSL over `i`, the sample, reading `ins` and writing `outs`)
/// after `source`, run on the GPU for `n` samples, `per` vec4s out each.
pub fn run<T: Input>(
    gpu: &(wgpu::Device, wgpu::Queue),
    source: &str,
    body: &str,
    ins: &[T],
    (n, per): (usize, usize),
) -> Vec<[f32; 4]> {
    run_reading(gpu, (source, body), ins, (n, per), None)
}

/// As [`run`], `source` and `body` reading a texture (`group(0)`: it at
/// binding 0, a linear sampler at 1) where one is given.
pub fn run_reading<T: Input>(
    (device, queue): &(wgpu::Device, wgpu::Queue),
    (source, body): (&str, &str),
    ins: &[T],
    (n, per): (usize, usize),
    texture: Option<&wgpu::TextureView>,
) -> Vec<[f32; 4]> {
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("jodeldiplom shader check"),
        source: wgpu::ShaderSource::Wgsl(code::<T>(source, body).into()),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("jodeldiplom shader check"),
        layout: None,
        module: &module,
        entry_point: Some("check"),
        compilation_options: Default::default(),
        cache: None,
    });
    let bytes: &[u8] = bytemuck::cast_slice(ins);
    let size = (n * per * 16) as u64;
    let buffer = |size: u64, usage| {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("jodeldiplom shader check"),
            size,
            usage,
            mapped_at_creation: false,
        })
    };
    use wgpu::BufferUsages as U;
    let input = buffer(bytes.len() as u64, U::STORAGE | U::COPY_DST);
    let output = buffer(size, U::STORAGE | U::COPY_SRC);
    let back = buffer(size, U::MAP_READ | U::COPY_DST);
    queue.write_buffer(&input, 0, bytes);
    let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("jodeldiplom shader check"),
        layout: &pipeline.get_bind_group_layout(1),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: input.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: output.as_entire_binding(),
            },
        ],
    });
    let read = texture.map(|view| {
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("jodeldiplom shader check"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        })
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        if let Some(read) = &read {
            pass.set_bind_group(0, read, &[]);
        }
        pass.set_bind_group(1, &bind, &[]);
        pass.dispatch_workgroups(n as u32, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&output, 0, &back, 0, size);
    queue.submit([encoder.finish()]);
    let slice = back.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("the GPU done");
    rx.recv().unwrap().expect("mapped");
    let mapped = slice.get_mapped_range().expect("read back");
    bytemuck::cast_slice::<u8, [f32; 4]>(&mapped).to_vec()
}

/// Whether `a` is `b` within `abs` (and a share `rel` of it).
pub fn near(a: f32, b: f32, abs: f32, rel: f32) -> bool {
    (a - b).abs() <= abs + rel * b.abs()
}
