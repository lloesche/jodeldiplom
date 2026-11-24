//! The GPU this process uses, chosen once ([`chosen`]), and the turns
//! every setup and teardown of a `wgpu` instance, adapter or device takes
//! ([`setting_up`]).

use std::cell::Cell;
use std::ops::{Deref, DerefMut};
use std::sync::{Mutex, MutexGuard, OnceLock, PoisonError};

/// Whose turn it is to set up or tear down the GPU ([`setting_up`]).
static SETUP: Mutex<()> = Mutex::new(());

thread_local! {
    /// The turns this thread holds (a setup inside another's turn takes
    /// none of its own).
    static TURNS: Cell<usize> = const { Cell::new(0) };
}

/// A turn at the GPU's setup: while one is held, no other thread of the
/// process creates or drops a `wgpu` instance, adapter or device through
/// this crate. Some drivers do not take two at once: Intel's on Windows,
/// whose control library each instance and device starts and stops,
/// corrupts the heap (an access violation under a debug heap) when two
/// threads do it together. The instance and the adapter are made once
/// ([`chosen`]), so a turn now mostly guards that one-time choice and
/// each device's setup and teardown. Every setup and teardown here takes
/// its turn; code with a GPU setup of its own (a window's) takes one
/// around it. A thread may take turns inside its own.
pub struct Turn {
    _held: Option<MutexGuard<'static, ()>>,
}

/// Takes a turn at the GPU's setup, waiting for the one under way.
pub fn setting_up() -> Turn {
    let held = (TURNS.get() == 0).then(|| SETUP.lock().unwrap_or_else(PoisonError::into_inner));
    TURNS.set(TURNS.get() + 1);
    Turn { _held: held }
}

impl Drop for Turn {
    fn drop(&mut self) {
        TURNS.set(TURNS.get() - 1);
    }
}

/// Something that holds a device (a model's state on it, a window's
/// surface), dropped in a turn ([`setting_up`]): the last of a device's
/// handles to go tears it down.
pub struct Turned<T>(Option<T>);

impl<T> Turned<T> {
    pub fn new(value: T) -> Turned<T> {
        Turned(Some(value))
    }
}

impl<T> Deref for Turned<T> {
    type Target = T;
    fn deref(&self) -> &T {
        self.0.as_ref().expect("dropped only once")
    }
}

impl<T> DerefMut for Turned<T> {
    fn deref_mut(&mut self) -> &mut T {
        self.0.as_mut().expect("dropped only once")
    }
}

impl<T> Drop for Turned<T> {
    fn drop(&mut self) {
        let _turn = setting_up();
        drop(self.0.take());
    }
}

/// The process's GPU, chosen once ([`chosen`]): one `wgpu` instance and
/// the adapter on it that every device of this crate comes from.
pub struct Chosen {
    /// The instance the adapters are on (a surface for any of them is
    /// made on it).
    pub instance: wgpu::Instance,
    /// The best adapter: a high-performance one preferred, a software
    /// renderer refused; why there is none, if there is none.
    pub adapter: Result<wgpu::Adapter, String>,
    /// The same card ([`card`]) through DirectX 12, kept only where
    /// [`Chosen::adapter`] is on another backend and the card has
    /// DirectX 12 too.
    pub sibling: Option<wgpu::Adapter>,
}

/// The GPU's choice, made once.
static CHOSEN: OnceLock<Chosen> = OnceLock::new();

/// The card an adapter is on: its PCI vendor and device ids, the same on
/// every backend.
pub fn card(info: &wgpu::AdapterInfo) -> (u32, u32) {
    (info.vendor, info.device)
}

/// The process's GPU: chosen on first use, in a turn ([`setting_up`]),
/// from every adapter on every backend, and kept (a failed choice too:
/// no GPU is looked for again). Listing the adapters starts every GPU's
/// driver (an unused integrated one's too), so the choice does it once;
/// the adapters not kept are dropped then.
pub fn chosen() -> &'static Chosen {
    if let Some(chosen) = CHOSEN.get() {
        return chosen;
    }
    // The turn first: a thread waiting here for another's choice holds
    // none that the other could be waiting for.
    let _turn = setting_up();
    CHOSEN.get_or_init(choose)
}

/// Lists the adapters and chooses ([`chosen`]).
fn choose() -> Chosen {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let (adapter, sibling) = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut adapters = pollster::block_on(instance.enumerate_adapters(wgpu::Backends::all()));
        // As `request_adapter` sorts them for high performance: a discrete
        // GPU first, then an integrated one; the backends in their order
        // within each.
        adapters.sort_by_key(|a| match a.get_info().device_type {
            wgpu::DeviceType::DiscreteGpu => 0,
            wgpu::DeviceType::IntegratedGpu => 1,
            wgpu::DeviceType::Other => 2,
            wgpu::DeviceType::VirtualGpu => 3,
            wgpu::DeviceType::Cpu => 4,
        });
        let mut adapters = adapters.into_iter();
        let Some(best) = adapters.next() else {
            return (Err("no GPU: no adapter on any backend".to_string()), None);
        };
        let info = best.get_info();
        if info.device_type == wgpu::DeviceType::Cpu {
            return (Err(format!("{} is a software renderer", info.name)), None);
        }
        let sibling = (info.backend != wgpu::Backend::Dx12)
            .then(|| {
                adapters.find(|a| {
                    let other = a.get_info();
                    other.backend == wgpu::Backend::Dx12 && card(&other) == card(&info)
                })
            })
            .flatten();
        (Ok(best), sibling)
    }))
    .unwrap_or_else(|_| (Err("no GPU: looking for one panicked".into()), None));
    Chosen {
        instance,
        adapter,
        sibling,
    }
}

/// The adapter every device here comes from ([`chosen`]). Only in a turn
/// ([`setting_up`]), held until the device is set up.
fn adapter() -> Result<wgpu::Adapter, String> {
    debug_assert!(TURNS.get() > 0, "the GPU set up outside a turn");
    chosen().adapter.clone()
}

/// A device of its own on the chosen adapter ([`chosen`]), with all of
/// the adapter's limits, labelled `label`, and what `make` sets up on it,
/// all in one turn ([`setting_up`]) and inside error scopes (a failure on
/// the device is an `Err`); with the GPU's name. Errors name the GPU.
pub(crate) fn open<T>(
    label: &'static str,
    make: impl FnOnce(wgpu::Device, wgpu::Queue, &wgpu::Limits) -> Result<T, String>,
) -> Result<(T, String), String> {
    let _turn = setting_up();
    let adapter = adapter()?;
    let info = adapter.get_info();
    let limits = adapter.limits();
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some(label),
        required_limits: limits.clone(),
        ..Default::default()
    }))
    .map_err(|e| format!("{}: {e}", info.name))?;
    let probe = device.clone();
    let made = super::guarded(&probe, || make(device, queue, &limits))
        .map_err(|e| format!("{}: {e}", info.name))?;
    Ok((made, info.name))
}

/// The name of the GPU the separator would use, without setting it up;
/// `None` without a usable one. Once the GPU is chosen ([`chosen`]), it
/// waits for no setup.
pub fn gpu_name() -> Option<String> {
    chosen().adapter.as_ref().ok().map(|a| a.get_info().name)
}

/// The adapters' lines of the diagnostic report ([`crate::report`]): every
/// adapter `wgpu` sees on every backend with the limits the separator
/// needs (listed again, on demand only), the one chosen (or why none) and
/// its sibling. In a turn.
pub(crate) fn report() -> Vec<String> {
    let _turn = setting_up();
    let mut lines = Vec::new();
    let mb = |bytes: u64| format!("{} MB", bytes / (1024 * 1024));
    let chosen = chosen();
    let adapters = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        pollster::block_on(chosen.instance.enumerate_adapters(wgpu::Backends::all()))
    }));
    match adapters {
        Err(_) => lines.push("adapters: listing them panicked".into()),
        Ok(adapters) if adapters.is_empty() => {
            lines.push("adapters: none on any backend (Vulkan, Metal, DirectX 12, OpenGL)".into())
        }
        Ok(adapters) => {
            lines.push("adapters:".into());
            for a in adapters {
                let info = a.get_info();
                let limits = a.limits();
                lines.push(format!(
                    "  [{:?}] {} ({:?}), vendor 0x{:04x} device 0x{:04x}, driver {:?} {:?}",
                    info.backend,
                    info.name,
                    info.device_type,
                    info.vendor,
                    info.device,
                    info.driver,
                    info.driver_info
                ));
                lines.push(format!(
                    "      max buffer {}, max storage binding {}, workgroup storage {} KB, {} invocations per workgroup",
                    mb(limits.max_buffer_size),
                    mb(limits.max_storage_buffer_binding_size),
                    limits.max_compute_workgroup_storage_size / 1024,
                    limits.max_compute_invocations_per_workgroup
                ));
            }
        }
    }
    match &chosen.adapter {
        Ok(a) => {
            let info = a.get_info();
            lines.push(format!(
                "the separator would use: {} ({:?})",
                info.name, info.backend
            ));
        }
        Err(e) => lines.push(format!("the separator would use: none ({e})")),
    }
    if let Some(a) = &chosen.sibling {
        let info = a.get_info();
        lines.push(format!(
            "the same card kept on: {} ({:?})",
            info.name, info.backend
        ));
    }
    lines
}

/// The first `count` f32s of `staging` (a buffer to map for reading),
/// after a [`crate::pace::dispatch`] that wrote them (done, so this waits
/// for no GPU work: not a wait in its tally).
pub(crate) fn read_back(
    device: &wgpu::Device,
    staging: &wgpu::Buffer,
    count: usize,
) -> Result<Vec<f32>, String> {
    let slice = staging.slice(..(count * 4) as u64);
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
    let out = slice
        .get_mapped_range()
        .map_err(|e| e.to_string())?
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    staging.unmap();
    Ok(out)
}
