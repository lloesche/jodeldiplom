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
}

/// The one backend the models and the window run on: Metal on Apple's
/// systems, Vulkan everywhere else (the only backends built).
const BACKEND: &str = if cfg!(target_vendor = "apple") {
    "Metal"
} else {
    "Vulkan"
};

/// What is said where [`BACKEND`] finds no graphics card: on Vulkan, that
/// its driver may not be installed (Metal is always there).
const NO_CARD: &str = if cfg!(target_vendor = "apple") {
    "no graphics card on Metal"
} else {
    "no graphics card on Vulkan (is its driver installed?)"
};

/// The GPU's choice, made once.
static CHOSEN: OnceLock<Chosen> = OnceLock::new();

/// The card an adapter is on: its PCI vendor and device ids.
pub fn card(info: &wgpu::AdapterInfo) -> (u32, u32) {
    (info.vendor, info.device)
}

/// The process's GPU: chosen on first use, in a turn ([`setting_up`]),
/// from every adapter on [`BACKEND`], and kept (a failed choice too:
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
    let adapter = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut adapters = pollster::block_on(instance.enumerate_adapters(wgpu::Backends::all()));
        // As `request_adapter` sorts them for high performance: a discrete
        // GPU first, then an integrated one.
        adapters.sort_by_key(|a| match a.get_info().device_type {
            wgpu::DeviceType::DiscreteGpu => 0,
            wgpu::DeviceType::IntegratedGpu => 1,
            wgpu::DeviceType::Other => 2,
            wgpu::DeviceType::VirtualGpu => 3,
            wgpu::DeviceType::Cpu => 4,
        });
        let Some(best) = adapters.into_iter().next() else {
            return Err(NO_CARD.to_owned());
        };
        let info = best.get_info();
        if info.device_type == wgpu::DeviceType::Cpu {
            return Err(format!("only {}, a software renderer", info.name));
        }
        Ok(best)
    }))
    .unwrap_or_else(|_| Err("looking for a graphics card panicked".into()));
    Chosen { instance, adapter }
}

/// The adapter every device here comes from ([`chosen`]). Only in a turn
/// ([`setting_up`]), held until the device is set up.
fn adapter() -> Result<wgpu::Adapter, String> {
    debug_assert!(TURNS.get() > 0, "the GPU set up outside a turn");
    chosen().adapter.clone()
}

/// A device of its own on the chosen adapter ([`chosen`]), with all of
/// the adapter's limits, its memory taken in small blocks (each model has
/// a device of its own: the allocator's larger blocks would leave up to a
/// quarter of a gigabyte a device unused, and what a model takes on the
/// card would no longer be what its buffers take; as measured, setups take
/// a little longer, and runs as long), labelled `label`, and what `make`
/// sets up on it,
/// all in one turn ([`setting_up`]) and inside error scopes (a failure on
/// the device is an `Err`); with the GPU's name. Errors name the GPU.
pub(crate) fn open<T>(
    label: &'static str,
    make: impl FnOnce(wgpu::Device, wgpu::Queue, &wgpu::Limits) -> Result<T, String>,
) -> Result<(T, String), String> {
    open_with(
        label,
        |_| {
            (
                wgpu::Features::empty(),
                wgpu::ExperimentalFeatures::disabled(),
                (),
            )
        },
        |device, queue, limits, ()| make(device, queue, limits),
    )
}

/// [`open`], with the features `wanted` picks from the adapter's too, and
/// wgpu's experimental features where it allows them (the caller's to
/// answer for: the picture models' cooperative matrices); what else it
/// chooses from the adapter is handed to `make`.
pub(crate) fn open_with<T, W>(
    label: &'static str,
    wanted: impl FnOnce(&wgpu::Adapter) -> (wgpu::Features, wgpu::ExperimentalFeatures, W),
    make: impl FnOnce(wgpu::Device, wgpu::Queue, &wgpu::Limits, W) -> Result<T, String>,
) -> Result<(T, String), String> {
    open_on(label, wgpu::MemoryHints::MemoryUsage, wanted, make)
}

/// [`open_with`], its memory taken as `memory` says.
pub(crate) fn open_on<T, W>(
    label: &'static str,
    memory: wgpu::MemoryHints,
    wanted: impl FnOnce(&wgpu::Adapter) -> (wgpu::Features, wgpu::ExperimentalFeatures, W),
    make: impl FnOnce(wgpu::Device, wgpu::Queue, &wgpu::Limits, W) -> Result<T, String>,
) -> Result<(T, String), String> {
    let _turn = setting_up();
    let adapter = adapter()?;
    let info = adapter.get_info();
    let limits = adapter.limits();
    let (wanted, experimental_features, chosen) = wanted(&adapter);
    let features = adapter.features() & (wgpu::Features::TIMESTAMP_QUERY | wanted);
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some(label),
        // Stamps for profiles, where the adapter has them.
        required_features: features,
        required_limits: limits.clone(),
        memory_hints: memory,
        experimental_features,
        ..Default::default()
    }))
    .map_err(|e| format!("{}: {e}", info.name))?;
    let probe = device.clone();
    let made = crate::engine::guarded(&probe, || make(device, queue, &limits, chosen))
        .map_err(|e| format!("{}: {e}", info.name))?;
    Ok((made, info.name))
}

/// Whether the adapter `info` tells of has subgroups of 32 invocations,
/// always. Through Metal the name is all there is to go by: wgpu says
/// subgroups of 4 to 64 for every Metal adapter and gives no vendor or
/// device (0), on Apple's GPUs and on the Intel and AMD GPUs of older
/// Macs, whose SIMD-groups need not be 32 wide. Apple's are 32; their names
/// start with "Apple", which keeps the others out. The kernels written for
/// it also take a workgroup's invocations 32s to 32s + 31 (by local index,
/// in one dimension) to be one subgroup: Metal's SIMD-groups are so; on
/// Vulkan that is what the drivers do, not a guarantee without full
/// subgroups asked for.
pub(crate) fn subgroups_of_32(info: &wgpu::AdapterInfo) -> bool {
    (info.subgroup_min_size, info.subgroup_max_size) == (32, 32)
        || (info.backend == wgpu::Backend::Metal && info.name.starts_with("Apple"))
}

/// The most bytes a buffer of the process's GPU ([`chosen`]) is bound
/// with (`wgpu`'s storage binding, within its largest buffer); without a
/// usable GPU, why there is none.
pub(crate) fn binding() -> Result<u64, String> {
    let limits = chosen().adapter.as_ref().map_err(Clone::clone)?.limits();
    Ok(limits
        .max_storage_buffer_binding_size
        .min(limits.max_buffer_size))
}

/// The most bytes the Vulkan driver of `adapter` binds a storage buffer
/// with (`maxStorageBufferRange`), for the report beside what `wgpu`
/// allows (it keeps every backend within 2 GiB); `None` on Metal.
#[cfg(not(target_vendor = "apple"))]
fn driver_binding(adapter: &wgpu::Adapter) -> Option<u64> {
    // SAFETY: `as_hal`'s contract: the hal adapter is used only while the
    // guard lives (this call), and its physical device is only queried.
    let hal = unsafe { adapter.as_hal::<wgpu::hal::api::Vulkan>() }?;
    let instance = hal.shared_instance().raw_instance();
    // SAFETY: the physical device is the instance's, alive through the
    // call.
    let properties = unsafe { instance.get_physical_device_properties(hal.raw_physical_device()) };
    Some(u64::from(properties.limits.max_storage_buffer_range))
}

/// [`driver_binding`] on Metal: none.
#[cfg(target_vendor = "apple")]
fn driver_binding(_: &wgpu::Adapter) -> Option<u64> {
    None
}

/// The name of the GPU the separator would use, without setting it up;
/// without a usable one, why there is none. Once the GPU is chosen
/// ([`chosen`]), it waits for no setup.
pub fn gpu_name() -> Result<String, String> {
    match &chosen().adapter {
        Ok(adapter) => Ok(adapter.get_info().name),
        Err(why) => Err(why.clone()),
    }
}

/// The adapters' lines of the diagnostic report ([`crate::report`]): every
/// adapter `wgpu` sees with the limits the networks need (listed again, on
/// demand only; on Vulkan with the driver's own storage binding beside
/// `wgpu`'s), and the one chosen (or why none). In a turn.
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
        Ok(adapters) if adapters.is_empty() => lines.push(format!("adapters: none on {BACKEND}")),
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
                    "      max buffer {}, max storage binding {}{}, workgroup storage {} KB, {} invocations per workgroup",
                    mb(limits.max_buffer_size),
                    mb(limits.max_storage_buffer_binding_size),
                    driver_binding(&a)
                        .map_or_else(String::new, |b| format!(" (the driver's {})", mb(b))),
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
    lines
}

/// The first `count` f32s of `staging` (a buffer to map for reading),
/// after the run that wrote them is done ([`crate::pace::dispatch`], so
/// not a wait in its tally); everything else sent to the device is waited
/// for too.
pub(crate) fn read_back(
    device: &wgpu::Device,
    staging: &wgpu::Buffer,
    count: usize,
) -> Result<Vec<f32>, String> {
    read_when(device, staging, count, wgpu::PollType::wait_indefinitely())
}

/// [`read_back`] once the submission `written` that wrote them is done
/// (done already: not a wait in a tally), leaving the work sent after it
/// running: a decoding model's next run, made while this one ran.
/// `staging` must not be used by work sent after `written`: its mapping
/// would wait for that work, which nothing here waits for, and this would
/// never return.
pub(crate) fn read_after(
    device: &wgpu::Device,
    staging: &wgpu::Buffer,
    count: usize,
    written: wgpu::SubmissionIndex,
) -> Result<Vec<f32>, String> {
    let poll = wgpu::PollType::Wait {
        submission_index: Some(written),
        timeout: None,
    };
    read_when(device, staging, count, poll)
}

/// [`read_back`]'s values, the device polled as `poll` says until they
/// can be read.
fn read_when(
    device: &wgpu::Device,
    staging: &wgpu::Buffer,
    count: usize,
    poll: wgpu::PollType,
) -> Result<Vec<f32>, String> {
    let slice = staging.slice(..(count * 4) as u64);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    device.poll(poll).map_err(|e| e.to_string())?;
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
