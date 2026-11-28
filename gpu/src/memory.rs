//! What the card's memory allows this process ([`Budget`]), read through
//! `wgpu`'s escape hatch to the backend under it (`wgpu` itself does not
//! say): `VK_EXT_memory_budget` on Vulkan (the largest device-local
//! heap's), and the device's recommended working set and what it has
//! allocated on Metal. Each is the operating system's
//! own figure: the budget already leaves out what the other programs hold,
//! and the usage is everything this process holds on the card (every
//! model's device and the window's).
//!
//! Going over the budget is never told: Windows pages video memory out to
//! system memory silently and everything crawls. So what is loaded is
//! planned from it before loading ([`budget`]), never found out by failing.

/// The card's memory as this process may use it, in bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Budget {
    /// The most this process should hold on the card.
    pub budget: u64,
    /// What it holds there now.
    pub usage: u64,
}

impl Budget {
    /// What is left of the budget.
    pub fn free(&self) -> u64 {
        self.budget.saturating_sub(self.usage)
    }
}

/// The budget of the process's GPU ([`crate::chosen`]) now; `None` without
/// a GPU, or where its backend does not tell (an old Vulkan driver without
/// `VK_EXT_memory_budget`).
pub fn budget() -> Option<Budget> {
    of(crate::chosen().adapter.as_ref().ok()?)
}

/// The budget of `adapter` now, through its backend; `None` where it does
/// not tell.
pub fn of(adapter: &wgpu::Adapter) -> Option<Budget> {
    match adapter.get_info().backend {
        #[cfg(not(target_vendor = "apple"))]
        wgpu::Backend::Vulkan => vulkan(adapter),
        #[cfg(target_vendor = "apple")]
        wgpu::Backend::Metal => metal(adapter),
        _ => None,
    }
}

/// A memory heap as Vulkan tells it: whether it is the device's own, its
/// size, and `VK_EXT_memory_budget`'s budget and usage of it.
#[cfg(not(target_vendor = "apple"))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Heap {
    local: bool,
    size: u64,
    budget: u64,
    usage: u64,
}

/// The budget of the largest of the device's own heaps (a card's video
/// memory, not the small window of it the processor can map, nor system
/// memory); `None` without one.
#[cfg(not(target_vendor = "apple"))]
fn largest_local(heaps: &[Heap]) -> Option<Budget> {
    let heap = heaps.iter().filter(|h| h.local).max_by_key(|h| h.size)?;
    Some(Budget {
        budget: heap.budget,
        usage: heap.usage,
    })
}

/// Whether a physical device has `VK_EXT_memory_budget`, asked of its
/// driver once per card (by its PCI ids).
#[cfg(not(target_vendor = "apple"))]
fn has_budget(card: (u32, u32), ask: impl FnOnce() -> bool) -> bool {
    static KNOWN: std::sync::Mutex<Vec<((u32, u32), bool)>> = std::sync::Mutex::new(Vec::new());
    let mut known = KNOWN
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if let Some(&(_, has)) = known.iter().find(|(c, _)| *c == card) {
        return has;
    }
    let has = ask();
    known.push((card, has));
    has
}

/// `VK_EXT_memory_budget`'s figures for the adapter's largest device-local
/// heap, where its driver has the extension and the instance is Vulkan 1.1
/// (`vkGetPhysicalDeviceMemoryProperties2` is an instance-level command of
/// 1.1).
#[cfg(not(target_vendor = "apple"))]
fn vulkan(adapter: &wgpu::Adapter) -> Option<Budget> {
    use ash::vk;
    // SAFETY: `as_hal`'s contract: the hal adapter is used only while the
    // guard lives (this call), no handle of it (the instance, the physical
    // device) is destroyed, kept or changed; the calls below only query.
    let hal = unsafe { adapter.as_hal::<wgpu::hal::api::Vulkan>() }?;
    let shared = hal.shared_instance();
    if shared.instance_api_version() < vk::API_VERSION_1_1 {
        return None;
    }
    let (instance, device) = (shared.raw_instance(), hal.raw_physical_device());
    let card = crate::card(&adapter.get_info());
    let has = has_budget(card, || {
        // SAFETY: the physical device is the instance's, alive through the
        // call (above).
        let extensions = unsafe { instance.enumerate_device_extension_properties(device) };
        extensions.is_ok_and(|all| {
            all.iter()
                .any(|e| e.extension_name_as_c_str() == Ok(ash::ext::memory_budget::NAME))
        })
    });
    if !has {
        return None;
    }
    let mut budget = vk::PhysicalDeviceMemoryBudgetPropertiesEXT::default();
    let mut memory = vk::PhysicalDeviceMemoryProperties2::default().push_next(&mut budget);
    // SAFETY: as above; a Vulkan 1.1 instance has the command, the device
    // has the extension whose struct is chained, and both structs live
    // through the call.
    unsafe { instance.get_physical_device_memory_properties2(device, &mut memory) };
    let base = memory.memory_properties;
    let heaps: Vec<Heap> = (0..base.memory_heap_count as usize)
        .map(|k| Heap {
            local: base.memory_heaps[k]
                .flags
                .contains(vk::MemoryHeapFlags::DEVICE_LOCAL),
            size: base.memory_heaps[k].size,
            budget: budget.heap_budget[k],
            usage: budget.heap_usage[k],
        })
        .collect();
    largest_local(&heaps)
}

/// The device's recommended working set (what it can hold without its
/// performance suffering) and what has been allocated on it. Metal gives
/// a process one `MTLDevice` per GPU (every device `wgpu` makes on the
/// adapter is that one), so what it has allocated is the process's usage.
#[cfg(target_vendor = "apple")]
fn metal(adapter: &wgpu::Adapter) -> Option<Budget> {
    use objc2_metal::MTLDevice;
    // SAFETY: `as_hal`'s contract: the hal adapter is used only while the
    // guard lives (this call), and its device is only queried.
    let hal = unsafe { adapter.as_hal::<wgpu::hal::api::Metal>() }?;
    let device = hal.raw_device();
    Some(Budget {
        budget: device.recommendedMaxWorkingSetSize(),
        usage: device.currentAllocatedSize() as u64,
    })
}

/// The memory's line of the diagnostic report ([`crate::report`]): the
/// budget and usage of the process's GPU.
pub(crate) fn report() -> Vec<String> {
    let mb = |bytes: u64| bytes / (1024 * 1024);
    let line = |adapter: &wgpu::Adapter| {
        let info = adapter.get_info();
        match of(adapter) {
            Some(b) => format!(
                "memory [{:?}] {}: budget {} MB, used {} MB, free {} MB",
                info.backend,
                info.name,
                mb(b.budget),
                mb(b.usage),
                mb(b.free())
            ),
            None => format!(
                "memory [{:?}] {}: its backend does not tell",
                info.backend, info.name
            ),
        }
    };
    match &crate::chosen().adapter {
        Ok(adapter) => vec![line(adapter)],
        Err(why) => vec![format!("memory: no GPU ({why})")],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The card's own heap is the largest device-local one: not the small
    /// window of it the processor maps (also device-local), nor system
    /// memory.
    #[cfg(not(target_vendor = "apple"))]
    #[test]
    fn the_budget_is_the_largest_device_local_heaps() {
        let heap = |local, size, budget, usage| Heap {
            local,
            size,
            budget,
            usage,
        };
        let heaps = [
            heap(true, 8 << 30, 7 << 30, 1 << 30),
            heap(false, 32 << 30, 16 << 30, 0),
            heap(true, 256 << 20, 200 << 20, 10 << 20),
        ];
        let b = largest_local(&heaps).unwrap();
        assert_eq!((b.budget, b.usage), (7 << 30, 1 << 30));
        assert_eq!(b.free(), 6 << 30);
        assert_eq!(largest_local(&heaps[1..2]), None);
    }

    /// On this machine's GPU (skipped without one, or where its backend
    /// does not tell): a budget, and a buffer held on a device counted in
    /// what is used (other tests hold and free theirs meanwhile, so only
    /// that much is sure).
    #[test]
    fn the_budget_is_read_and_counts_what_is_held() {
        let Some(first) = budget() else {
            eprintln!("no budget told here: skipped");
            return;
        };
        assert!(first.budget > 0, "{first:?}");
        let held = crate::device::open("jodel-gpu memory", |device, _, _| {
            Ok(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("jodel-gpu memory"),
                size: 256 << 20,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            }))
        });
        let Ok((buffer, _)) = held else {
            eprintln!("no device: skipped");
            return;
        };
        let during = budget().unwrap();
        let _turn = crate::setting_up();
        drop(buffer);
        assert!(during.usage >= 200 << 20, "{during:?}");
    }
}
