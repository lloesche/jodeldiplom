//! The GPU is chosen once a process ([`jodel_gpu::chosen`]): every probe
//! and model after that gets the same adapter, or the same failure where
//! there is no GPU. One test in a binary of its own, so that nothing has
//! chosen before it.

use std::sync::mpsc;
use std::thread;
use std::time::Duration;

#[test]
fn the_gpu_is_chosen_once() {
    // The first probe chooses, so it waits for the setup under way.
    let turn = jodel_gpu::setting_up();
    let (tx, rx) = mpsc::channel();
    let probe = thread::spawn(move || {
        let name = jodel_gpu::gpu_name();
        tx.send(()).unwrap();
        name
    });
    assert!(
        rx.recv_timeout(Duration::from_millis(300)).is_err(),
        "the first probe ran during another setup"
    );
    drop(turn);
    rx.recv_timeout(Duration::from_secs(60))
        .expect("the first probe never ran");
    let first = probe.join().unwrap();
    let chosen = jodel_gpu::chosen();

    // Once chosen, a probe waits for no setup: it reads the choice.
    let turn = jodel_gpu::setting_up();
    let probe = thread::spawn(jodel_gpu::gpu_name);
    let (tx, rx) = mpsc::channel();
    thread::spawn(move || tx.send(probe.join().unwrap()).unwrap());
    let second = rx
        .recv_timeout(Duration::from_secs(10))
        .expect("the second probe waited for another setup");
    drop(turn);
    assert_eq!(second, first);

    // A model's setup takes the same adapter, or fails as the choice did.
    let network = jodel_gpu::GpuNetwork::new();
    match (&chosen.adapter, &network) {
        (Ok(adapter), Ok(_)) => {
            let info = adapter.get_info();
            assert_eq!(first.as_deref(), Some(info.name.as_str()));
            eprintln!("chosen: {} ({:?})", info.name, info.backend);
        }
        (Ok(adapter), Err(e)) => {
            // A GPU too small for the network: not this test's concern.
            eprintln!("chosen: {}; the network: {e}", adapter.get_info().name);
            assert!(first.is_some());
        }
        (Err(why), Err(e)) => {
            assert_eq!(e, why, "the failure is the choice's, kept");
            assert!(first.is_none());
            eprintln!("no GPU: {why}");
        }
        (Err(why), Ok(_)) => panic!("a network set up without a GPU ({why})"),
    }
    assert!(std::ptr::eq(chosen, jodel_gpu::chosen()));

    the_choice_is_wgpus_high_performance_pick(chosen);
}

/// The choice is the adapter `wgpu` itself picks for high performance on
/// the same instance (where that is a software renderer, or none, there
/// is no choice).
fn the_choice_is_wgpus_high_performance_pick(chosen: &jodel_gpu::Chosen) {
    let picked = {
        let _turn = jodel_gpu::setting_up();
        pollster::block_on(
            chosen
                .instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    ..Default::default()
                }),
        )
        .map(|a| a.get_info())
    };
    let key = |info: &wgpu::AdapterInfo| (info.backend, info.vendor, info.device, info.device_type);
    match &chosen.adapter {
        Ok(adapter) => {
            let picked = picked.expect("wgpu picks none, but there is a choice");
            assert_eq!(key(&adapter.get_info()), key(&picked));
        }
        Err(why) => {
            if let Ok(picked) = picked {
                assert_eq!(
                    picked.device_type,
                    wgpu::DeviceType::Cpu,
                    "no choice ({why}), but wgpu picks {}",
                    picked.name
                );
            }
        }
    }
}
