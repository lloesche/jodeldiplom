//! Setting up the GPU from several threads at once. Some drivers (Intel's
//! on Windows, through its control library) corrupt the heap when two
//! threads create or drop their instances, adapters or devices at the
//! same time, so every setup in this crate takes its turn
//! ([`jodel_gpu::setting_up`]).

use std::sync::mpsc;
use std::thread;
use std::time::Duration;

/// Probes of the GPU on several threads at once, as the player makes them
/// (each new player asks its name on a thread of its own, while the
/// stems and the beat network set theirs up): each in its turn. Without
/// the turns, on a machine with such a driver, this process does not
/// survive it under a debug heap (half its runs), and now and then not
/// without one.
#[test]
fn the_gpu_is_probed_from_several_threads_at_once() {
    let threads: Vec<_> = (0..4)
        .map(|_| {
            thread::spawn(|| {
                for _ in 0..2 {
                    let _ = jodel_gpu::gpu_name();
                }
            })
        })
        .collect();
    for t in threads {
        t.join().unwrap();
    }
}

/// A probe waits for the setup under way (here, the turn the test holds)
/// and goes on once it is over.
#[test]
fn a_probe_waits_for_the_setup_under_way() {
    let turn = jodel_gpu::setting_up();
    let (tx, rx) = mpsc::channel();
    let probe = thread::spawn(move || {
        let name = jodel_gpu::gpu_name();
        tx.send(()).unwrap();
        name
    });
    assert!(
        rx.recv_timeout(Duration::from_millis(300)).is_err(),
        "the probe ran during another setup"
    );
    drop(turn);
    rx.recv_timeout(Duration::from_secs(60))
        .expect("the probe never ran");
    probe.join().unwrap();
}

/// A setup inside a turn its own thread holds goes on (it takes none of
/// its own), and the turn is free again once the outer one is dropped.
#[test]
fn a_setup_inside_its_own_threads_turn_goes_on() {
    let turn = jodel_gpu::setting_up();
    let _ = jodel_gpu::gpu_name();
    drop(turn);
    let other = thread::spawn(|| drop(jodel_gpu::setting_up()));
    other.join().unwrap();
}
