//! The vision kernels through naga's validation and its Metal writer
//! (on any machine).

use super::*;

#[test]
fn the_kernels_validate_and_write_for_metal() {
    let entries: Vec<&str> = Kernel::ALL.iter().map(|k| k.entry()).collect();
    crate::engine::check_source(&source(), naga::valid::Capabilities::empty(), &entries);
}
