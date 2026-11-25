//! The vision kernels through naga's validation and its Metal and HLSL
//! writers (on any machine).

use super::*;

#[test]
fn the_kernels_validate_and_write_for_metal_and_directx() {
    let entries: Vec<&str> = Kernel::ALL.iter().map(|k| k.entry()).collect();
    crate::engine::check_source(&source(), naga::valid::Capabilities::empty(), &entries);
}
