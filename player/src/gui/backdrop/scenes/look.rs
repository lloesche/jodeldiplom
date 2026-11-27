//! The story scenes' shader's maths, each here twice, for the processor and
//! in WGSL ([`wgsl`]), held to each other in `crate::gui::shader_check`:
//! where a point of the window reads the picture ([`view`]), a layer seated
//! into what is behind it ([`seat`]) and the haze by depth ([`haze`]). The
//! warp and the layers' parallax read where the cover's layers do
//! ([`crate::gui::backdrop::look::shift`]), and the corners' shade is the
//! cover's ([`crate::gui::backdrop::look::vignette`]).

/// How much of the light behind it a layer's edge takes (where its blur is
/// none of it).
pub const WRAP: f32 = 0.35;

/// Where the window's point `p` (its shares) reads the picture (its
/// shares): round the `centre` seen, the window `extent` of the picture
/// at zoom 1 (its shares) made smaller by `zoom`.
pub fn view(p: [f32; 2], extent: [f32; 2], centre: [f32; 2], zoom: f32) -> [f32; 2] {
    [0, 1].map(|i| centre[i] + (p[i] - 0.5) * extent[i] / zoom.max(1e-3))
}

/// A layer's colour `c` seated into what is `behind` it (blurred): its edge
/// (where its share, `blurred` round, is less than all) takes [`WRAP`] of
/// the light behind and the `rim`'s light (its colour, then how much).
pub fn seat(c: [f32; 3], behind: [f32; 3], blurred: f32, rim: [f32; 4]) -> [f32; 3] {
    let edge = (1.0 - blurred).clamp(0.0, 1.0);
    [0, 1, 2].map(|k| c[k] + (behind[k] - c[k]) * (WRAP * edge) + rim[k] * (rim[3] * edge))
}

/// `c` hazed into `fog` by `amount` the farther back (`near` 0) it is.
pub fn haze(c: [f32; 3], fog: [f32; 3], amount: f32, near: f32) -> [f32; 3] {
    let k = amount * (1.0 - near.clamp(0.0, 1.0));
    [0, 1, 2].map(|i| c[i] + (fog[i] - c[i]) * k)
}

/// The functions above in WGSL, their constants with them.
pub fn wgsl() -> String {
    format!("const WRAP: f32 = {WRAP:?};\n{WGSL}")
}

const WGSL: &str = r"
fn scene_view(p: vec2<f32>, extent: vec2<f32>, centre: vec2<f32>, zoom: f32) -> vec2<f32> {
    return centre + (p - vec2<f32>(0.5)) * extent / max(zoom, 1e-3);
}

fn seat(c: vec3<f32>, behind: vec3<f32>, blurred: f32, rim: vec4<f32>) -> vec3<f32> {
    let edge = clamp(1.0 - blurred, 0.0, 1.0);
    return c + (behind - c) * (WRAP * edge) + rim.rgb * (rim.w * edge);
}

fn haze(c: vec3<f32>, fog: vec3<f32>, amount: f32, near: f32) -> vec3<f32> {
    let k = amount * (1.0 - clamp(near, 0.0, 1.0));
    return c + (fog - c) * k;
}
";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_window_reads_the_picture_round_what_is_seen() {
        // At zoom 1 the window's corners are the extent's.
        let e = [0.8, 1.0];
        assert_eq!(view([0.5, 0.5], e, [0.4, 0.5], 1.0), [0.4, 0.5]);
        let corner = view([0.0, 1.0], e, [0.5, 0.5], 1.0);
        assert!((corner[0] - 0.1).abs() < 1e-6 && (corner[1] - 1.0).abs() < 1e-6);
        // Zoomed in, less of it.
        let near = view([0.0, 0.0], e, [0.5, 0.5], 2.0);
        assert!((near[0] - 0.3).abs() < 1e-6 && (near[1] - 0.25).abs() < 1e-6);
    }

    #[test]
    fn a_layer_s_edge_takes_the_light_behind_it_and_its_inside_does_not() {
        let (c, behind) = ([1.0, 0.0, 0.0], [0.0, 0.0, 1.0]);
        let none = [0.0; 4];
        assert_eq!(seat(c, behind, 1.0, none), c);
        let edge = seat(c, behind, 0.5, none);
        assert!((edge[0] - (1.0 - WRAP / 2.0)).abs() < 1e-6 && (edge[2] - WRAP / 2.0).abs() < 1e-6);
        let lit = seat(c, behind, 0.0, [0.0, 1.0, 0.0, 0.5]);
        assert!((lit[1] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn the_far_is_hazed_and_the_near_is_not() {
        let (c, fog) = ([1.0, 1.0, 1.0], [0.0, 0.2, 0.4]);
        assert_eq!(haze(c, fog, 0.3, 1.0), c);
        let far = haze(c, fog, 0.3, 0.0);
        assert!((far[0] - 0.7).abs() < 1e-6 && (far[2] - 0.82).abs() < 1e-6);
    }
}
