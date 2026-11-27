// Karaoke's backdrop showing a story scene: one full-screen triangle into
// the backdrop's texture, filling it. The picture is read round the
// camera's middle at its zoom (`scene_view`); a scene whole (`layers.x`
// 1) is moved by a gentle warp of its nearness (the shape's red, its heads
// held flat); one taken apart is drawn in layers, back to front, each read
// where its own slide puts it: the farthest from the plate (filled behind
// the others), the middle and the nearest from the picture by their
// shares (the shape's green and blue), each seated into what is behind it
// (`seat`: its edge taking a little of the light behind, lit on a bar's
// first beat), the nearest's shadow on what is behind it. Then the haze by
// nearness, the lift, the corners' shade, the light's tone and the
// theme's grade. Colours are linear light until
// they are shown (`tone`); `LINEAR` (set as the pipeline is made) says
// whether the texture takes them so.

struct Scene {
    // The level the picture is read blurred at for the light behind a
    // layer; .yzw unused.
    read: vec4<f32>,
    // The window's extent in the picture at zoom 1, the middle seen.
    view: vec4<f32>,
    // The zoom, unused, the slide.
    cam: vec4<f32>,
    // How far each layer slides, the farthest to the nearest; the haze.
    depth: vec4<f32>,
    // The light on a layer's edge: its colour, how much.
    light: vec4<f32>,
    // The haze's colour, the lift.
    fog: vec4<f32>,
    // The theme's grade: the tint, how much.
    grade: vec4<f32>,
    // How many layers (1: the warp); .yzw unused.
    layers: vec4<u32>,
};

@group(0) @binding(0) var<uniform> scene: Scene;
@group(0) @binding(1) var picture: texture_2d<f32>;
@group(0) @binding(2) var plate: texture_2d<f32>;
@group(0) @binding(3) var shape: texture_2d<f32>;
@group(0) @binding(4) var soft: sampler;

// The scene's light (its white a little under white, the stage over it).
const SCENE_LIGHT: f32 = 0.85;
// How dark the nearest's shadow is on what is behind it, and where it
// falls (the picture's shares).
const SHADOW: f32 = 0.35;
const SHADOW_OFF: vec2<f32> = vec2<f32>(0.004, 0.008);
@fragment
fn fs(o: Out) -> @location(0) vec4<f32> {
    let uv = scene_view(o.uv, scene.view.xy, scene.view.zw, scene.cam.x);
    let slide = scene.cam.zw;
    // The light behind a layer: the plate blurred; a layer's share blurred
    // over about the same (the shape is half the picture's size).
    let behind_l = scene.read.x;
    let edge_l = max(scene.read.x - 2.0, 0.0);
    var col = vec3<f32>(0.0);
    var near = 0.0;
    if scene.layers.x == 1u {
        near = textureSampleLevel(shape, soft, uv, 0.0).r;
        let at = shift(uv, near, slide);
        col = textureSample(picture, soft, at).rgb;
    } else {
        let uf = uv - slide * scene.depth.x;
        let um = uv - slide * scene.depth.y;
        let un = uv - slide * scene.depth.z;
        // The farthest: the plate, as far as the haze goes where a layer
        // in front was taken out of it.
        let sf = textureSampleLevel(shape, soft, uf, 0.0);
        col = textureSample(plate, soft, uf).rgb;
        near = sf.r * (1.0 - min(sf.g + sf.b, 1.0));
        if scene.layers.x == 3u {
            let sm = textureSampleLevel(shape, soft, um, 0.0);
            let bm = textureSampleLevel(shape, soft, um, edge_l).g;
            let behind = textureSampleLevel(plate, soft, um, behind_l).rgb;
            let cm = seat(textureSample(picture, soft, um).rgb, behind, bm, scene.light);
            col = mix(col, cm, sm.g);
            near = mix(near, sm.r, sm.g);
        }
        let sn = textureSampleLevel(shape, soft, un, 0.0);
        let shade = textureSampleLevel(shape, soft, un - SHADOW_OFF, edge_l).b;
        col = col * (1.0 - SHADOW * shade * (1.0 - sn.b));
        let bn = textureSampleLevel(shape, soft, un, edge_l).b;
        let behind = textureSampleLevel(plate, soft, un, behind_l).rgb;
        let cn = seat(textureSample(picture, soft, un).rgb, behind, bn, scene.light);
        col = mix(col, cn, sn.b);
        near = mix(near, sn.r, sn.b);
    }
    col = haze(col, scene.fog.rgb, scene.depth.w, near);
    col = col * (SCENE_LIGHT * (1.0 + scene.fog.w) * vignette(o.uv.x, o.uv.y));
    // Shown: toned, the theme's grade.
    return vec4<f32>(finish(col, scene.grade, o.pos.xy), 1.0);
}
