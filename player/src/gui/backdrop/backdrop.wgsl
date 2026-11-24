// Karaoke mode's backdrop: a cover taken apart, lit by the music. One
// full-screen triangle a cover showing, into a texture half the window's
// size, over the ground (the blurred cover, `ground_fs`); each over what
// is under it by its share. Colours are linear light until they are shown
// (`tone`); `LINEAR` (set as the pipeline is made) says whether the
// texture takes them so.

struct Look {
    // The window's width and height, the drift's time, unused.
    size: vec4<f32>,
    // The drift (the cover's units), the breathing, this cover's share.
    cam: vec4<f32>,
    // The rim's pulse (drums), the vocals, the rest, the bass.
    music: vec4<f32>,
    // A section begun, the singer's glow, the rays, the lift.
    moments: vec4<f32>,
    // The window's shares to the cover's units: scale, then offset.
    fit: vec4<f32>,
    // The light behind the subject (window shares), its reach (a share of
    // the height), unused.
    light: vec4<f32>,
    // What the breathing is centred on (the cover's units), and where the
    // text over it is (its top and bottom, shares of the height).
    focus: vec4<f32>,
    // The palette's lights: the vivid, the next, the largest.
    accent: vec4<f32>,
    second: vec4<f32>,
    ground: vec4<f32>,
};

@group(0) @binding(0) var<uniform> look: Look;
@group(0) @binding(1) var plate: texture_2d<f32>;
@group(0) @binding(2) var subject: texture_2d<f32>;
@group(0) @binding(3) var depth: texture_2d<f32>;
@group(0) @binding(4) var rim: texture_2d<f32>;
@group(0) @binding(5) var soft: sampler;

struct Out {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs(@builtin(vertex_index) i: u32) -> Out {
    let p = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    var o: Out;
    o.pos = vec4<f32>(p * 2.0 - vec2<f32>(1.0), 0.0, 1.0);
    o.uv = vec2<f32>(p.x, 1.0 - p.y);
    return o;
}

// Diva's gold, as light.
const GOLD: vec3<f32> = vec3<f32>(1.0, 0.6, 0.16);

// The window's point `p` (shares) in the cover's units.
fn cover_at(p: vec2<f32>) -> vec2<f32> {
    return p * look.fit.xy + look.fit.zw;
}

fn hash(p: vec2<f32>) -> f32 {
    return fract(sin(dot(p, vec2<f32>(12.9898, 78.233))) * 43758.5453);
}

@fragment
fn fs(o: Out) -> @location(0) vec4<f32> {
    let p = o.uv;
    let t = look.size.z;
    let aspect = look.size.x / max(look.size.y, 1.0);
    let calm = calm_at(p.y, look.focus.zw);
    let free = 1.0 - calm;
    let base = cover_at(p);
    let cam = look.cam.xy;
    let breath = look.cam.z;
    let focus = look.focus.xy;
    let near = textureSampleLevel(depth, soft, base, 0.5).rg;
    let pulse = look.music.x;
    let vocals = look.music.y;
    let other = look.music.z;
    let section = look.moments.x;
    let glow = look.moments.y;

    // The plate: moved by its depth, softer under the text, its colour
    // drifting through the palette with the rest of the music, a broad
    // light sweeping over it; a little darker as the voice lights the
    // subject.
    let up = shift(base, near.g, cam, breath, focus);
    let plate_c = textureSampleLevel(plate, soft, up, 0.6 + 2.6 * calm).rgb;
    let w = drift(t);
    let tint = look.accent.rgb * w.x + look.second.rgb * w.y + look.ground.rgb * w.z;
    let lum = dot(plate_c, vec3<f32>(0.2126, 0.7152, 0.0722));
    var col = mix(plate_c, lum * tint * 1.3, 0.1 + 0.35 * other);
    let sweep_c = mix(look.second.rgb, vec3<f32>(1.0), 0.3);
    col = col + plate_c * sweep_c * sweep(p.x, p.y, t, aspect) * (0.08 + 0.9 * other);
    col = col * (1.0 - 0.2 * vocals);

    // The subject: nearer, so moved more; as soft as the plate under the
    // text; lit by the voice, the shine and a section, a halo round it in
    // the palette's light (golden as the singer shines), none of it where
    // the text is.
    let us = shift(base, near.r, cam, breath, focus);
    let s = textureSampleLevel(subject, soft, us, 3.2 * calm);
    let a = s.a;
    let lit = 1.0 + free * (0.5 * vocals + 0.4 * glow + 0.3 * section);
    col = col * (1.0 - a) + s.rgb * a * lit;
    // Light, not paint: the palette's colour towards white.
    let warm = mix(mix(look.accent.rgb, GOLD, 0.6 * glow), vec3<f32>(1.0), 0.45 - 0.2 * glow);
    let halo = textureSampleLevel(subject, soft, us, 4.5).a;
    col = col + warm * halo * (1.0 - a) * free * (0.6 * vocals + 0.5 * glow + 0.4 * section);

    // The rim: the drums' hits.
    let edge = textureSampleLevel(rim, soft, us, 0.0).r;
    let edge_glow = textureSampleLevel(rim, soft, us, 2.5).r;
    let rim_c = mix(look.accent.rgb, vec3<f32>(1.0), 0.15);
    let hits = rim_light(edge, pulse, us.x, us.y, t) + edge_glow * 0.25 * pulse;
    col = col + rim_c * hits * (1.0 - 0.6 * calm);

    // The rays from behind the subject: Amazing, Diva and the sections.
    let ray = look.moments.z;
    if ray > 0.002 {
        let d = (p - look.light.xy) * vec2<f32>(aspect, 1.0);
        col = col + warm * ray_fan(d.x, d.y, t, look.light.z) * ray * (1.0 - a) * free * 1.2;
    }

    col = settle(col * exposure(calm, look.moments.w) * vignette(p.x, p.y), calm);
    let c = tone(col, LINEAR) + vec3<f32>((hash(o.pos.xy) - 0.5) / 255.0);
    return vec4<f32>(c, look.cam.w);
}
