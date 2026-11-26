// Karaoke mode's backdrop: a cover taken apart, lit by the music. One
// full-screen triangle a cover showing, into a texture the window's size,
// over the ground (the blurred cover the pixmap would have); each over
// what is under it by its share. The whole cover sits on a card in the
// middle, its edge fading into the same cover blurred heavily under it; at
// rest the card is the cover itself (`original`, the plate's again where it
// is not at hand), the plate showing only where the subject, moving,
// uncovers it. With a subject cut out, the subject stays sharp and what is
// round it is blurred (the farther back the more), the plate's blur where
// the cover's would carry the subject's colours; the subject pushes out
// with the beat round its middle, its rim lit on the bar's first, the
// spotlight rays fanning out from behind it.
// Colours are linear light until they are shown (`tone`); `LINEAR` (set
// as the pipeline is made) says whether the texture takes them so.

struct Look {
    // The window's width and height, the drift's time, unused.
    size: vec4<f32>,
    // The card: its middle, its half width and height (window shares).
    card: vec4<f32>,
    // The level a card smaller than its cover reads, how smoothly it is
    // read and how much softer still, 1 with a subject cut out.
    read: vec4<f32>,
    // The levels read blurred: under the card, the subject's shadow;
    // unused.
    levels: vec4<f32>,
    // The layers' drift (the cover's units), the subject's growing, this
    // cover's share.
    cam: vec4<f32>,
    // The vocals, the rest, the bass; unused.
    music: vec4<f32>,
    // A section begun, the singer's glow, the rays, the lift.
    moments: vec4<f32>,
    // The window's shares to the blur's cover units: scale, then offset.
    fill: vec4<f32>,
    // The subject's middle (window shares), the rays' reach (a share of
    // the height), unused.
    light: vec4<f32>,
    // What the subject grows round with the bass, its middle (the cover's
    // units).
    focus: vec4<f32>,
    // The palette's lights: the vivid, the next, the largest.
    accent: vec4<f32>,
    second: vec4<f32>,
    ground: vec4<f32>,
    // The beat: how far the subject is pushed out, its rim's light, the
    // rays' swell, a phrase's flare.
    pulse: vec4<f32>,
    // The subject's nearness (in focus), the level blurred the most;
    // unused; 1 where what is round the subject is blurred.
    defocus: vec4<f32>,
};

@group(0) @binding(0) var<uniform> look: Look;
@group(0) @binding(1) var plate: texture_2d<f32>;
@group(0) @binding(2) var subject: texture_2d<f32>;
@group(0) @binding(3) var depth: texture_2d<f32>;
@group(0) @binding(4) var soft: sampler;
@group(0) @binding(5) var original: texture_2d<f32>;
@group(0) @binding(6) var rim: texture_2d<f32>;

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
// The card's light (its white shows a little under white).
const CARD_LIGHT: f32 = 0.62;
// The blur under the card's light: dark.
const UNDER: f32 = 0.13;
// The rays' light over the card, and round it.
const RAYS_ON: f32 = 3.0;
const RAYS_ROUND: f32 = 4.0;
// The rim's light at its brightest.
const RIM_LIGHT: f32 = 0.7;
// How much more of the plate's blur than the cover's is read for each
// share of the subject's colours the cover's blur would carry.
const BLEED: f32 = 3.0;

fn luma(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
}

fn hash(p: vec2<f32>) -> f32 {
    return fract(sin(dot(p, vec2<f32>(12.9898, 78.233))) * 43758.5453);
}

// `t` on the card at `uv`: as it is where the card is about its size, its
// smaller level where the card is smaller, smoothly where much enlarged and
// softer still the more a small cover is.
fn card_at(t: texture_2d<f32>, uv: vec2<f32>) -> vec4<f32> {
    let cubic = look.read.y;
    var c = vec4<f32>(0.0);
    if cubic >= 1.0 {
        c = smooth_at(t, uv, 0u);
    } else {
        c = textureSampleLevel(t, soft, uv, look.read.x);
        if cubic > 0.0 {
            c = mix(c, smooth_at(t, uv, 0u), cubic);
        }
    }
    if look.read.z > 0.0 {
        c = mix(c, smooth_at(t, uv, 1u), look.read.z);
    }
    return c;
}

// `t` at `uv` blurred to the level `lv`: smoothly, a level between two read
// from both.
fn blurred(t: texture_2d<f32>, uv: vec2<f32>, lv: f32) -> vec3<f32> {
    let l0 = floor(lv);
    let f = lv - l0;
    let below = smooth_at(t, uv, u32(l0)).rgb;
    if f < 0.02 {
        return below;
    }
    return mix(below, smooth_at(t, uv, u32(l0) + 1u).rgb, f);
}

@fragment
fn fs(o: Out) -> @location(0) vec4<f32> {
    let p = o.uv;
    let t = look.size.z;
    let px = look.size.xy;
    let vocals = look.music.x;
    let other = look.music.y;
    let bass = look.music.z;
    let section = look.moments.x;
    let glow = look.moments.y;
    let cut = look.read.w;
    let blur = u32(look.levels.x);
    let shadow_l = u32(look.levels.y);

    // How far each pixel is outside the card, in pixels.
    let half = look.card.zw * px;
    let side = 2.0 * min(half.x, half.y);
    let to = (p - look.card.xy) * px;
    let d = rounded_box(to.x, to.y, half.x, half.y, RADIUS * side);
    let shown = card_edge(d, side);

    // The palette's light, drifting through its colours; the singer's
    // light, golden as they shine.
    let w = drift(t);
    let tint = look.accent.rgb * w.x + look.second.rgb * w.y + look.ground.rgb * w.z;
    let warm = mix(mix(look.accent.rgb, GOLD, 0.6 * glow), vec3<f32>(1.0), 0.35 - 0.15 * glow);

    // Under the card: the whole cover blurred heavily, filling the window,
    // dark, darker towards the corners; lit round the card by the palette
    // with the music, the card's shadow on it.
    var col = vec3<f32>(0.0);
    if shown < 1.0 {
        let fuv = p * look.fill.xy + look.fill.zw;
        let pb = smooth_at(plate, fuv, blur).rgb;
        let sb = smooth_at(subject, fuv, blur);
        let mixed = mix(pb, sb.rgb, sb.a);
        // A little more colourful than the cover: blurred, it greys.
        let blur = max(mix(vec3<f32>(luma(mixed)), mixed, 1.4), vec3<f32>(0.0));
        col = blur * UNDER * (1.0 + 0.35 * other + 0.25 * bass) * vignette(p.x, p.y);
        let around = falloff(d, 3.0 * REACH * side);
        col = col + tint * around * (0.012 + 0.05 * other + 0.03 * vocals + 0.06 * section);
        col = col * (1.0 - 0.6 * falloff(d - 0.01 * side, REACH * side));
    }

    // The card: the plate, the subject over it, their depth.
    var a = 0.0;
    if shown > 0.0 {
        let uv = (p - look.card.xy + look.card.zw) / (2.0 * look.card.zw);
        let near = textureSampleLevel(depth, soft, uv, 0.0).rg;
        let cam = look.cam.xy;
        let focus = look.focus.xy;
        let up = shift(uv, near.g, cam, 0.0, focus);
        // The subject, grown with the bass, pushed out by the beat round
        // its middle.
        let us = kicked(shift(uv, near.r, cam, look.cam.z, focus), look.focus.zw, look.pulse.x);
        let s = card_at(subject, us);
        a = s.a;
        var plate_c = vec3<f32>(0.0);
        if look.defocus.w > 0.5 {
            // Round the subject, blurred the more the farther back: the
            // cover's blur, the plate's where the cover's would carry the
            // subject's colours (no halo, no hole as it grows).
            let lv = max(defocus(near.g, look.defocus.x) * look.defocus.y, look.read.x);
            let bleed = textureSampleLevel(subject, soft, up, lv + 1.0).a;
            let k = clamp(BLEED * bleed, 0.0, 1.0);
            if k < 1.0 {
                plate_c = blurred(original, up, lv);
            }
            if k > 0.0 {
                plate_c = mix(plate_c, blurred(plate, up, lv), k);
            }
        } else {
            // Under the subject: the cover itself, the plate where the
            // subject moved off it.
            let rest = textureSampleLevel(subject, soft, up, look.read.x).a;
            let uncovered = clamp(4.0 * (rest - a), 0.0, 1.0);
            plate_c = mix(card_at(original, up).rgb, card_at(plate, up).rgb, uncovered);
        }
        // Lit where the cut is sure of it. Without a cut-out its nearest
        // parts stand for it.
        var presence = smoothstep(0.5, 0.95, a);
        if cut < 0.5 {
            presence = 0.8 * smoothstep(0.55, 0.95, near.r);
        }
        // The subject's shadow on the plate, deeper as the bass pops it out.
        let off = vec2<f32>(0.004, 0.009) * (0.5 + bass);
        let shade = smooth_at(subject, us - off, shadow_l).a;
        plate_c = plate_c * (1.0 - (0.2 + 0.4 * bass) * shade * (1.0 - a));
        // The plate takes the palette's colour with the rest of the music,
        // its light sweeping across.
        plate_c = mix(plate_c, luma(plate_c) * tint * 1.5, 0.1 + 0.3 * other);
        let sweep_c = mix(look.second.rgb, vec3<f32>(1.0), 0.4);
        let swept = sweep(uv.x, uv.y, t) * (0.12 + 1.1 * other);
        plate_c = plate_c + (plate_c + vec3<f32>(0.04)) * sweep_c * swept;
        var card_c = plate_c * (1.0 - a) + s.rgb * a;
        // The voice lights the subject; a phrase landing flares it.
        let lit = 0.55 * vocals + 0.45 * glow + 0.3 * section + 0.35 * look.pulse.w;
        card_c = card_c * (1.0 + presence * lit);
        // Its rim lit on the bar's first beat, as the rays' light.
        if look.pulse.y > 0.002 && a > 0.004 {
            let edge = textureSampleLevel(rim, soft, us, look.read.x).r;
            card_c = card_c + warm * (edge * a * look.pulse.y * RIM_LIGHT);
        }
        col = mix(col, card_c * CARD_LIGHT, shown);
    }

    // The spotlight rays from behind the subject, in front of what is round
    // it: as the tiers' rows and the sections hold them, swelling on the
    // bar's first beat.
    let ray = look.moments.z * look.pulse.z;
    if ray > 0.002 {
        let aspect = px.x / max(px.y, 1.0);
        let dl = (p - look.light.xy) * vec2<f32>(aspect, 1.0);
        let beams = ray_fan(dl.x, dl.y, t, look.light.z);
        // Gold from Amazing's strength (1) to Diva's (1.6).
        let gold = clamp((look.moments.z - 1.0) / 0.6, 0.0, 1.0);
        let tint = mix(warm, GOLD, gold);
        col = col + tint * beams * ray * (1.0 - a) * mix(RAYS_ROUND, RAYS_ON, shown);
    }

    col = col * (1.0 + look.moments.w);
    let c = tone(col, LINEAR) + vec3<f32>((hash(o.pos.xy) - 0.5) / 255.0);
    return vec4<f32>(c, look.cam.w);
}
