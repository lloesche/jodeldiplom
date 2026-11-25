// Karaoke mode's stage on the GPU (`stage_gpu.rs`): its shapes, each a
// signed distance, and its text from the atlas. Its kinds come before it;
// the liquid's maths, the tubes, a streak's shine, the glory and the
// diploma's shapes after it (`stage_gpu::shader`).
struct Screen {
    size: vec2<f32>,
    srgb: f32,
    atlas: f32,
    time: f32,
    pad0: f32,
    pad1: f32,
    pad2: f32,
};

@group(0) @binding(0) var<uniform> screen: Screen;
@group(0) @binding(1) var atlas: texture_2d<f32>;
@group(0) @binding(2) var soft: sampler;
// The tubes' middle lines, point by point (`snake.rs`).
@group(0) @binding(3) var<storage, read> paths: array<vec4<f32>>;

fn path_point(i: u32) -> vec4<f32> {
    return paths[i];
}

struct In {
    @location(0) quad: vec4<f32>,
    @location(1) shape: vec4<f32>,
    @location(2) color: vec4<f32>,
    @location(3) color2: vec4<f32>,
    @location(4) params: vec4<f32>,
    @location(5) uv: vec4<f32>,
    @location(6) strain: vec4<f32>,
};

struct Out {
    @builtin(position) pos: vec4<f32>,
    @location(0) p: vec2<f32>,
    @location(1) @interpolate(flat) shape: vec4<f32>,
    @location(2) @interpolate(flat) color: vec4<f32>,
    @location(3) @interpolate(flat) color2: vec4<f32>,
    @location(4) @interpolate(flat) params: vec4<f32>,
    @location(5) @interpolate(flat) uv: vec4<f32>,
    @location(6) @interpolate(flat) quad: vec4<f32>,
    @location(7) @interpolate(flat) strain: vec4<f32>,
};

@vertex
fn vs(@builtin(vertex_index) v: u32, i: In) -> Out {
    var quad = i.quad;
    var o: Out;
    o.shape = i.shape;
    o.color = i.color;
    o.color2 = i.color2;
    o.params = i.params;
    // Text or a badge popping (round where, from when, how long), the
    // badge drawn as a pill.
    let k = u32(round(i.params.x));
    var pop = vec4<f32>(0.0);
    if k == K_TEXT && i.shape.w > 0.0 {
        pop = i.shape;
    }
    if k == K_BADGE {
        pop = vec4<f32>(i.shape.xy, i.params.yz);
        o.params = vec4<f32>(f32(K_PILL), 0.0, i.params.w, 0.0);
    }
    if pop.w > 0.0 {
        let s = label_pop(screen.time - pop.z, pop.w);
        quad = pop.xyxy + (quad - pop.xyxy) * s.x;
        o.color.a = o.color.a * s.y;
        o.color2.a = o.color2.a * s.y;
        if k == K_BADGE {
            o.shape = vec4<f32>(i.shape.xy, i.shape.zw * s.x);
        }
    }
    let corner = vec2<f32>(f32(v & 1u), f32((v >> 1u) & 1u));
    let p = mix(quad.xy, quad.zw, corner);
    // Text turned round its quad's middle (a stamp's words): drawn there,
    // read where it was.
    var at = p;
    if k == K_TEXT && i.params.z != 0.0 {
        at = turn_about(p, (quad.xy + quad.zw) * 0.5, i.params.z);
    }
    o.pos = vec4<f32>(at.x / screen.size.x * 2.0 - 1.0, 1.0 - at.y / screen.size.y * 2.0, 0.0, 1.0);
    o.p = p;
    o.uv = i.uv;
    o.quad = quad;
    o.strain = i.strain;
    return o;
}

// A colour as the frame takes it: linear where the frame is sRGB.
fn out_rgb(c: vec3<f32>) -> vec3<f32> {
    if screen.srgb > 0.5 {
        return pow(c, vec3<f32>(2.2));
    }
    return c;
}

// Over: `c` covering `a`; plus light `l` added.
fn over_plus(c: vec3<f32>, a: f32, l: vec3<f32>) -> vec4<f32> {
    return vec4<f32>(out_rgb(c) * a + out_rgb(l), a);
}

fn round_box(p: vec2<f32>, half: vec2<f32>, r: f32) -> f32 {
    let q = abs(p) - half + vec2<f32>(r);
    return length(max(q, vec2<f32>(0.0))) + min(max(q.x, q.y), 0.0) - r;
}

fn segment(p: vec2<f32>, a: vec2<f32>, b: vec2<f32>) -> f32 {
    let pa = p - a;
    let ba = b - a;
    let h = clamp(dot(pa, ba) / max(dot(ba, ba), 1e-4), 0.0, 1.0);
    return length(pa - ba * h);
}

// The glow of a shape `d` pixels outside it, over `size`.
fn glow(d: f32, size: f32) -> f32 {
    if size <= 0.0 {
        return 0.0;
    }
    let t = max(d, 0.0) / size;
    return exp(-t * t * 3.0) * (1.0 - smoothstep(0.85, 1.0, t));
}

// How a bubble strains at the time (bulging, wobbling as it nears full;
// overfilling, brighter, as it bursts): how much larger across and down,
// 1 while it shows (0 once it has torn: its drops and bits are drawn as
// particles), and its overfilling's flash.
fn bulged(o: Out) -> vec4<f32> {
    let s = strained(o.strain, screen.time);
    var flash = 0.0;
    if (u32(o.strain.w) & POPPING) != 0u {
        flash = clamp((screen.time - o.strain.y) / SWELL, 0.0, 1.0) * 0.3;
    }
    return vec4<f32>(s, flash);
}

// A liquid at `p`, `d` outside its shape, `u` along it (across, for a
// pill; along its middle line, for a tube) within `span` (its front's
// waves in), `down` its depth there (0 its top, 1 its bottom), `flash`
// its bubble's overfilling: its top light to its bottom, a shine along its
// top and (a bubble's, `shimmering`) the light moving through it.
fn shade_liquid(o: Out, p: vec2<f32>, d: f32, u: f32, span: vec2<f32>, down: f32, flash: f32, shimmering: bool) -> vec4<f32> {
    let cover = clamp(0.5 - d, 0.0, 1.0);
    let along = smoothstep(span.x - 0.5, span.x + 0.5, u) * (1.0 - smoothstep(span.y - 0.5, span.y + 0.5, u));
    let col = mix(o.color.rgb, o.color2.rgb, down) + vec3<f32>(flash);
    var shine = (1.0 - smoothstep(0.0, 0.35, down)) * 0.35;
    if shimmering {
        shine = shine + 0.3 * shimmer(p.x, p.y, screen.time);
    }
    let gs = max(o.params.y, 1.0);
    let soft = smoothstep(span.x - gs, span.x, u) * (1.0 - smoothstep(span.y, span.y + gs, u));
    let a = cover * along * o.color.a;
    let g = glow(d, o.params.y) * soft * o.color2.a * 0.45;
    return over_plus(min(col + vec3<f32>(shine), vec3<f32>(1.0)), a, o.color2.rgb * g);
}

// A skin of glass `d` outside it, `down` its depth there, `flash` its
// overfilling: its body lighter at its top, outlined (`params.z` wide) and
// glowing (`params.y` far) in `color2`, the glow behind it (a dark body
// stays dark, a see-through one lit).
fn shade_skin(o: Out, d: f32, down: f32, flash: f32) -> vec4<f32> {
    let cover = clamp(0.5 - d, 0.0, 1.0);
    let outline = o.params.z;
    let edge = clamp(0.5 - abs(d + outline * 0.5) + outline * 0.5, 0.0, 1.0) * cover;
    let body = mix(o.color.rgb * (1.15 - 0.3 * down), vec3<f32>(1.0), flash);
    let a_body = cover * o.color.a;
    let col = mix(body, o.color2.rgb, edge * o.color2.a / max(a_body + edge * o.color2.a, 1e-4));
    let a = max(a_body, edge * o.color2.a);
    let g = glow(d, o.params.y) * o.color2.a * 0.5 * (1.0 - a);
    return over_plus(col, a, o.color2.rgb * g);
}

// A pill, a fill, or (`bubble`) a picture's bubble's liquid (straining
// with it), at `p`.
fn pill(o: Out, p: vec2<f32>, k: u32, bubble: bool) -> vec4<f32> {
    var s = vec4<f32>(1.0, 1.0, 1.0, 0.0);
    if bubble {
        s = bulged(o);
        if s.z <= 0.0 {
            return vec4<f32>(0.0);
        }
    }
    let c = o.shape.xy;
    let half = o.shape.zw * s.xy;
    var r = o.params.w;
    if bubble {
        r = o.params.z;
    }
    if k == K_FILL || r <= 0.0 {
        r = min(half.x, half.y);
    }
    let d = round_box(p - c, half, r);
    let down = clamp((p.y - (c.y - half.y)) / max(2.0 * half.y, 1.0), 0.0, 1.0);
    if k == K_FILL || bubble {
        // Between its span (bulging with its bubble); its front (where it
        // is filling) a living surface.
        var span = vec2<f32>(c.x) + (o.uv.xy - vec2<f32>(c.x)) * s.x;
        if bubble && (u32(o.strain.w) & FRONT) != 0u {
            span.y = span.y + front((p.y - c.y) / max(half.y, 0.5), screen.time, amplitude(o.shape.w), o.uv.z);
        }
        return shade_liquid(o, p, d, p.x, span, down, s.w, bubble);
    }
    return shade_skin(o, d, down, s.w);
}

// A word's tube or its liquid at `p` (`snake.rs`): its middle line
// bulging a little with its strain (`PATH_GROW`), its thickness all of it;
// its liquid along the line, its front square to it.
fn tube(o: Out, p: vec2<f32>, k: u32) -> vec4<f32> {
    let s = bulged(o);
    if s.z <= 0.0 {
        return vec4<f32>(0.0);
    }
    let c = o.shape.xy;
    let sp = vec2<f32>(1.0) + (s.xy - vec2<f32>(1.0)) * PATH_GROW;
    let q = c + (p - c) / sp;
    let r0 = o.shape.z;
    let r = r0 * s.y;
    let reach = r0 * (1.0 + GROW) + o.params.y + 2.0;
    let n = tube_near(q, u32(o.shape.w), u32(o.params.w), reach);
    let d = n.x * sp.y - r;
    // Across it: -1 at its top side, 1 at its bottom.
    let v = clamp(n.z * sp.y / max(r, 0.5), -1.0, 1.0);
    let down = (v + 1.0) * 0.5;
    if k == K_TUBE_LIQUID {
        var span = o.uv.xy;
        if (u32(o.strain.w) & FRONT) != 0u {
            span.y = span.y + front(v, screen.time, amplitude(r0), o.uv.z);
        }
        return shade_liquid(o, p, d, n.y, span, down, s.w, true);
    }
    return shade_skin(o, d, down, s.w);
}

// A shape of this pipeline's group (`GROUP`: `stage_gpu::Group`; the
// other groups' code is left out where it is compiled).
@fragment
fn fs(o: Out) -> @location(0) vec4<f32> {
    let k = u32(round(o.params.x));
    let p = o.p;
    if GROUP == G_BUBBLE {
        if k == K_LIQUID {
            return pill(o, p, k, true);
        }
        return tube(o, p, k);
    }
    if GROUP == G_GLORY {
        return glory(o, p, k);
    }
    if GROUP == G_SHINE {
        return shine(o, p, k);
    }
    if GROUP == G_SEAL {
        return seal_shape(o, p, k);
    }
    if GROUP == G_PAPER {
        return paper_shape(o, p, k);
    }
    if k == K_PILL || k == K_FILL {
        return pill(o, p, k, false);
    }
    if k == K_SEGMENT {
        let d = segment(p, o.shape.xy, o.shape.zw) - o.params.y;
        let cover = clamp(0.5 - d, 0.0, 1.0);
        let g = glow(d, o.params.z) * o.color2.a;
        return over_plus(o.color.rgb, cover * o.color.a, o.color2.rgb * g);
    }
    if k == K_SPARK {
        // A spark: a core and two crossed rays, turned, all light.
        let s = sin(o.shape.w);
        let co = cos(o.shape.w);
        let q0 = p - o.shape.xy;
        let q = abs(vec2<f32>(co * q0.x - s * q0.y, s * q0.x + co * q0.y));
        let size = max(o.shape.z, 0.5);
        let core = exp(-dot(q, q) / (size * size * 0.05));
        let thin = max(size * 0.07, 0.8);
        let ray_x = exp(-q.y * q.y / (thin * thin)) * pow(clamp(1.0 - q.x / size, 0.0, 1.0), 1.6);
        let ray_y = exp(-q.x * q.x / (thin * thin)) * pow(clamp(1.0 - q.y / size, 0.0, 1.0), 1.6);
        let halo = exp(-dot(q, q) / (size * size * 0.18)) * 0.35;
        let light = (core + max(ray_x, ray_y) * 0.9 + halo) * o.color.a;
        return vec4<f32>(out_rgb(o.color.rgb) * light, 0.0);
    }
    if k == K_DISC {
        let d = length(p - o.shape.xy) - o.shape.z;
        let cover = clamp(0.5 - d, 0.0, 1.0);
        let g = glow(d, o.shape.w) * o.color2.a;
        return over_plus(o.color.rgb, cover * o.color.a, o.color2.rgb * g);
    }
    if k == K_ARROW {
        // An arrow pointing right: a rounded triangle, its tip at shape.xy.
        let len = o.shape.z;
        let q = p - (o.shape.xy - vec2<f32>(len * 0.5, 0.0));
        // The triangle's three edges, as distances (negative inside).
        let h = len * 0.5;
        let w = len * 0.3;
        let n1 = normalize(vec2<f32>(w, h));
        let e1 = dot(vec2<f32>(q.x, abs(q.y)) - vec2<f32>(h, 0.0), n1);
        let e2 = -q.x - h * 0.55;
        let d = max(e1, e2) - len * 0.06;
        let cover = clamp(0.5 - d, 0.0, 1.0);
        let inner = clamp(-d / (len * 0.25), 0.0, 1.0);
        let col = mix(o.color.rgb, vec3<f32>(1.0), inner * 0.55);
        let g = glow(d, o.params.y) * o.color2.a;
        return over_plus(col, cover * o.color.a, o.color2.rgb * g);
    }
    // Text: its coverage from the atlas, filled left of the split.
    let t = (p - o.quad.xy) / max(o.quad.zw - o.quad.xy, vec2<f32>(1.0));
    // Kept inside its own texels (their centres): never a column of the
    // text beside it in the atlas.
    let texel = clamp(mix(o.uv.xy, o.uv.zw, t), o.uv.xy + vec2<f32>(0.5), o.uv.zw - vec2<f32>(0.5));
    let uv = texel / screen.atlas;
    let a = textureSampleLevel(atlas, soft, uv, 0.0).r;
    let filled = smoothstep(o.params.y - 0.75, o.params.y + 0.75, p.x);
    var col = mix(o.color2.rgb, o.color.rgb, filled);
    if (u32(o.strain.w) & RAINBOW) != 0u {
        // In a streak's rainbow, running along it.
        col = text_tint(col, p.x, o.strain.xyz, screen.time);
    }
    let alpha = mix(o.color2.a, o.color.a, filled) * a;
    return over_plus(col, alpha, vec3<f32>(0.0));
}

// A streak's shine: its bands (within the lane, softly), a swirl, a
// rainbow edge; light, added.
fn shine(o: Out, p: vec2<f32>, k: u32) -> vec4<f32> {
    let s = o.params.yzw;
    if k == K_BANDS {
        let inside = lane_mask(p, o.uv.xy, o.uv.zw);
        let l = bands(p, o.shape.xy, o.shape.zw, s, screen.time);
        return vec4<f32>(out_rgb(l) * inside, 0.0);
    }
    if k == K_SWIRL {
        return vec4<f32>(out_rgb(swirl(p, o.shape.xy, o.shape.z, s, screen.time)), 0.0);
    }
    return vec4<f32>(out_rgb(rim_at(p, o.shape.xy, o.shape.zw, s, screen.time)), 0.0);
}
