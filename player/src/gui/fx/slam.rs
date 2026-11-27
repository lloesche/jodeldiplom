//! The word slam (an impact title): a big word slamming down onto the
//! stage, in the lane
//! behind the playhead (what has been sung: [`super::At::Over`]'s room;
//! never over what is coming to be sung, the lyric band, nor the HUD over
//! the lane). Fired by a phrase
//! reaching a row's tier it shouts the tier's name ("DIVA!"), by the row's
//! rung of the multiplier the multiplier reached ("×32") ([`word`]): the
//! word comes from what fired it ([`super::By`]), any row, no tier named
//! here (fired by both in one frame, the phrase's: its name). It falls in
//! from close (large and faint, growing solid), lands in the middle of its
//! room with a squash and springs back ([`pose`]), the screen shaking with the
//! blow ([`shake`]), a ring of dust and sparkles thrown up along the lane
//! where it landed ([`dust`]), a shine sweeping over it; it rests, and
//! rises and fades a second or so on. Fired again while it shows, the new
//! word takes its place. Fired by a phrase it is the phrase's verdict, and
//! the meter's stays away ([`super::Spec::shouts`]).
//!
//! It lands where the sound fired with it lands, by the grid's one rule
//! ([`when`], [`super::Music::land`]): a rung's on the next beat with its
//! stinger, a phrase's on a bar's first beat with room before a sung line,
//! with the jackpot; falling the last moment before it. Where it lands at
//! once (the grid unsure, the beat too far), it appears landing, squashed,
//! on the frame its sound does.
//!
//! The word is lettered by the stage ([`super::Letters`]: its faces, its
//! atlas, its size fixed for the stage and its strength its scale), drawn
//! by the effect's own pipeline (`slam_shape`): its letters from the atlas
//! and a dark outline round them ([`cover`]: 9 taps of the atlas a pixel,
//! one where it is inside a letter), its face lit from the top, in the
//! tier's colour ([`ink`], the shader's reference held to it on the GPU:
//! [`REFERENCE`]): a row's strong colour its own, a pale one's (the top
//! row's near white) the rainbow running across the letters ([`colours`]).
//! Its glow is the light's bloom: a copy of the word goes
//! into the light ([`super::Letters::glow`]). Its look ([`Spec::params`],
//! `p`): `x` how much it glows (1 as made).

use jodel_rhythm::grid::When;

use super::helpers::{mix3, smoothstep};
use super::{At, By, Fx, Layer, Letters, Live, Out, Playing, Reference, Slot, Spec};
use crate::gui::big_text::{Face, COVER_PAD, LINE};
use crate::gui::particles::{Kind, Look, Spray, Sprite};
use crate::gui::shine::hue;
use crate::gui::sparks::{rand, TAU, WHITE};
use crate::gui::stage::gpu::Instance;
use crate::theme::Theme;
use crate::tiers::{Tier, TIERS};

/// The word: its quad (`quad`), its box (`shape`: left, top, right,
/// bottom), its place in the atlas (`uv`), its face's colour (`color`),
/// its glow's (`color2`) and its outline's (`strain`), `params` how much
/// it shows, where its shine is and (its copy in the light) how strongly
/// it glows.
pub const SLAM: f32 = Fx::Slam.kind(0);

/// The word slam.
pub const SPEC: Spec = Spec {
    name: "word slam",
    ident: "slam",
    at: &[Layer::Over],
    draw,
    kinds: &["slam"],
    wgsl: Some(wgsl),
    params: [1.0, 0.0, 0.0, 0.0],
    reference: Some(REFERENCE),
    shouts: true,
    ..Spec::NONE
};

/// How long it falls before it lands, at most, in seconds.
const DROP: f32 = 0.16;
/// The room after a bar's first beat a phrase's slam lands on, before a
/// sung line, in beats: as long as the jackpot fired with it sounds, so
/// they land on the same bar's first beat.
pub const BEATS: f64 = crate::sfx::conductor::JACKPOT_BEATS;
/// How large it is as it starts falling, in its sizes.
const FROM: f32 = 2.2;
/// How far it squashes as it lands: wider by most of this, lower by it.
const SQUASH: f32 = 0.3;
/// The widest it is squashed, a share of its width.
const SPREAD: f32 = 1.0 + SQUASH * 0.8;
/// How much it grows as it fades, and how far it rises then (in its
/// sizes).
const GROW: f32 = 0.12;
const LIFT: f32 = 0.25;
/// Its spring back: how many times a second it wobbles, how fast that
/// dies away (seconds).
const SPRING: f32 = 5.5;
const SETTLE: f32 = 0.09;
/// How long it rests after landing before fading, and how long it fades.
const HOLD: f32 = 0.85;
const FADE: f32 = 0.35;
/// How much it shows landing, and resting.
const LANDED: f32 = 0.95;
const REST: f32 = 0.85;
/// The least it shows to be drawn at all (falling in, faded out).
const SHOWS: f32 = 0.02;
/// Its size: at most this share of the stage's height (and as large as
/// its room holds it: [`place`]).
const SIZE: f32 = 0.085;
/// Its quad's margin round its box (its outline), a share of its size.
const MARGIN: f32 = 0.1;
/// How far the screen shakes as it lands, a share of the stage's height;
/// how fast that dies away; for how long it shakes at all (seconds).
const SHAKE: f32 = 0.007;
const SHAKE_DECAY: f32 = 0.07;
const SHAKE_FOR: f32 = 0.35;
/// Its outline's width, a share of its text's height in the atlas; the
/// taps round each pixel it takes.
const RIM: f32 = 0.05;
const TAPS: u32 = 8;
/// How strongly its copy in the light glows (its look's `x` times this).
const GLOW: f32 = 0.6;
/// When its shine starts sweeping over it after it lands, and how long the
/// sweep takes, in seconds.
const SHINE_FROM: f32 = 0.06;
const SHINE_FOR: f32 = 0.5;
/// How long its dust lasts, in seconds.
pub const DUST_LIFE: f32 = 0.9;
/// Its dust's puffs (its first quads) and sparkles (the next).
const PUFFS: u32 = 12;
const SPARKS: u32 = 14;
/// The dust's size, a share of the word's.
const DUST_SIZE: f32 = 0.12;
/// The gold its glow leans to.
const GOLD: [f32; 3] = [1.0, 0.76, 0.26];

fn draw(at: &At, live: &Live, slot: &mut Slot, out: &mut Out) {
    let &At::Over { area, room, .. } = at else {
        return;
    };
    let t = live.clock as f32;
    if let (true, Some(by)) = (live.fired, live.by) {
        let lands = live.music.land(live.clock, when(by, &live.music));
        slot.playing = Some(Playing {
            by,
            at: t,
            strength: live.strength,
            next: lands as f32,
            left: 1,
        });
    }
    let Some(mut p) = slot.playing else {
        return;
    };
    let u = t - p.next;
    let fall = (p.next - p.at).min(DROP);
    let (Some(pose), Some(text)) = (pose(u, fall), word(p.by)) else {
        if u > 0.0 {
            slot.playing = None;
        }
        return;
    };
    let s = shake(u, area.3);
    out.shake[0] += s[0] * p.strength;
    out.shake[1] += s[1] * p.strength;
    let landed = u >= 0.0 && p.left > 0;
    if landed {
        p.left = 0;
    }
    slot.playing = Some(p);
    if pose.alpha < SHOWS {
        return;
    }
    let ((x, y), px) = place(area, room);
    let widest = widest(room, px);
    if widest <= 0.0 {
        return;
    }
    let (fill, deep, glow, bow) = colours(p.by, live.theme);
    let seed = (p.next * 1000.0) as u32 ^ 0x51_A700;
    let k = p.strength.clamp(0.3, 1.0);
    out.letters.push(Letters {
        text,
        face: Face::SHOUT,
        px,
        widest,
        foot: (x, y - pose.lift * px),
        scale: (pose.scale.0 * k, pose.scale.1 * k),
        margin: px * MARGIN,
        shape: Instance {
            color: [fill[0], fill[1], fill[2], 1.0],
            color2: [glow[0], glow[1], glow[2], 1.0],
            params: [SLAM, pose.alpha, pose.shine, 0.0],
            strain: [deep[0], deep[1], deep[2], bow],
            ..Instance::default()
        },
        glow: GLOW,
        burst: landed.then(|| slam_dust((x, y), 0.0, px * DUST_SIZE, p.next, seed, glow)),
    });
}

/// Where it lands, fired `by` (to `music`): as the sound fired with it
/// lands ([`super::Music::land`]): a rung's on the next beat, with its
/// stinger; a phrase's on a bar's first beat with room for [`BEATS`] before
/// a sung line, with the jackpot.
pub fn when(by: By, music: &super::Music) -> When {
    match by {
        By::Rung(_) => When::Beat,
        By::Phrase(_) => When::Downbeat {
            length: BEATS * f64::from(music.beat_length()),
        },
    }
}

/// What it shouts, fired `by`: a phrase reaching a row, its tier's name
/// ("DIVA!"); a row's rung reached, its multiplier ("×32",
/// [`Tier::multiplier`]); none for a row there is not.
pub fn word(by: By) -> Option<String> {
    match by {
        By::Phrase(row) => TIERS
            .get(row)
            .map(|t| format!("{}!", t.name.to_uppercase())),
        By::Rung(row) => Tier::ALL.get(row).map(|t| format!("×{}", t.multiplier())),
    }
}

/// Its colours in `theme`, fired `by` a row: its face the row's tier's
/// warmed a little with gold, its outline that darkened, its glow leaning
/// to gold; and how much its face is the rainbow instead: all of it for a
/// pale row's (the top's, near white), none for a strong colour's.
fn colours(by: By, theme: &Theme) -> ([f32; 3], [f32; 3], [f32; 3], f32) {
    let (By::Phrase(row) | By::Rung(row)) = by;
    let c = Tier::ALL.get(row).map_or(GOLD, |&t| theme.color(t));
    let face = mix3(c, GOLD, 0.25);
    let most = c[0].max(c[1]).max(c[2]).max(1e-4);
    let pale = 1.0 - smoothstep(0.15, 0.45, (most - c[0].min(c[1]).min(c[2])) / most);
    (face, face.map(|v| v * 0.16), mix3(c, GOLD, 0.5), pale)
}

/// Where it lands in `room` (left, top, right, bottom) on a stage in
/// `area` (left, top, width, height): the foot of its box (the middle of
/// its bottom) and its size, fixed for the stage. As large as the room
/// holds it at its tallest (risen and grown as it fades, its outline
/// round it, [`high`]), at most [`SIZE`] of the stage's height; its
/// tallest in the middle of the room.
pub fn place(area: (f32, f32, f32, f32), room: [f32; 4]) -> ((f32, f32), f32) {
    let tall = room[3] - room[1];
    let px = ((tall - high(0.0)) / (high(1.0) - high(0.0)))
        .min(area.3 * SIZE)
        .floor()
        .max(1.0);
    let slack = (tall - high(px)).max(0.0);
    let foot = room[3] - slack / 2.0 - px * MARGIN * SPREAD;
    (((room[0] + room[2]) / 2.0, foot), px)
}

/// How tall the word `px` large is at its tallest, its outline round it:
/// its box (the atlas's line and its pad) grown as it fades, risen, and
/// its quad's margin either side.
pub fn high(px: f32) -> f32 {
    (px * LINE + 2.0 * COVER_PAD as f32 + 1.0) * (1.0 + GROW) + px * (LIFT + 2.0 * MARGIN * SPREAD)
}

/// The widest the word `px` large may be lettered in `room` (left, top,
/// right, bottom): as wide as keeps it, squashed its widest as it lands
/// with its outline round it, inside the room (none where it has none).
pub fn widest(room: [f32; 4], px: f32) -> f32 {
    let pad = 2.0 * COVER_PAD as f32 + 1.0;
    (room[2] - room[0] - 2.0 * px * MARGIN * SPREAD) / SPREAD - pad
}

/// How the word is `u` seconds after it lands (before: falling, for
/// `fall` seconds): its scale across and down about its foot, how far it
/// has risen (in its sizes), how much it shows, and where its shine is (0
/// to 1 sweeping over it; none outside that). None before it falls or
/// once it has faded.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Pose {
    /// Its scale.
    pub scale: (f32, f32),
    /// How far it has risen.
    pub lift: f32,
    /// How much it shows.
    pub alpha: f32,
    /// Where its shine is.
    pub shine: f32,
}

/// The word `u` seconds after it lands ([`Pose`]).
pub fn pose(u: f32, fall: f32) -> Option<Pose> {
    if u < -fall || u >= HOLD + FADE {
        return None;
    }
    if u < 0.0 {
        let x = 1.0 + u / fall;
        let s = 1.0 + (FROM - 1.0) * (1.0 - x) * (1.0 - x);
        return Some(Pose {
            scale: (s, s),
            lift: 0.0,
            alpha: LANDED * smoothstep(0.0, 0.6, x),
            shine: -1.0,
        });
    }
    let k = (-u / SETTLE).exp() * (TAU * SPRING * u).cos();
    let fade = smoothstep(HOLD, HOLD + FADE, u);
    let grow = 1.0 + GROW * fade;
    Some(Pose {
        scale: ((1.0 + (SPREAD - 1.0) * k) * grow, (1.0 - SQUASH * k) * grow),
        lift: LIFT * fade,
        alpha: (LANDED + (REST - LANDED) * smoothstep(0.1, 0.6, u)) * (1.0 - fade),
        shine: (u - SHINE_FROM) / SHINE_FOR,
    })
}

/// How far the screen is shaken `u` seconds after the word lands on a
/// stage `height` high (pixels across and down): a blow downward dying
/// away in a few hundredths of a second, rattling a little sideways; none
/// before it lands or once it has died away.
pub fn shake(u: f32, height: f32) -> [f32; 2] {
    if !(0.0..SHAKE_FOR).contains(&u) {
        return [0.0; 2];
    }
    let a = SHAKE * height * (-u / SHAKE_DECAY).exp();
    [a * 0.4 * (u * 97.0 + 0.6).sin(), a * (u * 71.0).cos()]
}

/// The dust thrown up as the word lands with its foot at `foot`, its
/// letters `half` pixels either side, at `start`, puffs `size` large, in
/// `rgb` (its glow's).
pub fn slam_dust(
    foot: (f32, f32),
    half: f32,
    size: f32,
    start: f32,
    seed: u32,
    rgb: [f32; 3],
) -> Spray {
    Spray {
        at: foot,
        half: (half, 0.0),
        start,
        drift: 0.0,
        size,
        seed,
        rgb,
        kind: Kind::SlamDust,
    }
}

/// Quad `q` of the dust `s`, `age` seconds on: the first [`PUFFS`] puffs
/// rolling out either way along the floor from under the letters, growing,
/// rising a little, fading; the next [`SPARKS`] sparkles thrown up and out
/// off the letters, falling back, twinkling, fading.
pub fn dust(s: &Spray, q: u32, age: f32) -> Option<Sprite> {
    let r = |j: u32| rand(s.seed, q, j);
    let u = age / DUST_LIFE;
    if q < PUFFS {
        let side = if q.is_multiple_of(2) { -1.0 } else { 1.0 };
        let w = 1.0 - u;
        let go = 1.0 - w * w * w;
        let x0 = s.at.0 + side * s.half.0 * (0.15 + 0.85 * r(0));
        let half = s.size * (0.8 + 0.6 * r(3)) * (0.6 + 0.9 * go);
        return Some(Sprite {
            at: (
                x0 + side * s.size * (2.0 + 4.0 * r(1)) * go,
                s.at.1 - s.size * (0.2 + r(2)) * go,
            ),
            half: (half, half),
            alpha: 0.5 * w * w,
            rgb: mix3(s.rgb, WHITE, 0.65),
            look: Look::Puff,
            turn: 0.0,
        });
    }
    if q >= PUFFS + SPARKS {
        return None;
    }
    let v = age / (0.45 + 0.4 * r(4));
    if v >= 1.0 {
        return None;
    }
    let a = -TAU / 4.0 + (r(1) - 0.5) * 2.6;
    let speed = s.size * (9.0 + 12.0 * r(2));
    let g = s.size * 28.0;
    let x0 = s.at.0 + (r(0) * 2.0 - 1.0) * s.half.0 * 0.9;
    let ray = s.size * (0.7 + 0.6 * r(3)) * (1.0 - 0.5 * v);
    Some(Sprite {
        at: (
            x0 + a.cos() * speed * age,
            s.at.1 + a.sin() * speed * age + 0.5 * g * age * age,
        ),
        half: (ray, ray),
        alpha: (1.0 - v) * (0.7 + 0.3 * (age * 30.0 + r(5) * TAU).sin()),
        rgb: mix3(s.rgb, WHITE, 0.5),
        look: Look::Sparkle,
        turn: r(6) * TAU + 4.0 * age,
    })
}

/// [`dust`] in WGSL, for the particles' shader.
pub fn dust_wgsl() -> String {
    format!(
        "\nconst DUST_LIFE: f32 = {DUST_LIFE:?};\nconst DUST_PUFFS: u32 = {PUFFS}u;\nconst DUST_SPARKS: u32 = {SPARKS}u;\n{DUST_WGSL}"
    )
}

const DUST_WGSL: &str = r"
fn slam_dust(at: vec4<f32>, rgb: vec3<f32>, size: f32, seed: u32, q: u32, age: f32) -> Sp {
    var o: Sp;
    let u = age / DUST_LIFE;
    if q < DUST_PUFFS {
        var side = 1.0;
        if q % 2u == 0u {
            side = -1.0;
        }
        let w = 1.0 - u;
        let go = 1.0 - w * w * w;
        let x0 = at.x + side * at.z * (0.15 + 0.85 * r(seed, q, 0u));
        let half = size * (0.8 + 0.6 * r(seed, q, 3u)) * (0.6 + 0.9 * go);
        o.at = vec2<f32>(
            x0 + side * size * (2.0 + 4.0 * r(seed, q, 1u)) * go,
            at.y - size * (0.2 + r(seed, q, 2u)) * go,
        );
        o.half = vec2<f32>(half);
        o.alpha = 0.5 * w * w;
        o.rgb = mix(rgb, vec3<f32>(1.0), 0.65);
        o.look = LOOK_PUFF;
        o.turn = 0.0;
        o.shows = 1u;
        return o;
    }
    if q >= DUST_PUFFS + DUST_SPARKS {
        return o;
    }
    let v = age / (0.45 + 0.4 * r(seed, q, 4u));
    if v >= 1.0 {
        return o;
    }
    let a = -TAU / 4.0 + (r(seed, q, 1u) - 0.5) * 2.6;
    let speed = size * (9.0 + 12.0 * r(seed, q, 2u));
    let g = size * 28.0;
    let x0 = at.x + (r(seed, q, 0u) * 2.0 - 1.0) * at.z * 0.9;
    let ray = size * (0.7 + 0.6 * r(seed, q, 3u)) * (1.0 - 0.5 * v);
    o.at = vec2<f32>(x0 + cos(a) * speed * age, at.y + sin(a) * speed * age + 0.5 * g * age * age);
    o.half = vec2<f32>(ray);
    o.alpha = (1.0 - v) * (0.7 + 0.3 * sin(age * 30.0 + r(seed, q, 5u) * TAU));
    o.rgb = mix(rgb, vec3<f32>(1.0), 0.5);
    o.look = LOOK_SPARKLE;
    o.turn = r(seed, q, 6u) * TAU + 4.0 * age;
    o.shows = 1u;
    return o;
}

";

/// How a puff of dust is drawn, in WGSL (its look's line:
/// [`crate::gui::particles::LOOKS`]): a soft round cloud, over.
pub const DRAW_PUFF: &str = r"
fn draw_puff(p: vec2<f32>, c: vec2<f32>, half: vec2<f32>, turn: f32, look: vec4<f32>) -> vec4<f32> {
    let q = (p - c) / max(half.x, 0.5);
    let a = exp(-dot(q, q) * 2.5) * look.a;
    return vec4<f32>(out_rgb(look.rgb) * a, a);
}
";

/// The word at a pixel: its letters' coverage there and the most of it
/// round the pixel as far as the outline goes (`c`: [`cover`]); where the
/// pixel is in its box (`t`, 0 to 1 across and down); how much it shows
/// and where its shine is (`shown`); its face's and outline's colours
/// (sRGB) and how much its face is the rainbow (`bow`, running across the
/// letters, moving on as the shine sweeps). Its colour and coverage,
/// straight: its face lit from its top, the shine sweeping over it, the
/// dark outline round it.
pub fn ink(
    c: [f32; 2],
    t: (f32, f32),
    (alpha, shine): (f32, f32),
    (fill, deep, bow): ([f32; 3], [f32; 3], f32),
) -> [f32; 4] {
    let (face, rim) = (c[0], c[1].max(c[0]));
    let fill = mix3(
        fill,
        hue(t.0 * 0.8 + t.1 * 0.25 - shine.clamp(0.0, 1.0) * 0.4),
        bow,
    );
    let top = mix3(fill, [1.0; 3], 0.35);
    let mut col = mix3(top, fill, smoothstep(0.1, 0.8, t.1));
    let s = t.0 + 0.4 * t.1 - (shine * 1.6 - 0.3);
    let sweeping = f32::from(u8::from((0.0..=1.0).contains(&shine)));
    let band = (-s * s / 0.006).exp() * sweeping;
    col = mix3(col, [1.0; 3], band * 0.85);
    col = mix3(deep, col, face / rim.max(1e-4));
    [col[0], col[1], col[2], rim * alpha]
}

/// The word's coverage at `at` (atlas pixels), as the shader reads it
/// (`slam_cover`): `texel(x, y)` the atlas's coverage at a texel (0 to
/// 1), read between texels as the GPU's linear filter does (each texel's
/// middle at its halves, the atlas's edge held); none read outside the
/// word's own texels `rect` (left, top, right, bottom: never another
/// word's beside it); and the most of it round `at` as far as `r` (its
/// outline: [`TAPS`] taps), all of it inside a letter.
pub fn cover(texel: &dyn Fn(i32, i32) -> f32, at: (f32, f32), rect: [f32; 4], r: f32) -> [f32; 2] {
    let tap = |(x, y): (f32, f32)| -> f32 {
        if x < rect[0] || y < rect[1] || x > rect[2] || y > rect[3] {
            return 0.0;
        }
        let (u, v) = (x - 0.5, y - 0.5);
        let (i, j) = (u.floor(), v.floor());
        let (fx, fy) = (u - i, v - j);
        let (i, j) = (i as i32, j as i32);
        let row = |j: i32| texel(i, j) * (1.0 - fx) + texel(i + 1, j) * fx;
        row(j) * (1.0 - fy) + row(j + 1) * fy
    };
    let c = tap(at);
    if c >= 1.0 {
        return [1.0; 2];
    }
    let rim = (0..TAPS).fold(c, |rim, i| {
        let a = i as f32 * TAU / TAPS as f32;
        rim.max(tap((at.0 + a.cos() * r, at.1 + a.sin() * r)))
    });
    [c, rim]
}

/// [`ink`] and [`cover`] in WGSL (`fx_slam_ink`, `slam_cover`), and the
/// word drawn with them (`slam_shape`; in the light, its glow).
pub fn wgsl() -> String {
    format!("const SLAM_RIM: f32 = {RIM:?};\n{}{SHADER}", cover_wgsl())
}

/// [`cover`] in WGSL: `slam_cover`, reading `atlas` with `soft` (as the
/// stage's shader binds them; the shaders' check binds its own).
pub fn cover_wgsl() -> String {
    format!("const SLAM_TAPS: u32 = {TAPS}u;\n{COVER}")
}

const COVER: &str = r"
// The word's coverage at `at` (atlas pixels) in an atlas `side` square,
// none outside its own texels `rect`.
fn slam_tap(at: vec2<f32>, rect: vec4<f32>, side: f32) -> f32 {
    if at.x < rect.x || at.y < rect.y || at.x > rect.z || at.y > rect.w {
        return 0.0;
    }
    return textureSampleLevel(atlas, soft, at / side, 0.0).r;
}

// Its coverage at `at`, and the most of it as far round as `r`: all of it
// inside a letter.
fn slam_cover(at: vec2<f32>, rect: vec4<f32>, r: f32, side: f32) -> vec2<f32> {
    let c = slam_tap(at, rect, side);
    if c >= 1.0 {
        return vec2<f32>(1.0);
    }
    var rim = c;
    for (var i = 0u; i < SLAM_TAPS; i++) {
        let a = f32(i) * 6.2831855 / f32(SLAM_TAPS);
        rim = max(rim, slam_tap(at + vec2<f32>(cos(a), sin(a)) * r, rect, side));
    }
    return vec2<f32>(c, rim);
}
";

const SHADER: &str = r"
fn slam_shape(o: Out, p: vec2<f32>, k: u32) -> vec4<f32> {
    let t = (p - o.shape.xy) / max(o.shape.zw - o.shape.xy, vec2<f32>(1.0));
    let span = o.uv.zw - o.uv.xy;
    let c = slam_cover(o.uv.xy + span * t, o.uv, span.y * SLAM_RIM, screen.atlas);
    if o.params.w > 0.0 {
        // Its copy in the light: the word and its outline lit, to bloom.
        let lit = c.y * o.params.y * o.params.w * looks[FX_SLAM].x;
        return vec4<f32>(out_rgb(o.color2.rgb) * lit, 0.0);
    }
    let ink = fx_slam_ink(c, t, o.params.yz, o.color.rgb, o.strain);
    return over_plus(ink.rgb, ink.a, vec3<f32>(0.0));
}

// The word's colour and coverage at a pixel (straight).
fn fx_slam_ink(c: vec2<f32>, t: vec2<f32>, shown: vec2<f32>, own: vec3<f32>, deep_bow: vec4<f32>) -> vec4<f32> {
    let face = c.x;
    let rim = max(c.y, c.x);
    let deep = deep_bow.rgb;
    let fill = mix(own, hue(t.x * 0.8 + t.y * 0.25 - clamp(shown.y, 0.0, 1.0) * 0.4), deep_bow.w);
    let top = mix(fill, vec3<f32>(1.0), 0.35);
    var col = mix(top, fill, smoothstep(0.1, 0.8, t.y));
    let s = t.x + 0.4 * t.y - (shown.y * 1.6 - 0.3);
    let sweeping = select(0.0, 1.0, shown.y >= 0.0 && shown.y <= 1.0);
    let band = exp(-s * s / 0.006) * sweeping;
    col = mix(col, vec3<f32>(1.0), band * 0.85);
    col = mix(deep, col, face / max(rim, 1e-4));
    return vec4<f32>(col, rim * shown.x);
}
";

/// The word's ink at coverages in and round its letters, places in its
/// box, as it shows and shines, in colours: the shader's `fx_slam_ink` as
/// [`ink`]. Its taps of the atlas ([`cover`]) are held to the shader's
/// apart, with an atlas of their own (`super::super::shader_check`).
pub const REFERENCE: Reference = Reference {
    samples: || {
        (0..3000)
            .flat_map(|k| {
                let f = k as f32;
                let fr = |m: f32| jodel_looks::math::fract(f * m);
                let cover = [0.0, 1.0, fr(0.613)][k % 3];
                let rim = [cover, 1.0, fr(0.377).max(cover)][(k / 3) % 3];
                [
                    [cover, rim, fr(0.733), fr(0.157) * 2.0 - 0.5],
                    [fr(0.417) * 1.4 - 0.2, fr(0.519) * 1.4 - 0.2, 0.0, 0.0],
                    [fr(0.123), fr(0.456), fr(0.789), 0.0],
                    [
                        fr(0.321) * 0.3,
                        fr(0.654) * 0.3,
                        fr(0.987) * 0.3,
                        [0.0, 1.0, fr(0.246)][(k / 9) % 3],
                    ],
                ]
            })
            .collect()
    },
    ins: 4,
    outs: 1,
    body:
        "
    let a = ins[i * 4u];
    outs[i] = fx_slam_ink(a.xy, ins[i * 4u + 1u].xy, a.zw, ins[i * 4u + 2u].xyz, ins[i * 4u + 3u]);",
    cpu: |s| {
        let rgb = |v: [f32; 4]| [v[0], v[1], v[2]];
        vec![ink(
            [s[0][0], s[0][1]],
            (s[1][0], s[1][1]),
            (s[0][2], s[0][3]),
            (rgb(s[2]), rgb(s[3]), s[3][3]),
        )]
    },
    near: (2e-3, 2e-3),
    shows: |o| o[0][3] > 0.1,
    least: 1000,
};

#[cfg(test)]
mod tests;
