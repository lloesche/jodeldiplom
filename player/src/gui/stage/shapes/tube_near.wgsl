// A tube's nearest point (`snake.rs`: `nearest` is its reference on the
// processor), over the points `path_point` gives: the stage's path
// buffer, or the shader check's samples.
fn tube_near(p: vec2<f32>, first: u32, n: u32, reach: f32) -> vec3<f32> {
    var best = vec3<f32>(1e9, p.x, 1e9);
    if n < 2u {
        return best;
    }
    // The first point not left of the reach (a binary search), and the
    // piece before it.
    var lo = 0u;
    var hi = n;
    loop {
        if lo >= hi {
            break;
        }
        let m = (lo + hi) / 2u;
        if path_point(first + m).x < p.x - reach {
            lo = m + 1u;
        } else {
            hi = m;
        }
    }
    var i = max(lo, 1u) - 1u;
    loop {
        if i + 1u >= n {
            break;
        }
        let a = path_point(first + i).xy;
        if a.x > p.x + reach {
            break;
        }
        let b = path_point(first + i + 1u).xy;
        let ba = b - a;
        let pa = p - a;
        let h = dot(pa, ba) / max(dot(ba, ba), 1e-6);
        let q = pa - ba * clamp(h, 0.0, 1.0);
        let d = length(q);
        if d < best.x {
            var lo_h = 0.0;
            var hi_h = 1.0;
            if i == 0u {
                lo_h = -3.4e38;
            }
            if i + 2u == n {
                hi_h = 3.4e38;
            }
            let along = a.x + ba.x * clamp(h, lo_h, hi_h);
            var side = d;
            if ba.x * pa.y - ba.y * pa.x < 0.0 {
                side = -d;
            }
            best = vec3<f32>(d, along, side);
        }
        i = i + 1u;
    }
    return best;
}
