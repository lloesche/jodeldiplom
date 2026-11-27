//! The probe's maths and formatting, without a GPU.

use super::bounds::{held, judge_read, judge_write, Read, Wrote, BOUND};
use super::fast::{dot4_i8, dot4_u8, native, product, which_product, Product};
use super::half::{from_f16, to_f16, ulps16};
use super::lanes::{expected, Layout, Vector};
use super::maths::{breaks, function, judge, substitute, ulps, Edge, Op, Verdict};
use super::subgroups;
use super::sync::unhanded;
use super::text::{share, si, table, value};

#[test]
fn ulps_count_floats_between() {
    assert_eq!(ulps(1.0, 1.0), 0);
    assert_eq!(ulps(1.0, f32::from_bits(1.0f32.to_bits() + 3)), 3);
    assert_eq!(ulps(0.0, -0.0), 0);
    // Across zero: the smallest denormals either side are two apart.
    assert_eq!(ulps(f32::from_bits(1), -f32::from_bits(1)), 2);
    assert_eq!(ulps(f32::NAN, f32::NAN), 0);
    assert_eq!(ulps(f32::NAN, 1.0), u64::MAX);
}

#[test]
fn judging_names_the_difference() {
    assert_eq!(judge(f32::INFINITY, f32::INFINITY, 8), Verdict::Same);
    assert_eq!(judge(f32::NAN, f32::NAN, 8), Verdict::Same);
    assert_eq!(judge(1.0, f32::NAN, 8), Verdict::Differs);
    assert_eq!(judge(f32::INFINITY, 3.4e38, 8), Verdict::Differs);
    assert_eq!(judge(1e-40, 0.0, 8), Verdict::Flushed);
    assert_eq!(judge(1.0, 1.000_000_1, 8), Verdict::Close);
    assert_eq!(judge(1.0, 1.01, 8), Verdict::Differs);
    assert_eq!(judge(0.0, -0.0, 8), Verdict::Same);
    // A limit is met exactly, however much else is allowed.
    assert_eq!(judge(0.0, 1e-20, u64::MAX), Verdict::Differs);
    // WGSL's bound for exp(88) is 179 ulps: 100 off is close.
    let want = 88f32.exp();
    let off = f32::from_bits(want.to_bits() - 100);
    assert_eq!(judge(want, off, 179), Verdict::Close);
    assert_eq!(judge(want, off, 8), Verdict::Differs);
}

#[test]
fn only_a_missed_limit_or_bound_breaks_fast_maths() {
    let edge = |op: Op, a: f32, got: f32| {
        let want = op.reference(f64::from(a), 0.0, 0.0) as f32;
        Edge {
            op,
            inputs: [a, 0.0, 0.0],
            want,
            got,
            verdict: judge(want, got, 179),
        }
    };
    let want = 88f32.exp();
    // A conforming exp(88), 100 ulps off, breaks nothing.
    assert!(!breaks(&edge(
        Op::Exp,
        88.0,
        f32::from_bits(want.to_bits() - 100)
    )));
    // A flushed denormal breaks nothing either.
    assert!(!breaks(&edge(Op::Exp, -100.0, 0.0)));
    // Metal's tanh past 88, and an exp of -1e30 that is not 0, do.
    assert!(breaks(&edge(Op::Tanh, 88.0, f32::NAN)));
    assert!(breaks(&edge(Op::Exp, -1e30, 3.0)));
    assert!(breaks(&edge(Op::Exp, 1e30, 1.0)));
    // A finite result far outside WGSL's bound does too.
    assert!(breaks(&edge(Op::Exp, 88.0, 3.0)));
    // Only exp, exp2 and tanh are judged so.
    assert!(!breaks(&edge(Op::Log, 1e-40, f32::NEG_INFINITY)));
}

#[test]
fn references_are_ieee() {
    let r = |op: Op, a: f32, b: f32, c: f32| {
        op.reference(f64::from(a), f64::from(b), f64::from(c)) as f32
    };
    assert_eq!(r(Op::Exp, 1e30, 0.0, 0.0), f32::INFINITY);
    assert_eq!(r(Op::Exp, -1e30, 0.0, 0.0), 0.0);
    assert_eq!(r(Op::Tanh, -1e30, 0.0, 0.0), -1.0);
    assert_eq!(r(Op::ExpDiff, -1e30, -1e30, 0.0), 1.0);
    assert_eq!(r(Op::ExpHeld, -1e30, 0.0, 0.0), (-87.0f64).exp() as f32);
    assert_eq!(r(Op::Min, f32::NAN, 1.0, 0.0), 1.0);
    assert_eq!(r(Op::Clamp, f32::NAN, 0.0, 1.0), 0.0);
    assert_eq!(r(Op::SelectLess, f32::NAN, 2.0, 3.0), 2.0);
    assert_eq!(r(Op::IsNan, f32::NAN, 0.0, 0.0), 1.0);
    assert!(r(Op::Log, -1.0, 0.0, 0.0).is_nan());
    assert_eq!(r(Op::Log, 0.0, 0.0, 0.0), f32::NEG_INFINITY);
    assert!(r(Op::Mul, 1e-40, 1.0, 0.0).is_subnormal());
}

#[test]
fn held_functions_are_found_in_common_wgsl() {
    let common = include_str!("../common.wgsl");
    let exp = function(common, "exp_held").unwrap();
    assert!(
        exp.starts_with("fn exp_held(") && exp.ends_with('}'),
        "{exp}"
    );
    assert!(function(common, "tanh_held").unwrap().contains("tanh("));
    assert_eq!(function("fn a() {\n}\n", "b"), None);
}

#[test]
fn inputs_are_put_in_their_names_only() {
    let v = ["1e30".to_string(), "NaN".into(), "2".into()];
    assert_eq!(substitute("exp(a - b)", &v), "exp(1e30 - NaN)");
    assert_eq!(
        substitute("select(b, c, a < 1.0)", &v),
        "select(NaN, 2, 1e30 < 1.0)"
    );
    assert_eq!(substitute("tanh_held(a)", &v), "tanh_held(1e30)");
    assert_eq!(substitute("clamp(a, b, c)", &v), "clamp(1e30, NaN, 2)");
}

#[test]
fn f16_round_trips_and_rounds_to_even() {
    for h in [
        0u16, 1, 0x3c00, 0x7bff, 0x0400, 0x03ff, 0x8001, 0xc000, 0x7c00,
    ] {
        assert_eq!(to_f16(from_f16(h)), h, "{h:#x}");
    }
    assert_eq!(from_f16(to_f16(65504.0)), 65504.0);
    assert_eq!(from_f16(to_f16(65519.0)), 65504.0);
    assert_eq!(from_f16(to_f16(65520.0)), f32::INFINITY);
    assert_eq!(
        from_f16(to_f16(1.0 + 1.0 / 2048.0)),
        1.0,
        "a tie goes to even"
    );
    assert_eq!(from_f16(to_f16(6e-8)), 2f32.powi(-24));
    assert_eq!(
        from_f16(to_f16(2f32.powi(-25))),
        0.0,
        "a tie at the bottom goes to 0"
    );
    assert!(from_f16(to_f16(f32::NAN)).is_nan());
    assert_eq!(ulps16(1.0, from_f16(0x3c01)), 1);
}

#[test]
fn lanes_expected_are_the_writers_values() {
    // Adjacent: lane k of a workgroup's region is invocation k's.
    assert_eq!(expected(Vector::Vec4F32, Layout::Adjacent, 5), 6);
    // Strided vec4: vector 1, lane 2 is written by invocation 2 * 64 + 1.
    assert_eq!(
        expected(Vector::Vec4F32, Layout::Strided, 4 + 2),
        2 * 64 + 1 + 1
    );
    // The second workgroup starts at 256.
    assert_eq!(expected(Vector::Vec2F32, Layout::Strided, 256), 257);
    // Every invocation's value turns up exactly once.
    for vector in [Vector::Vec4F32, Vector::Vec2F32, Vector::Vec4U32] {
        let mut seen: Vec<u32> = (0..512)
            .map(|k| expected(vector, Layout::Strided, k))
            .collect();
        seen.sort_unstable();
        assert_eq!(seen, (1..=512).collect::<Vec<_>>());
    }
}

#[test]
fn out_of_bounds_reads_are_told_apart() {
    let v = |k: u32| held(k);
    assert_eq!(judge_read(false, false, 776, &[v(776)]), Read::Right);
    assert_eq!(judge_read(false, false, 777, &[0.0]), Read::Zero);
    assert_eq!(judge_read(false, false, 800, &[v(776)]), Read::Last);
    assert_eq!(judge_read(false, false, 800, &[v(800)]), Read::Real);
    assert_eq!(judge_read(false, true, 800, &[v(800)]), Read::Other(v(800)));
    let vec = |e: u32| (0..4).map(|w| v(e * 4 + w)).collect::<Vec<_>>();
    assert_eq!(judge_read(true, false, 194, &vec(193)), Read::Last);
    assert_eq!(judge_read(true, false, 194, &vec(194)), Read::Real);
    assert_eq!(
        judge_read(true, true, 194, &[v(776), 0.0, 0.0, 0.0]),
        Read::Real
    );
    assert_eq!(BOUND / 4, 194);
}

#[test]
fn out_of_bounds_writes_are_told_apart() {
    let words: Vec<f32> = (0..1024).map(held).collect();
    assert_eq!(judge_write(false, 777, &words), Wrote::Dropped);
    let mut w = words.clone();
    w[776] = -1.0;
    assert_eq!(judge_write(false, 777, &w), Wrote::Last);
    let mut w = words.clone();
    w[1000] = -1.0;
    assert_eq!(judge_write(false, 1000, &w), Wrote::Past);
    let mut w = words.clone();
    w[772..776].fill(-1.0);
    assert_eq!(judge_write(true, 194, &w), Wrote::Last);
    let mut w = words;
    w[3] = -1.0;
    assert_eq!(judge_write(true, 194, &w), Wrote::Elsewhere);
}

#[test]
fn packed_dots_on_the_processor() {
    assert_eq!(dot4_i8(0x8080_8080, 0x8080_8080), 4 * 128 * 128);
    assert_eq!(dot4_i8(0xffff_ffff, 0x0102_0304), -(1 + 2 + 3 + 4));
    assert_eq!(dot4_u8(0xffff_ffff, 0xffff_ffff), 4 * 255 * 255);
    assert!(native(3.0, 1.0));
    assert!(!native(1.2, 1.0));
}

#[test]
fn products_are_told_by_their_layout() {
    let a = [1.0, 2.0, 3.0, 4.0];
    let b = [0.0, 1.0, 1.0, 0.0];
    assert_eq!(product(&a, &b, 2), vec![2.0, 1.0, 4.0, 3.0]);
    assert_eq!(
        which_product(&a, &b, &product(&a, &b, 2), 2, 1e-3),
        Product::Rows
    );
    assert_eq!(
        which_product(&a, &b, &product(&b, &a, 2), 2, 1e-3),
        Product::Columns
    );
    assert_eq!(which_product(&a, &b, &[0.0; 4], 2, 1e-3), Product::Wrong);
}

#[test]
fn subgroup_records_are_checked() {
    // One workgroup of 256 in subgroups of 32, every record as it should
    // be, then one sum spoiled.
    let mut words = Vec::new();
    for i in 0..256u32 {
        let (id, first) = (i % 32, i - i % 32);
        let even = if id % 2 == 0 { 16 } else { 0 };
        let thirds = (0..32).filter(|k| k % 3 == 0).count() as u32;
        words.extend([
            32,
            id,
            first,
            32,
            496,
            id,
            first + (id + 1) % 32,
            first,
            thirds,
            even,
        ]);
    }
    let s = subgroups::check(&words, (32, 32));
    assert!(subgroups::correct(&s), "{:?}", s.checks);
    assert_eq!(s.sizes.iter().copied().collect::<Vec<_>>(), vec![32]);
    words[3] = 31;
    assert!(!subgroups::correct(&subgroups::check(&words, (32, 32))));
}

#[test]
fn atomics_hand_each_value_once() {
    assert_eq!(unhanded(&[2, 1, 3]), 0);
    assert_eq!(unhanded(&[2, 2, 0]), 2);
}

#[test]
fn the_report_lines_up() {
    let rows = vec![
        vec!["a".to_string(), "long one".into(), "x".into()],
        vec!["longer".to_string(), "b".into(), "y".into()],
    ];
    assert_eq!(
        table(&rows),
        vec!["a       long one  x", "longer  b         y"]
    );
    assert_eq!(si(1.5e12, "FLOP/s"), "1.50 TFLOP/s");
    assert_eq!(si(512e9, "B/s"), "512 GB/s");
    assert_eq!(si(25.0, "x"), "25.0 x");
    assert_eq!(value(f32::INFINITY), "inf");
    assert_eq!(value(-0.0), "-0");
    assert_eq!(value(f32::NAN), "NaN");
    assert_eq!(value(1.5), "1.5");
    assert_eq!(value(1e30), "1.00000e30");
    assert!(value(1e-40).ends_with("(denormal)"));
    assert_eq!(share(0, 10), "0 of 10");
    assert_eq!(share(5, 10), "5 of 10 (50.0 %)");
}
