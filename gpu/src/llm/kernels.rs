//! The decoder's kernels as a program ([`Kernel`], [`source`]) and the
//! interface they share with the plan: the kinds of matrix each matrix
//! multiply takes, the flags, modes and shapes both sides know, and each
//! kernel's numbers by name ([`NUMBERS`]: the plan writes them by name,
//! [`named`], and the kernels read them by the same names, `prep_rows()`).

use jodel_tensors::gguf::Kind;

use crate::engine::{self, params, Access, After, Kernels, MUL_NUMBERS};

/// The kernels, the kinds of matrix each matrix multiply takes on each
/// path ([`Q8_KINDS`], [`PLAIN_KINDS`]), their entry points and the
/// kinds' names in the source ([`suffix`]), written from one list: every
/// kind and its name, then the kinds of each path.
macro_rules! kernels {
    (
        names: [$($k:ident $name:literal),*],
        q8: [$($q:ident),*],
        plain: [$($f:ident),*]
    ) => {
        /// The kinds a matrix multiply takes at 8 bits (the q8 path) and as
        /// f32s.
        pub(super) const Q8_KINDS: &[Kind] = &[$(Kind::$q),*];
        pub(super) const PLAIN_KINDS: &[Kind] = &[$(Kind::$f),*];

        /// The name of a kind in the kernels' source (its readers in
        /// `blocks.wgsl`, its matrix multiplies).
        fn suffix(kind: Kind) -> &'static str {
            match kind {
                $(Kind::$k => $name,)*
            }
        }

        /// The kernels: the matrix multiplies one for each kind of matrix
        /// they take.
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
        pub(super) enum Kernel {
            Embed,
            Quant,
            Residual,
            PerLayer,
            Argmax,
            GemvQ8(Kind),
            GemvF32(Kind),
            GemmQ8(Kind),
            GemmF32(Kind),
            Prep,
            Attend,
            Join,
            Flash,
        }

        impl Kernels for Kernel {
            const ALL: &'static [Kernel] = &[
                Kernel::Embed,
                Kernel::Quant,
                Kernel::Residual,
                Kernel::PerLayer,
                Kernel::Argmax,
                $(Kernel::GemvQ8(Kind::$q), Kernel::GemmQ8(Kind::$q),)*
                $(Kernel::GemvF32(Kind::$f), Kernel::GemmF32(Kind::$f),)*
                Kernel::Prep,
                Kernel::Attend,
                Kernel::Join,
                Kernel::Flash,
            ];
            const PARAM_BYTES: u64 = 64;
            /// After the parameters: the weights (read), the ints.
            const EXTRA: &'static [Access] = &[Access::Read, Access::Write];

            /// Its entry point (a matrix multiply's is in the source only
            /// for the kinds of its path: [`Kernels::ALL`]'s).
            fn entry(self) -> &'static str {
                match self {
                    Kernel::Embed => "embed",
                    Kernel::Quant => "quant",
                    Kernel::Residual => "residual",
                    Kernel::PerLayer => "per_layer",
                    Kernel::Argmax => "argmax",
                    Kernel::GemvQ8(kind) => match kind {
                        $(Kind::$k => concat!("gemv_q8_", $name),)*
                    },
                    Kernel::GemmQ8(kind) => match kind {
                        $(Kind::$k => concat!("gemm_q8_", $name),)*
                    },
                    Kernel::GemvF32(kind) => match kind {
                        $(Kind::$k => concat!("gemv_f32_", $name),)*
                    },
                    Kernel::GemmF32(kind) => match kind {
                        $(Kind::$k => concat!("gemm_f32_", $name),)*
                    },
                    Kernel::Prep => "prep",
                    Kernel::Attend => "attend",
                    Kernel::Join => "join",
                    Kernel::Flash => "flash",
                }
            }
        }
    };
}

kernels!(
    names: [F32 "f32", Bf16 "bf16", Q4K "q4k", Q5K "q5k", Q6K "q6k"],
    q8: [Q4K, Q6K],
    plain: [Q4K, Q6K, Bf16, F32]
);

/// The rows a GEMV's workgroup takes, the columns of a GEMM's tile (the
/// matrix multiplies' layouts, `matmul*.wgsl`).
pub(super) const GEMV_ROWS: usize = 16;
pub(super) const GEMM_TILE: usize = 64;
/// The values a scale of the rows written at 8 bits covers.
pub(super) const Q8_BLOCK: usize = 32;
/// The head's logits are capped at this (`cap · tanh(x / cap)`).
pub(super) const LOGIT_CAP: f32 = 30.0;
/// The invocations of a workgroup of flash attention, each 16 of a
/// query's values (`flash`: a block of FLASH_LANES / (D / 16) queries).
pub(super) const FLASH_LANES: usize = 256;
/// The words before the tokens in `ints` (the state).
pub(super) const TOKENS: usize = 16;
/// No norm (`residual`), no other rows (`quant`).
pub(super) const NONE: usize = u32::MAX as usize;
/// `residual`'s flags.
pub(super) const ADD: usize = 1;
pub(super) const SCALE: usize = 2;
pub(super) const OUT: usize = 4;
/// The rows written at 8 bits (the q8 path's mode; 0 is f32s).
pub(super) const Q8: usize = 1;

/// A matrix multiply's numbers: the engine's ten ([`MUL_NUMBERS`]), then
/// the rows' scales in `ints` and (the head) where its workgroups'
/// largest go.
const MM: [&str; MUL_NUMBERS.len() + 2] = {
    let mut all = [""; MUL_NUMBERS.len() + 2];
    let mut i = 0;
    while i < MUL_NUMBERS.len() {
        all[i] = MUL_NUMBERS[i];
        i += 1;
    }
    all[i] = "qs";
    all[i + 1] = "best";
    all
};

/// Each kernel's numbers by name, in the order they are written; the
/// kernels read them by these names (`prep_rows()`: [`constants`]).
const NUMBERS: [(&str, &[&str]); 10] = [
    ("mm", &MM),
    (
        "embed",
        &["rows", "first", "stated", "at", "scale", "width", "kind"],
    ),
    (
        "quant",
        &[
            "rows",
            "width",
            "stride",
            "qa",
            "qs",
            "mode",
            "times",
            "times_stride",
        ],
    ),
    (
        "residual",
        &[
            "width", "stride", "add", "scalar", "norm", "qa", "qs", "mode", "flags", "x_row",
        ],
    ),
    (
        "per_layer",
        &[
            "rows",
            "first",
            "stated",
            "stride",
            "slice",
            "norm",
            "at",
            "out",
            "width",
            "x_row",
            "scale",
            "embed_scale",
            "kind",
        ],
    ),
    ("argmax", &["groups", "best", "positions"]),
    (
        "prep",
        &[
            "rows",
            "first",
            "stated",
            "d",
            "stride",
            "q_norm",
            "k_norm",
            "queries",
            "keys",
            "values",
            "positions",
            "rope",
            "kv",
            "heads",
            "mask",
            "turned",
        ],
    ),
    (
        "attend",
        &[
            "d",
            "window",
            "queries",
            "keys",
            "values",
            "positions",
            "splits",
            "mask",
            "least",
            "per",
        ],
    ),
    (
        "join",
        &[
            "d", "splits", "qa", "qs", "mode", "heads", "window", "least", "per",
        ],
    ),
    (
        "flash",
        &[
            "rows",
            "first",
            "d",
            "window",
            "queries",
            "keys",
            "values",
            "positions",
            "heads",
            "group",
            "mask",
        ],
    ),
];

/// The names of `kernel`'s numbers.
pub(super) fn fields(kernel: &str) -> &'static [&'static str] {
    NUMBERS
        .iter()
        .find(|(k, _)| *k == kernel)
        .map(|(_, f)| *f)
        .expect("a kernel with numbers")
}

/// `kernel`'s numbers from `values`, given by name in its order (checked).
pub(super) fn named(kernel: &str, values: &[(&str, usize)]) -> [usize; 16] {
    debug_assert!(
        fields(kernel).iter().eq(values.iter().map(|v| &v.0)),
        "{kernel}'s numbers by name, in order"
    );
    params(&values.iter().map(|v| v.1).collect::<Vec<_>>())
}

/// The numbers the kernels share with the plan, declared for them: the
/// flags, modes and shapes, each kind's id, its blocks' bytes and the
/// bytes of a row of it (`row_bytes_KIND`), and each kernel's numbers by
/// name (`fn prep_rows() -> u32`).
fn constants() -> String {
    let mut out = engine::wgsl_consts(&[
        ("TOKENS", TOKENS),
        ("NONE", NONE),
        ("ADD", ADD),
        ("SCALE", SCALE),
        ("OUT", OUT),
        ("Q8", Q8),
        ("Q8_BLOCK", Q8_BLOCK),
        ("GEMV_ROWS", GEMV_ROWS),
        ("GEMM_TILE", GEMM_TILE),
        ("FLASH_LANES", FLASH_LANES),
        ("GELU", After::Gelu.flag()),
        ("GEGLU", After::GeGlu.flag()),
        ("ARGMAX", After::Argmax { stride: 0 }.flag()),
    ]);
    for kind in Kind::ALL {
        let (values, bytes) = kind.block();
        let name = suffix(kind);
        let upper = name.to_uppercase();
        out += &engine::wgsl_consts(&[
            (&upper, kind.id() as usize),
            (&format!("{upper}_BYTES"), bytes),
        ]);
        out += &format!(
            "fn row_bytes_{name}(cols: u32) -> u32 {{ return cols / {values}u * {bytes}u; }}\n"
        );
    }
    out += &format!(
        "const LOGIT_CAP: f32 = {:?};\nconst HALF_SQRT2: f32 = {:?};\n",
        LOGIT_CAP,
        std::f32::consts::FRAC_1_SQRT_2
    );
    for (kernel, names) in NUMBERS {
        let named = (0..).zip(names).map(|(i, n)| (format!("{kernel}_{n}"), i));
        out += &engine::wgsl_numbers("f", named);
    }
    out
}

/// The kernels' source: the matrix multiplies written out for each kind
/// they take.
pub(super) fn source() -> String {
    let mut source = [
        &constants(),
        include_str!("prelude.wgsl"),
        include_str!("../common.wgsl"),
        include_str!("blocks.wgsl"),
        include_str!("matmul.wgsl"),
        include_str!("attention.wgsl"),
        include_str!("rows.wgsl"),
    ]
    .join("\n");
    for &kind in Q8_KINDS {
        source += &include_str!("matmul_q8.wgsl").replace("KIND", suffix(kind));
    }
    for &kind in PLAIN_KINDS {
        source += &include_str!("matmul_plain.wgsl").replace("KIND", suffix(kind));
    }
    source
}
