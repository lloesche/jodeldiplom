//! Parameters as data: what a patch's and each engine's numbers are
//! called, what they take, and how they are read and set ([`Param`]), so
//! a patch can be shown, edited and written out by its tables alone
//! ([`crate::patch::PARAMS`], each engine module's `PARAMS`, and the one
//! view of them all on [`crate::engines::AnyParams`]).
//!
//! Every value is a number: a range's in its unit, a choice's the index
//! of its option, a switch's 0 or 1. Setting one clamps it to what it
//! takes first, so a value from anywhere (a slider, a file) is one the
//! sound takes.

/// How a range's values sit along a slider.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Scale {
    /// Evenly.
    Linear,
    /// Finer near the low end: the value grows as the square of the
    /// share (times that start at 0).
    Fine,
    /// By ratios: each step the same factor (frequencies, rates; the low
    /// end above 0).
    Log,
    /// Whole numbers, evenly.
    Whole,
}

/// What a parameter takes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Kind {
    /// A number from `min` to `max`, in `unit` (none: a plain number).
    Range {
        /// The least.
        min: f32,
        /// The most.
        max: f32,
        /// Its unit, as written after it (`"Hz"`, `"s"`; empty: none).
        unit: &'static str,
        /// How it sits along a slider.
        scale: Scale,
    },
    /// One of the options, by its index.
    Choice(&'static [&'static str]),
    /// Off (0) or on (1).
    Toggle,
}

impl Kind {
    /// `v` as this kind takes it: in the range (a whole number where it
    /// takes those), an option's index, 0 or 1; not a number, the least.
    pub fn clamp(self, v: f32) -> f32 {
        let v = if v.is_nan() { f32::NEG_INFINITY } else { v };
        match self {
            Kind::Range {
                min, max, scale, ..
            } => {
                let v = v.clamp(min, max);
                if scale == Scale::Whole {
                    v.round().clamp(min, max)
                } else {
                    v
                }
            }
            Kind::Choice(options) => v.round().clamp(0.0, options.len().saturating_sub(1) as f32),
            Kind::Toggle => {
                if v >= 0.5 {
                    1.0
                } else {
                    0.0
                }
            }
        }
    }

    /// Where `v` sits along a slider, 0 to 1.
    pub fn share(self, v: f32) -> f32 {
        let v = self.clamp(v);
        match self {
            Kind::Range {
                min, max, scale, ..
            } => {
                if max <= min {
                    return 0.0;
                }
                let linear = (v - min) / (max - min);
                match scale {
                    Scale::Linear | Scale::Whole => linear,
                    Scale::Fine => linear.sqrt(),
                    Scale::Log => (v.max(min).ln() - min.ln()) / (max.ln() - min.ln()),
                }
            }
            Kind::Choice(options) => v / options.len().saturating_sub(1).max(1) as f32,
            Kind::Toggle => v,
        }
        .clamp(0.0, 1.0)
    }

    /// The value at `share` (0 to 1) along a slider.
    pub fn at(self, share: f32) -> f32 {
        let s = if share.is_nan() {
            0.0
        } else {
            share.clamp(0.0, 1.0)
        };
        let v = match self {
            Kind::Range {
                min, max, scale, ..
            } => match scale {
                Scale::Linear | Scale::Whole => min + (max - min) * s,
                Scale::Fine => min + (max - min) * s * s,
                Scale::Log => (min.ln() + (max.ln() - min.ln()) * s).exp(),
            },
            Kind::Choice(options) => s * options.len().saturating_sub(1) as f32,
            Kind::Toggle => s,
        };
        self.clamp(v)
    }

    /// `v` as it is shown: a range's number to as many places as it needs,
    /// with its unit; a choice's option; on or off.
    pub fn text(self, v: f32) -> String {
        let v = self.clamp(v);
        match self {
            Kind::Range { unit, scale, .. } => {
                let a = v.abs();
                let number = if scale == Scale::Whole || a >= 100.0 {
                    format!("{v:.0}")
                } else if a >= 10.0 {
                    format!("{v:.1}")
                } else if a >= 1.0 {
                    format!("{v:.2}")
                } else {
                    format!("{v:.3}")
                };
                if unit.is_empty() {
                    number
                } else {
                    format!("{number} {unit}")
                }
            }
            Kind::Choice(options) => options.get(v as usize).copied().unwrap_or("").to_owned(),
            Kind::Toggle => if v >= 0.5 { "on" } else { "off" }.to_owned(),
        }
    }
}

/// What a parameter is, without how it is read: its group (empty: none),
/// its name, what it takes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Spec {
    /// The group it is shown in (`"op 1"`, `"lfo"`; empty: none).
    pub group: &'static str,
    /// Its name in its group.
    pub name: &'static str,
    /// What it takes.
    pub kind: Kind,
}

impl Spec {
    /// Its group and name together, as a file writes it: `"op 1 ratio"`.
    pub fn key(&self) -> String {
        if self.group.is_empty() {
            self.name.to_owned()
        } else {
            format!("{} {}", self.group, self.name)
        }
    }
}

/// A parameter of `P`: what it is ([`Spec`]), how it is read and set, and
/// when it applies (a pulse's width only to a pulse): one that does not
/// is neither shown nor written out.
pub struct Param<P> {
    /// What it is.
    pub spec: Spec,
    read: fn(&P) -> f32,
    write: fn(&mut P, f32),
    applies: fn(&P) -> bool,
}

/// Always.
fn always<P>(_: &P) -> bool {
    true
}

impl<P> Clone for Param<P> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<P> Copy for Param<P> {}

impl<P> std::fmt::Debug for Param<P> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.spec.fmt(f)
    }
}

impl<P> Param<P> {
    /// A parameter of `kind`, read by `read` and set by `write` (given a
    /// value it takes).
    pub const fn new(
        group: &'static str,
        name: &'static str,
        kind: Kind,
        read: fn(&P) -> f32,
        write: fn(&mut P, f32),
    ) -> Param<P> {
        Param {
            spec: Spec { group, name, kind },
            read,
            write,
            applies: always,
        }
    }

    /// Whether it applies to `p`.
    pub fn applies(&self, p: &P) -> bool {
        (self.applies)(p)
    }

    /// A number from `min` to `max` in `unit`, on `scale`.
    pub const fn range(
        (group, name): (&'static str, &'static str),
        (min, max, unit, scale): (f32, f32, &'static str, Scale),
        read: fn(&P) -> f32,
        write: fn(&mut P, f32),
    ) -> Param<P> {
        Param::new(
            group,
            name,
            Kind::Range {
                min,
                max,
                unit,
                scale,
            },
            read,
            write,
        )
    }

    /// One of `options`.
    pub const fn choice(
        (group, name): (&'static str, &'static str),
        options: &'static [&'static str],
        read: fn(&P) -> f32,
        write: fn(&mut P, f32),
    ) -> Param<P> {
        Param::new(group, name, Kind::Choice(options), read, write)
    }

    /// Off (0) or on (1).
    pub const fn toggle(
        (group, name): (&'static str, &'static str),
        read: fn(&P) -> f32,
        write: fn(&mut P, f32),
    ) -> Param<P> {
        Param::new(group, name, Kind::Toggle, read, write)
    }

    /// Its value in `p`.
    pub fn get(&self, p: &P) -> f32 {
        (self.read)(p)
    }

    /// Sets it in `p` to `v`, as it takes it ([`Kind::clamp`]).
    pub fn set(&self, p: &mut P, v: f32) {
        (self.write)(p, self.spec.kind.clamp(v));
    }
}

/// `param`, applying only where `applies` says.
pub const fn when<P>(param: Param<P>, applies: fn(&P) -> bool) -> Param<P> {
    Param { applies, ..param }
}

/// What has parameters as data, each by its place in its tables: how
/// many, what each is, whether it applies now, its value, and setting it.
pub trait Knobs {
    /// How many.
    fn count(&self) -> usize;

    /// What the `k`th is.
    fn spec(&self, k: usize) -> Option<Spec>;

    /// Whether the `k`th applies now.
    fn applies(&self, k: usize) -> bool;

    /// The `k`th's value.
    fn get(&self, k: usize) -> Option<f32>;

    /// Sets the `k`th to `v` (as it takes it); whether there is one.
    fn set(&mut self, k: usize, v: f32) -> bool;
}

/// What `table`'s `k`th is (the helpers [`Knobs`] is made of, over one
/// table).
pub(crate) fn spec_of<P>(table: &[Param<P>], k: usize) -> Option<Spec> {
    table.get(k).map(|q| q.spec)
}

/// Whether `table`'s `k`th applies to `p`.
pub(crate) fn applies_to<P>(table: &[Param<P>], p: &P, k: usize) -> bool {
    table.get(k).is_some_and(|q| q.applies(p))
}

/// `table`'s `k`th's value in `p`.
pub(crate) fn get_in<P>(table: &[Param<P>], p: &P, k: usize) -> Option<f32> {
    table.get(k).map(|q| q.get(p))
}

/// Sets `table`'s `k`th in `p`; whether there is one.
pub(crate) fn set_in<P>(table: &[Param<P>], p: &mut P, k: usize, v: f32) -> bool {
    table.get(k).map(|q| q.set(p, v)).is_some()
}

/// A number parameter of a field (a path of fields: `adsr.attack`):
/// `field!("group", "name", path, min, max, "unit", Scale)`.
macro_rules! field {
    ($group:expr, $name:expr, $($f:ident).+, $min:expr, $max:expr, $unit:expr, $scale:ident) => {
        $crate::params::Param::range(
            ($group, $name),
            ($min, $max, $unit, $crate::params::Scale::$scale),
            |p| p.$($f).+,
            |p, v| p.$($f).+ = v,
        )
    };
}

/// A switch parameter of a `bool` field: `switch!("group", "name", path)`.
macro_rules! switch {
    ($group:expr, $name:expr, $($f:ident).+) => {
        $crate::params::Param::toggle(
            ($group, $name),
            |p| $crate::params::on(p.$($f).+),
            |p, v| p.$($f).+ = v >= 0.5,
        )
    };
}

/// A choice parameter of a field whose values are `all`, named `names`:
/// `pick!("group", "name", path, ALL, NAMES)`.
macro_rules! pick {
    ($group:expr, $name:expr, $($f:ident).+, $all:expr, $names:expr) => {
        $crate::params::Param::choice(
            ($group, $name),
            &$names,
            |p| $crate::params::index_of(&$all, &p.$($f).+),
            |p, v| p.$($f).+ = $crate::params::nth(&$all, v),
        )
    };
}

pub(crate) use {field, pick, switch};

/// A switch's value: 1 on, 0 off.
pub fn on(b: bool) -> f32 {
    f32::from(u8::from(b))
}

/// The option `v` picks of `all` (its index; past the last, the last).
///
/// # Panics
///
/// With no options.
pub fn nth<T: Copy>(all: &[T], v: f32) -> T {
    all[(v.max(0.0) as usize).min(all.len() - 1)]
}

/// Where `x` is among `all`, as a choice's value (not there: the first).
pub fn index_of<T: PartialEq>(all: &[T], x: &T) -> f32 {
    all.iter().position(|a| a == x).unwrap_or(0) as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    const HZ: Kind = Kind::Range {
        min: 20.0,
        max: 20_000.0,
        unit: "Hz",
        scale: Scale::Log,
    };

    #[test]
    fn a_range_goes_along_its_slider_and_back() {
        for kind in [
            HZ,
            Kind::Range {
                min: 0.0,
                max: 2.0,
                unit: "s",
                scale: Scale::Fine,
            },
            Kind::Range {
                min: -24.0,
                max: 24.0,
                unit: "st",
                scale: Scale::Linear,
            },
        ] {
            for k in 0..=10 {
                let share = k as f32 / 10.0;
                let v = kind.at(share);
                assert!((kind.share(v) - share).abs() < 1e-4, "{kind:?} {share}");
            }
        }
        assert!((HZ.at(0.5) - 632.46).abs() < 0.1);
        assert_eq!(HZ.text(800.0), "800 Hz");
    }

    #[test]
    fn a_value_is_clamped_to_what_it_takes() {
        assert_eq!(HZ.clamp(5.0), 20.0);
        assert_eq!(HZ.clamp(f32::NAN), 20.0);
        let whole = Kind::Range {
            min: 0.0,
            max: 15.0,
            unit: "",
            scale: Scale::Whole,
        };
        assert_eq!(whole.clamp(3.6), 4.0);
        let choice = Kind::Choice(&["saw", "square", "sine"]);
        assert_eq!(choice.clamp(7.0), 2.0);
        assert_eq!(choice.text(1.2), "square");
        assert_eq!(choice.share(1.0), 0.5);
        assert_eq!(Kind::Toggle.clamp(0.7), 1.0);
        assert_eq!(Kind::Toggle.text(0.0), "off");
    }
}
