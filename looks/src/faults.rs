//! A look held to the catalogue strictly ([`Look::faults`]), for a look a
//! model wrote and may be asked to write again: every way it is not
//! exactly as the catalogue has it, said. That is more than
//! [`Look::check`] notes (it also mends some things silently: a drive's
//! amount held, a knob moved twice, a turn folded round), so a look
//! without faults is checked without a note.

use crate::catalogue::n;
use crate::{guard, passes, Look};

impl Look {
    /// Every way it is not exactly as the catalogue has it, each said
    /// precisely (none: it draws as it asks, nothing held, folded or
    /// dropped, and [`Look::check`] notes nothing): a pass there is none
    /// of; a knob its pass has not, or set outside its range (a turn
    /// too, a count of steps not a whole number, a number that is none);
    /// a mix outside 0 to 1; a drive of a knob there is not or the music
    /// cannot move, by an amount outside -1 to 1, or of a knob moved
    /// twice; over the budget ([`guard::over`]).
    pub fn faults(&self) -> Vec<String> {
        let mut faults = Vec::new();
        let mut cost = 0.0;
        for layer in &self.layers {
            let Some(pass) = passes::named(&layer.pass) else {
                faults.push(format!("there is no pass {:?}", layer.pass));
                continue;
            };
            cost += pass.cost;
            let name = pass.name;
            for (knob, &v) in &layer.knobs {
                let Some((_, k)) = pass.knob(knob) else {
                    let knobs: Vec<&str> = pass.knobs.iter().map(|k| k.name).collect();
                    faults.push(format!(
                        "{name} has no knob {knob:?} (its knobs: {})",
                        knobs.join(", ")
                    ));
                    continue;
                };
                let within = v.is_finite() && v >= k.lo && v <= k.hi;
                if !within {
                    faults.push(format!(
                        "{name}.{knob} is {v}: its range is {} to {}",
                        n(k.lo),
                        n(k.hi)
                    ));
                } else if k.clamp(v) != v && !k.wraps() {
                    faults.push(format!("{name}.{knob} is {v}: a count, a whole number"));
                }
            }
            if !(0.0..=1.0).contains(&layer.mix) {
                faults.push(format!("{name}'s mix is {}: 0 to 1", layer.mix));
            }
            let drives = layer.drives(pass);
            for (j, d) in drives.iter().enumerate() {
                match pass.drivable(&d.knob) {
                    None => faults.push(format!("{name} has no knob {:?} to move", d.knob)),
                    Some((_, swing, _)) if swing <= 0.0 => {
                        faults.push(format!("the music cannot move {name}.{}", d.knob));
                    }
                    Some(_) => {}
                }
                if !d.amount.is_finite() || d.amount.abs() > 1.0 {
                    faults.push(format!(
                        "{name}.{} is moved by {}: -1 to 1",
                        d.knob, d.amount
                    ));
                }
                if drives[..j].iter().any(|o| o.knob == d.knob) {
                    faults.push(format!("{name}.{} is moved twice", d.knob));
                }
            }
        }
        if guard::over(self.layers.len(), cost) {
            faults.push(format!(
                "{} passes costing {}: at most {} costing {}",
                self.layers.len(),
                n(cost),
                guard::MOST_LAYERS,
                n(guard::BUDGET)
            ));
        }
        faults
    }
}

#[cfg(test)]
mod tests {
    use crate::{presets, Drive, Driver, Layer, Look};

    /// A preset with one thing changed in it, a layer at a time.
    fn changed(id: &str, layer: usize, change: impl FnOnce(&mut Layer)) -> Look {
        let mut look = presets::named(id).expect(id).clone();
        change(&mut look.layers[layer]);
        look
    }

    /// A look without faults is checked without a note: each way out of
    /// the catalogue, alone in a preset, is a fault (whether `check` notes
    /// it or mends it silently), and the presets have neither.
    #[test]
    fn a_look_without_faults_is_checked_without_a_note() {
        let drive = |knob: &str, amount: f32| {
            Some(vec![Drive {
                knob: knob.to_owned().into(),
                by: Driver::Bass,
                amount,
            }])
        };
        // rock: curve, halation, grain, chromatic.
        let looks = [
            changed("rock", 2, |l| l.pass = "laser".into()),
            changed("rock", 2, |l| {
                l.knobs.insert("sise".into(), 1.0);
            }),
            changed("rock", 2, |l| {
                l.knobs.insert("size".into(), 9.0);
            }),
            changed("rock", 2, |l| {
                l.knobs.insert("size".into(), f32::NAN);
            }),
            changed("rock", 2, |l| l.mix = 2.0),
            changed("rock", 2, |l| l.drive = drive("sise", 0.5)),
            changed("rock", 2, |l| l.drive = drive("amount", 3.0)),
            changed("rock", 0, |l| {
                l.pass = "cutout".into();
                l.knobs.clear();
                l.knobs.insert("layers".into(), 2.5);
            }),
            changed("rock", 1, |l| {
                l.knobs.insert("radius".into(), -4.0);
            }),
        ];
        let mut over = presets::named("rock").unwrap().clone();
        for pass in ["glow", "kuwahara", "god_rays"] {
            over.layers.push(Layer::of(pass));
        }
        for look in looks.iter().chain([&over]) {
            let notes = look.check().notes;
            assert!(!look.faults().is_empty(), "no fault, but noted: {notes:?}");
        }
        for p in presets::all() {
            assert!(
                p.faults().is_empty() && p.check().notes.is_empty(),
                "{}",
                p.id
            );
        }
    }

    /// Each way out of the catalogue is said precisely.
    #[test]
    fn a_look_out_of_the_catalogue_is_said_precisely() {
        let mut grain = Layer::of("grain");
        grain.knobs.insert("size".into(), 4.0);
        grain.knobs.insert("sise".into(), 1.0);
        grain.mix = 1.5;
        grain.drive = Some(vec![
            Drive {
                knob: "amount".into(),
                by: Driver::Bass,
                amount: 1.5,
            },
            Drive {
                knob: "amount".into(),
                by: Driver::Kick,
                amount: 0.5,
            },
        ]);
        let mut cutout = Layer::of("cutout");
        cutout.knobs.insert("layers".into(), 3.5);
        let mut dust = Layer::of("dust");
        dust.drive = Some(vec![Drive {
            knob: "lit".into(),
            by: Driver::Bass,
            amount: 0.5,
        }]);
        let look = Look {
            id: "x".into(),
            name: String::new(),
            about: String::new(),
            tags: Vec::new(),
            layers: vec![
                grain,
                cutout,
                dust,
                Layer::of("laser"),
                Layer::of("glow"),
                Layer::of("halation"),
                Layer::of("kuwahara"),
            ],
        };
        let faults = look.faults();
        let want = [
            "grain has no knob \"sise\" (its knobs: amount, size, chroma)",
            "grain.size is 4: its range is 0.6 to 3",
            "grain's mix is 1.5: 0 to 1",
            "grain.amount is moved by 1.5: -1 to 1",
            "grain.amount is moved twice",
            "cutout.layers is 3.5: a count, a whole number",
            "the music cannot move dust.lit",
            "there is no pass \"laser\"",
            "7 passes costing 13.3: at most 6 costing 8",
        ];
        assert_eq!(faults, want);
    }
}
