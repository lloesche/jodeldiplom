//! The fader's taper: where on its travel a level sits, as on a desk's
//! fader. +6 dB at the top, 0 dB at three quarters of the way up, the
//! steps of 10 dB below it closer and closer, −∞ at the bottom. Drawing
//! the cap, the strip's meters beside it and dragging it all go through
//! [`position_of`] and [`db_at`], so one ruler serves them all.

/// The top of the travel, in dB.
pub const TOP_DB: f64 = 6.0;

/// The lowest mark of the scale, in dB: below it the next step down is
/// −∞.
pub const LAST_MARK_DB: f64 = -60.0;

/// The scale's marks, top to bottom: (position, dB), the position from 0
/// (the bottom) to 1 (the top). Between two marks the position is linear
/// in dB; below the last it is linear in the gain, down to 0 (−∞) at the
/// bottom.
pub const MARKS: [(f64, f64); 7] = [
    (1.0, 6.0),
    (0.75, 0.0),
    (0.6, -10.0),
    (0.45, -20.0),
    (0.32, -30.0),
    (0.2, -40.0),
    (0.08, -60.0),
];

/// Where 0 dB is on the travel.
pub const UNITY: f64 = 0.75;

/// The level in dB at `position` (0: the bottom, −∞; 1: the top, +6 dB).
pub fn db_at(position: f64) -> f64 {
    let p = position.clamp(0.0, 1.0);
    let (last_p, last_db) = MARKS[MARKS.len() - 1];
    if p < last_p {
        // Linear in the gain below the last mark.
        return db_of_gain(gain(last_db) * p / last_p);
    }
    for pair in MARKS.windows(2) {
        let ((p0, db0), (p1, db1)) = (pair[0], pair[1]);
        if p >= p1 {
            return db1 + (db0 - db1) * (p - p1) / (p0 - p1);
        }
    }
    last_db
}

/// The position of `db` on the travel: [`db_at`] undone (above the top:
/// the top; −∞: the bottom).
pub fn position_of(db: f64) -> f64 {
    if db >= TOP_DB {
        return 1.0;
    }
    let (last_p, last_db) = MARKS[MARKS.len() - 1];
    if db < last_db {
        return (gain(db) / gain(last_db) * last_p).clamp(0.0, last_p);
    }
    for pair in MARKS.windows(2) {
        let ((p0, db0), (p1, db1)) = (pair[0], pair[1]);
        if db >= db1 {
            return p1 + (p0 - p1) * (db - db1) / (db0 - db1);
        }
    }
    last_p
}

/// The linear gain of `db` (−∞: 0).
pub fn gain(db: f64) -> f64 {
    if db == f64::NEG_INFINITY {
        0.0
    } else {
        10f64.powf(db / 20.0)
    }
}

/// The dB of a linear gain (0: −∞).
pub fn db_of_gain(gain: f64) -> f64 {
    if gain > 0.0 {
        20.0 * gain.log10()
    } else {
        f64::NEG_INFINITY
    }
}

/// `db` as a fader shows it: to a tenth of a dB, never above [`TOP_DB`].
pub fn snap(db: f64) -> f64 {
    if db == f64::NEG_INFINITY {
        return db;
    }
    let db = (db.min(TOP_DB) * 10.0).round() / 10.0;
    // −0.0 is 0.0.
    db + 0.0
}

/// `db` moved by `by` dB, as the keys move a fader: never above the top;
/// a step across 0 dB stops there; from −∞ up to the last mark, and from
/// the last mark (or below it) down to −∞.
pub fn step(db: f64, by: f64) -> f64 {
    if db == f64::NEG_INFINITY {
        return if by > 0.0 { LAST_MARK_DB } else { db };
    }
    if by < 0.0 && db <= LAST_MARK_DB {
        return f64::NEG_INFINITY;
    }
    let moved = db + by;
    if (db < 0.0 && moved > 0.0) || (db > 0.0 && moved < 0.0) {
        return 0.0;
    }
    let moved = if db > LAST_MARK_DB {
        moved.max(LAST_MARK_DB)
    } else {
        moved
    };
    snap(moved)
}

/// A level in dB as a number for people, to `decimals`, with a `+` above
/// 0 when `plus`: `0.0`, `−3.2`, `+6`, `−∞`; a level that rounds to 0 is
/// `0`.
pub fn number(db: f64, decimals: usize, plus: bool) -> String {
    if db == f64::NEG_INFINITY {
        return "−∞".into();
    }
    let scale = 10f64.powi(decimals as i32);
    let rounded = (db * scale).round() / scale;
    let digits = format!("{:.*}", decimals, rounded.abs());
    if rounded > 0.0 && plus {
        format!("+{digits}")
    } else if rounded < 0.0 {
        format!("−{digits}")
    } else {
        digits
    }
}

/// A fader's level for people: `0.0 dB`, `−38.5 dB`, `+6.0 dB`, `−∞ dB`.
pub fn label(db: f64) -> String {
    format!("{} dB", number(db, 1, true))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_marks_sit_where_a_desk_prints_them() {
        assert_eq!(db_at(1.0), 6.0);
        assert_eq!(db_at(UNITY), 0.0);
        assert_eq!(position_of(0.0), UNITY);
        assert_eq!(db_at(0.0), f64::NEG_INFINITY);
        for (p, db) in MARKS {
            assert!((db_at(p) - db).abs() < 1e-9, "{p} {db}");
            assert!((position_of(db) - p).abs() < 1e-9, "{p} {db}");
        }
        assert_eq!(position_of(f64::NEG_INFINITY), 0.0);
        assert_eq!(position_of(20.0), 1.0);
    }

    #[test]
    fn position_and_db_undo_each_other_and_only_rise() {
        let mut last = f64::NEG_INFINITY;
        for i in 0..=1000 {
            let p = f64::from(i) / 1000.0;
            let db = db_at(p);
            assert!(db >= last, "{p}: {db} after {last}");
            last = db;
            if p > 0.0 {
                assert!((position_of(db) - p).abs() < 1e-9, "{p}");
            }
        }
    }

    #[test]
    fn keys_step_through_the_scale_to_off_and_back_and_stop_at_unity() {
        assert_eq!(step(0.0, 1.0), 1.0);
        assert_eq!(step(5.5, 1.0), 6.0);
        assert_eq!(step(-0.3, 1.0), 0.0);
        assert_eq!(step(0.4, -1.0), 0.0);
        assert_eq!(step(-2.5, 6.0), 0.0);
        assert_eq!(step(-59.5, -1.0), -60.0);
        assert_eq!(step(-60.0, -1.0), f64::NEG_INFINITY);
        assert_eq!(step(f64::NEG_INFINITY, 1.0), -60.0);
        assert_eq!(step(f64::NEG_INFINITY, -6.0), f64::NEG_INFINITY);
        assert_eq!(step(-38.5, 6.0), -32.5);
    }

    #[test]
    fn levels_are_shown_to_a_tenth() {
        assert_eq!(snap(0.04), 0.0);
        assert!(snap(-0.04).is_sign_positive());
        assert_eq!(snap(-0.26), -0.3);
        assert_eq!(snap(-38.54), -38.5);
        assert_eq!(label(0.0), "0.0 dB");
        assert_eq!(label(-38.5), "−38.5 dB");
        assert_eq!(label(6.0), "+6.0 dB");
        assert_eq!(label(f64::NEG_INFINITY), "−∞ dB");
        assert_eq!(number(-0.02, 1, false), "0.0");
        assert_eq!(number(-10.0, 0, true), "−10");
        assert_eq!(number(6.0, 0, true), "+6");
        assert_eq!(number(3.24, 1, false), "3.2");
        assert_eq!(gain(0.0), 1.0);
        assert_eq!(gain(f64::NEG_INFINITY), 0.0);
    }
}
