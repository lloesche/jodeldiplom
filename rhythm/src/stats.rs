//! Medians, the two the analysis takes.

use std::cmp::Ordering;

/// The middle value of `values` by `order`, the upper of the two middle
/// ones for an even count (reorders `values`); `None` when empty.
pub(crate) fn upper_median<T: Copy>(
    values: &mut [T],
    order: impl FnMut(&T, &T) -> Ordering,
) -> Option<T> {
    let mid = values.len() / 2;
    (!values.is_empty()).then(|| *values.select_nth_unstable_by(mid, order).1)
}

/// The median of `values`, the mean of the two middle ones for an even
/// count (sorts `values`); `None` when empty.
pub(crate) fn median(values: &mut [f64]) -> Option<f64> {
    values.sort_by(f64::total_cmp);
    let n = values.len();
    match n {
        0 => None,
        _ if n % 2 == 1 => Some(values[n / 2]),
        _ => Some(0.5 * (values[n / 2 - 1] + values[n / 2])),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn medians_of_odd_and_even_counts() {
        assert_eq!(
            upper_median(&mut [3.0, 1.0, 2.0], f64::total_cmp),
            Some(2.0)
        );
        assert_eq!(
            upper_median(&mut [4.0f32, 1.0, 3.0, 2.0], f32::total_cmp),
            Some(3.0)
        );
        assert_eq!(upper_median::<f64>(&mut [], f64::total_cmp), None);
        assert_eq!(median(&mut [3.0, 1.0, 2.0]), Some(2.0));
        assert_eq!(median(&mut [4.0, 1.0, 3.0, 2.0]), Some(2.5));
        assert_eq!(median(&mut []), None);
    }
}
