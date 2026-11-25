//! The day where the player is, from the system: what the result's diploma is
//! dated with.

/// Today's date where the player is (year, month 1 to 12, day), if the
/// system says.
pub fn today() -> Option<(i32, u32, u32)> {
    local::today()
}

#[cfg(windows)]
mod local {
    pub fn today() -> Option<(i32, u32, u32)> {
        // SAFETY: it takes nothing and only returns the local time.
        let now = unsafe { windows::Win32::System::SystemInformation::GetLocalTime() };
        Some((
            i32::from(now.wYear),
            u32::from(now.wMonth),
            u32::from(now.wDay),
        ))
    }
}

#[cfg(unix)]
mod local {
    use std::time::{SystemTime, UNIX_EPOCH};

    pub fn today() -> Option<(i32, u32, u32)> {
        let secs = SystemTime::now().duration_since(UNIX_EPOCH).ok()?.as_secs();
        let now = libc::time_t::try_from(secs).ok()?;
        // SAFETY: a plain struct of integers (and on some systems a
        // pointer, left null), all zeros a valid one.
        let mut tm: libc::tm = unsafe { std::mem::zeroed() };
        // SAFETY: a time to read and a tm to write into, both ours.
        let made = unsafe { libc::localtime_r(&now, &mut tm) };
        let (month, day) = (
            u32::try_from(tm.tm_mon + 1).ok()?,
            u32::try_from(tm.tm_mday).ok()?,
        );
        (!made.is_null()).then_some((tm.tm_year + 1900, month, day))
    }
}

#[cfg(not(any(windows, unix)))]
mod local {
    pub fn today() -> Option<(i32, u32, u32)> {
        None
    }
}

#[cfg(test)]
mod tests {
    /// A day in this century, its month and day in range.
    #[test]
    fn today_is_a_day() {
        let (year, month, day) = super::today().expect("the system's date");
        assert!((2020..2200).contains(&year));
        assert!((1..=12).contains(&month) && (1..=31).contains(&day));
    }
}
