//! # Performance Timing
//!
//! Lightweight instrumentation for measuring function execution times.
//! All timing is gated behind a `debug: bool` flag — when disabled,
//! the cost is a single branch (predicted not-taken).

/// Get the current high-resolution timestamp in milliseconds.
///
/// Uses `performance.now()` from the Web Performance API.
/// Returns `0.0` if the browser APIs are unavailable (e.g. in a worker
/// without a global `window`).
fn now_ms() -> f64 {
    web_sys::window()
        .and_then(|w| w.performance())
        .map_or(0.0, |p| p.now())
}

/// Lightweight performance timer that logs elapsed time on `finish()`.
///
/// When `debug` is `false`, all methods are no-ops (zero overhead beyond
/// a single branch prediction).
pub(crate) struct PerfTimer {
    label: &'static str,
    start: f64,
    debug: bool,
}

impl PerfTimer {
    /// Start a new timer. When `debug` is false, no timestamp is taken.
    pub fn new(label: &'static str, debug: bool) -> Self {
        let start = if debug { now_ms() } else { 0.0 };
        Self {
            label,
            start,
            debug,
        }
    }

    /// Log the elapsed time with the original label.
    pub fn finish(self) {
        if self.debug {
            let elapsed = now_ms() - self.start;
            log::info!("[perf] {}: {:.2}ms", self.label, elapsed);
        }
        // Prevent Drop from firing since we already logged
        std::mem::forget(self);
    }

    /// Log elapsed time with additional context (for dynamic info like dimensions).
    #[allow(dead_code)]
    pub fn finish_with(self, extra: &str) {
        if self.debug {
            let elapsed = now_ms() - self.start;
            log::info!("[perf] {} ({}): {:.2}ms", self.label, extra, elapsed);
        }
        // Prevent Drop from firing since we already logged
        std::mem::forget(self);
    }
}

impl Drop for PerfTimer {
    fn drop(&mut self) {
        if self.debug {
            let elapsed = now_ms() - self.start;
            log::info!("[perf] {}: {:.2}ms", self.label, elapsed);
        }
    }
}
