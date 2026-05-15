---
name: leibniz-fast-rust-perf-instrumentation
description: Add gated performance timer instrumentation to LeibnizFast Rust functions. Use when measuring or profiling Rust/WASM performance in src/.
---

# LeibnizFast Rust Perf Instrumentation

Use this skill when adding performance instrumentation to Rust code under
`src/`.

## Implementation

1. Import the timer where needed:

   ```rust
   use crate::perf::PerfTimer;
   ```

2. Ensure the function has access to a `debug: bool` flag. If a new path
   needs the flag, thread it from the nearest struct that already stores
   `debug`. Do not hardcode `true`.

3. Instantiate the timer at the start of the measured function:

   ```rust
   let _timer = PerfTimer::new("StructName::method_name", debug);
   ```

4. If dimensions, byte counts, tile counts, or similar runtime context are
   useful, call `finish_with` before returning:

   ```rust
   _timer.finish_with(&format!("{}x{}", rows, cols));
   ```

5. For simple cases with no useful dynamic context, rely on the timer's
   `Drop` behavior and keep `let _timer = ...`.

## Exclusions

Do not instrument:

- Trivial getters and setters.
- Pure math functions called in tight loops.
- Functions that only delegate to another instrumented function.
- Test-only code.
