---
name: rust-perf-instrumentation
description: Adds performance timer instrumentation to Rust methods and functions. Use this when instructed to measure performance or add profiling to Rust code.
---

When adding performance instrumentation to the Rust codebase (`src/`), you must adhere to the following rules to ensure measurements only occur when debugging is enabled.

## Implementation Steps
1. **Import:** Ensure the timer is in scope: `use crate::perf::PerfTimer;`
2. **Flag Resolution:** Ensure the function has access to a `debug: bool` flag. If the new function requires passing `debug` through a new code path, thread it from the nearest struct that already has a `debug` field. Do not hardcode the flag to `true`.
3. **Instantiation:** Add `let _timer = PerfTimer::new("StructName::method_name", debug);` at the start of the function.
4. **Dynamic Context:** For functions with useful dynamic context (dimensions, sizes, counts), use `_timer.finish_with(&format!("{}x{}", rows, cols));` before the function returns instead of relying on auto-drop.
5. **Auto-Drop:** For simple cases without dynamic context, the timer auto-logs via the `Drop` trait, so `let _timer = ...` is sufficient.

## Exclusion Criteria
Do NOT add instrumentation to:
* Trivial getters/setters.
* Pure math functions that are called in tight loops.
* Functions that simply delegate to another already-instrumented function.
* Test-only code.