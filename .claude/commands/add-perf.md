After adding new functions or methods to the LeibnizFast codebase, add performance instrumentation if the function performs meaningful work worth measuring (GPU operations, data processing, buffer allocation, pipeline creation, etc.).

Steps:
1. For **Rust** functions/methods in `src/`:
   - Import `PerfTimer` if not already in scope: `use crate::perf::PerfTimer;`
   - Ensure the function has access to a `debug: bool` flag (from `self.debug`, a parameter, or the containing struct).
   - Add `let _timer = PerfTimer::new("StructName::method_name", debug);` at the start of the function.
   - For functions with useful dynamic context (dimensions, sizes, counts), use `_timer.finish_with(&format!("{}x{}", rows, cols));` instead of relying on auto-drop.
   - The timer auto-logs via `Drop`, so `let _timer = ...` is sufficient for simple cases.

2. For **TypeScript** methods in `js/index.ts`:
   - Use the existing `timeSync` helper: `this.timeSync('methodName', () => this.inner.methodName(...))`.

3. Do NOT add instrumentation to:
   - Trivial getters/setters
   - Pure math functions that are called in tight loops
   - Functions that just delegate to another already-instrumented function
   - Test-only code

4. If the new function requires passing `debug` through a new code path, thread it from the nearest struct that already has a `debug` field.
