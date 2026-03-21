---
name: ts-perf-instrumentation
description: Adds performance timing wrappers to TypeScript methods. Use this when instructed to measure performance or add profiling to the TypeScript API layer.
---

When adding performance instrumentation to the TypeScript codebase (`js/index.ts` or similar API boundaries), you must adhere to the following rules.

## Implementation Steps
1. **Wrapper Usage:** Use the existing `timeSync` helper to wrap the inner call. 
2. **Syntax:** Implement the timing using the following pattern: `this.timeSync('methodName', () => this.inner.methodName(...))`

## Exclusion Criteria
Do NOT add instrumentation to:
* Trivial getters/setters.
* Functions executed in tight iteration loops, such as per-vertex or per-pixel operations.
* Functions that simply delegate to another already-instrumented function.
* Test files.