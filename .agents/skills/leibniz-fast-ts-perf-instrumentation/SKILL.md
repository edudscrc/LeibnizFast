---
name: leibniz-fast-ts-perf-instrumentation
description: Add performance timing wrappers to the LeibnizFast TypeScript API layer. Use when measuring or profiling TypeScript methods around WASM/API boundaries.
---

# LeibnizFast TypeScript Perf Instrumentation

Use this skill when adding performance instrumentation to TypeScript API
methods, especially in `js/index.ts` or nearby wrapper code.

## Implementation

Wrap the inner call with the existing `timeSync` helper:

```ts
this.timeSync('methodName', () => this.inner.methodName(...));
```

Keep timing at API boundaries or other coarse operations where it helps
explain user-visible performance. Do not add timing inside per-element,
per-vertex, per-pixel, or tight iteration loops.

## Exclusions

Do not instrument:

- Trivial getters and setters.
- Functions executed inside tight loops.
- Functions that only delegate to another already-instrumented function.
- Test files.
