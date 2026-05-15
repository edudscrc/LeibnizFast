# LeibnizFast Codex Guide

## Project

LeibnizFast is a GPU-accelerated 2D matrix visualization library for
browsers, published as the npm package `leibniz-fast`. It combines a
Rust/WASM core built on `wgpu` with a thin TypeScript wrapper. WebGPU is
required; do not add CPU or WebGL fallback paths.

## Commands

- Build WASM: `npm run build:wasm`
- Build JS/types: `npm run build:js`
- Full build: `npm run build`
- Run examples: `npm run dev`, `npm run dev:chart`,
  `npm run dev:waterfall`, `npm run dev:cpp-stream`
- Rust tests: `npm run test:rs` or `cargo test <test_name>`
- Full tests: `npm run test`
- Rust lint: `npm run lint:rs`
- TypeScript lint: `npm run lint:ts`
- Full lint: `npm run lint`
- Docs: `npm run docs:dev`, `npm run docs:build`,
  `npm run docs:preview`

## Architecture

Data flows from a JavaScript `Float32Array` ring buffer into the
TypeScript `LeibnizFast` wrapper, then across WASM exports in `src/lib.rs`
into Rust core state. GPU compute copies raw `f32` data into tiled
`R32Float` textures, and the render shader applies range, colormap, ring
unwrap, and canvas rendering.

Pure Rust logic should stay testable without WebGPU where possible:
`camera`, `chunked_upload`, `colormap`, `colormap_data`, `interaction`,
`matrix`, and `tile_grid`. GPU/WASM-only modules are `perf`, `pipeline`,
and `renderer`.

The TypeScript layer should stay thin. `js/index.ts` owns WASM loading,
public API wrapping, DOM event listeners, callbacks, and interaction
routing. `js/types.ts` owns public API types. `js/axes.ts` renders the 2D
overlay for axes, labels, title, selections, and hover highlights.

## Critical Invariants

- JS streaming buffers are column-major: `data[col * rows + row]`.
  Avoid CPU transposes and O(cols) shifts in the streaming path.
- The compute shader's column-major path indexes staging data as
  `col * col_stride + row` when `col_major = 1`.
- Ring-buffer rendering computes `ring_offset = ring_cursor / total_cols`
  in full-matrix UV space. The render shader must apply
  `fract(full_x + ring_offset)` before mapping to tile-local UVs.
- Multi-tile ring behavior depends on keeping X in full-matrix space until
  after ring unwrap. Do not move ring offset math into tile-local space.
- Y camera mapping is pre-composed on the CPU; X remains full-matrix until
  ring unwrap.
- Pan and zoom should only update uniform/camera state. They must not
  rerun compute work.
- `set_data` full rerenders must reset the ring cursor so subsequent
  `setDataScrolled` frames match the JavaScript waterfall buffer.
- `setDataChunks()` may skip CPU-side retention with `retainData: false`;
  hover must then report unavailable values instead of fabricating data.
- Matrices larger than `maxTextureDimension2D` use tiled textures. Tile
  textures need both `COPY_SRC` and `COPY_DST` usage flags.
- WebGPU adapter/device failures should surface clear WebGPU-required
  errors. Use `LeibnizFast.checkSupport()` in examples for friendly
  preflight checks.

## Coding Standards

- Prefer test-first changes when behavior is changing. Cover happy paths
  and edge cases, and keep tests order-independent.
- Avoid magic numbers and string literals inside logic. Extract meaningful
  constants for grid sizes, memory limits, offsets, and configuration.
- Use descriptive names and keep functions focused on one responsibility.
- Validate public API inputs at boundaries and fail fast with clear errors.
- Keep public TypeScript API shapes strict. Do not use `any`; use
  `unknown` with narrowing when needed.
- Prefer `interface` over `type` for public API shapes, and keep exports
  limited to the intentional public surface.
- Public-facing classes, interfaces, and functions need useful JSDoc.
- When a change affects documented behavior, public APIs, examples, build
  steps, or user-facing usage, update the relevant docs in the same change;
  skip only when there is no meaningful docs impact.
- Keep the TypeScript layer focused on API wrapping, memory coordination,
  WASM interaction, and browser events.
- Rust production code should avoid `unwrap()` and `expect()`. Propagate
  errors with `?` and descriptive error types where feasible.
- Rust code must pass `cargo fmt` and `cargo clippy -- -D warnings`.
- Put Rust unit tests next to implementation in `#[cfg(test)]` modules.
- Minimize JS/WASM boundary crossings, especially for large arrays. Prefer
  passing pointers or typed buffers over repeated serialization.
- WGSL workgroup sizes should have comments explaining the choice.
- Respect WebGPU alignment rules. Prefer explicit padding or `vec4` when it
  avoids cross-platform layout mismatch.
- Create render and compute pipelines during initialization and reuse them.
  Do not create new pipelines in main render or compute loops.
- Destroy WebGPU resources when users call cleanup/disposal APIs.

## Verification

Scale verification to the risk of the change. For narrow Rust logic, prefer
focused `cargo test <test_name>` first, then broader checks when needed. For
shared behavior, public API changes, shaders, or GPU resource changes, run
the relevant lint/test/build commands before finishing.

Before publishing or handing off larger changes, prefer:

- `npm run lint`
- `npm run test`
- `npm run build`
