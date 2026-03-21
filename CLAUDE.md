# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

LeibnizFast is a GPU-accelerated 2D matrix visualization library for browsers, published as npm package `leibniz-fast`. It combines a Rust/WASM core (wgpu) with a thin TypeScript wrapper. WebGPU is the primary backend; WebGL2 is the fallback.

## Build & Dev Commands

```bash
npm run build:wasm    # wasm-pack build --target web --release → pkg/
npm run build:js      # tsup js/index.ts --format esm --dts → dist/
npm run build         # both WASM + JS
npm run dev           # build + serve at localhost:8080/examples/basic/
```

## Test & Lint

```bash
npm run test:rs       # cargo test (unit tests across Rust modules)
cargo test <test_name>  # run a single Rust test
npm run lint:rs       # cargo fmt --check && cargo clippy -- -D warnings
npm run lint:ts       # prettier --check js/ && eslint js/
npm run lint          # all linting (Rust + TypeScript)
```

## Architecture

### Data Flow

```
JavaScript Float32Array
  → TypeScript LeibnizFast class (js/index.ts) — event handling, WASM init
    → WASM exports (src/lib.rs, wasm_entry module)
      → Rust core: MatrixData (CPU), MatrixView (GPU buffer), chunked upload
        → GPU compute shader (shaders/colormap.wgsl): normalize + colormap → RGBA texture
          → GPU render shader (shaders/render.wgsl): camera transform → canvas
```

### Rust Modules

**Pure logic (testable natively, no GPU):** `camera`, `chunked_upload`, `colormap`, `colormap_data`, `interaction`, `matrix`, `tile_grid`

**GPU/WASM-only (cfg=wasm32):** `perf`, `pipeline`, `renderer`

Key entry point: `src/lib.rs` contains the `wasm_entry` module with the main `LeibnizFast` struct exported via `#[wasm_bindgen]`.

### TypeScript Layer

`js/index.ts` — thin wrapper that lazy-loads WASM, wraps the WASM instance with a typed API, manages DOM event listeners, and provides callbacks (`onCreate`, `onHover`).

`js/types.ts` — public API types (`ColormapName`, `CreateOptions`, etc.).

### GPU Resource Patterns

- **Chunked upload:** Large matrices split into ~16MB chunks, 16-row aligned (`chunked_upload.rs`)
- **Tiling:** Matrices exceeding `maxTextureDimension2D` are split into a grid of smaller textures (`tile_grid.rs`)
- **Staging path:** If data exceeds `max_buffer_size`, compute shader runs per-chunk instead of all-at-once
- **Camera-only updates:** Pan/zoom only updates a uniform buffer; compute shader does not re-run

### WebGL2 Fallback Limitations

No hover tooltips; colormap changes require full data reload; colormapping is CPU-side.

## Toolchain

- **Rust:** Stable channel, `wasm32-unknown-unknown` target (see `rust-toolchain.toml`)
- **wasm-pack:** Compiles Rust → WASM, outputs to `pkg/`
- **tsup:** Bundles TypeScript (ESM), outputs to `dist/`
- **Prettier:** 80 cols, semicolons, 2-space indent (`.prettierrc`)
- **ESLint:** TypeScript-ESLint (`eslint.config.js`)

## Key Dependencies

- `wgpu 24` — WebGPU abstraction for Rust
- `wasm-bindgen` + `web-sys` — Rust↔JS interop
- `bytemuck` — zero-copy type casting for GPU buffers
