# LeibnizFast — Claude Code Project Context

## Project Overview
GPU-accelerated 2D matrix visualization library for the browser. Renders matrices of millions to billions of pixels as interactive heatmaps with zoom, pan, and tooltip inspection.

**Tech stack**: Rust + wgpu → WASM (via wasm-pack) → npm package
**Backend**: WebGPU (primary), WebGL2 fallback (limited — no compute shaders)

## Architecture

```
JS (Float32Array) → WASM (Rust) → GPU buffers
                                  ↓
                    Compute shader: apply colormap → RGBA texture
                                  ↓
                    Render pass: textured quad with camera transform → canvas
```

- Compute shader pre-applies colormap to texture (only re-runs on data/colormap change, not on pan/zoom)
- Camera is UV-space offset/scale in fragment shader
- Events handled in JS, forwarded to Rust via wasm-bindgen
- CPU copy of matrix data for tooltip lookups (avoids async GPU readback)
- Nearest-neighbor sampling for pixel-perfect cells at high zoom

## Build & Test Commands

```bash
# Rust
cargo fmt --check          # Check formatting
cargo clippy -- -D warnings # Lint
cargo test                  # Unit tests (camera, colormap, interaction, matrix)
wasm-pack build --target web # Build WASM

# TypeScript
npx prettier --check js/   # Check formatting
npx eslint js/              # Lint

# Full build
npm run build               # WASM + JS bundle
npm run dev                 # Build + serve example
```

## Coding Style

### Rust
- Doc comments (`///`) on every public item
- Module-level docs (`//!`) at the top of each file
- 4-space indent, snake_case functions, PascalCase types, SCREAMING_SNAKE constants
- Group imports: std → external → crate-internal
- Prefer `Result`/`Option` over panics; use `?` operator

### TypeScript
- JSDoc on every export
- 2-space indent, single quotes, trailing commas (Prettier)
- Strict mode, explicit return types on public functions

### WGSL
- Comment every binding
- Explain workgroup sizes and dispatch logic
- Comment math operations

## Key Patterns

- **TDD**: Write tests before implementation for pure-logic modules
- **DI via traits**: `ColormapProvider` trait allows testing without GPU
- **Factory pattern**: `PipelineFactory` centralizes wgpu pipeline creation
- **State pattern**: `InteractionState` enum for mouse events (Idle/Dragging)
- **Pure/GPU split**: `CameraState`/`Camera`, `MatrixData`/`MatrixView` — pure math is testable, GPU wrapper adds buffers

## File Structure

- `src/lib.rs` — wasm-bindgen entry point, public API
- `src/renderer.rs` — wgpu setup, pipelines, render loop
- `src/camera.rs` — CameraState (pure math) + Camera (GPU uniform)
- `src/colormap.rs` — ColormapProvider trait + ColormapTexture
- `src/colormap_data.rs` — Const 256-entry RGB tables
- `src/matrix.rs` — MatrixData (CPU) + MatrixView (GPU buffer)
- `src/interaction.rs` — InteractionState enum (Idle/Dragging)
- `src/pipeline.rs` — PipelineFactory for compute + render pipelines
- `src/shaders/colormap.wgsl` — Compute shader
- `src/shaders/render.wgsl` — Vertex + fragment shader
- `js/index.ts` — TypeScript wrapper
- `js/types.ts` — Type definitions

## Common Tasks

### Add a new colormap
1. Add const `[[u8; 3]; 256]` to `src/colormap_data.rs`
2. Add name to `COLORMAP_NAMES` and `get_colormap_by_name` match
3. Add test in `src/colormap.rs` tests
4. Add option to `js/types.ts` `ColormapName`

### Add an interaction mode
1. Add variant to `InteractionState` enum in `src/interaction.rs`
2. Add transition logic in `mouse_down`/`mouse_move`/`mouse_up`
3. Add tests for the new transitions
4. Wire up in `src/lib.rs` event handlers
