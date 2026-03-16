# LeibnizFast

GPU-accelerated 2D matrix visualization for the browser.

Render matrices of millions to billions of pixels as interactive heatmaps with smooth zoom, pan, and cell-level tooltip inspection — powered by WebGPU via Rust and WASM.

## Features

- **GPU-accelerated rendering** — compute shader applies colormaps, fragment shader handles zoom/pan
- **Interactive** — scroll to zoom (cursor-anchored), drag to pan, hover for cell values
- **Multiple colormaps** — viridis, inferno, magma, plasma, cividis, grayscale
- **Large matrices** — handles millions of cells at interactive frame rates
- **Pixel-perfect** — nearest-neighbor sampling shows individual cells at high zoom
- **TypeScript API** — clean typed wrapper over Rust/WASM core

## Browser Requirements

WebGPU support required. Currently available in:
- Chrome 113+ / Edge 113+
- Firefox Nightly (behind flag)
- Safari Technology Preview

Falls back to WebGL2 with reduced features (no compute shaders).

## Installation

```bash
npm install leibniz-fast
```

## Quick Start

```html
<canvas id="canvas" width="800" height="600"></canvas>
```

```typescript
import { LeibnizFast } from 'leibniz-fast';

const canvas = document.getElementById('canvas') as HTMLCanvasElement;
const viewer = await LeibnizFast.create(canvas, { colormap: 'viridis' });

// Set a 1000×2000 matrix
const data = new Float32Array(1000 * 2000);
// ... fill data ...
viewer.setData(data, { rows: 1000, cols: 2000 });

// Change colormap
viewer.setColormap('inferno');

// Set custom data range
viewer.setRange(0.0, 1.0);

// Hover tooltip
viewer.onHover((row, col, value) => {
  console.log(`[${row}, ${col}] = ${value}`);
});

// Cleanup
viewer.destroy();
```

## API Reference

### `LeibnizFast.create(canvas, options?)`
Async factory method. Initializes WebGPU and returns a viewer instance.
- `canvas`: `HTMLCanvasElement`
- `options.colormap`: `ColormapName` (default: `'viridis'`)

### `viewer.setData(data, options)`
Set the matrix data to visualize.
- `data`: `Float32Array` in row-major order
- `options.rows`: number of rows
- `options.cols`: number of columns

### `viewer.setColormap(name)`
Change the colormap. Available: `'viridis'`, `'inferno'`, `'magma'`, `'plasma'`, `'cividis'`, `'grayscale'`.

### `viewer.setRange(min, max)`
Override auto-detected data range for colormap mapping.

### `viewer.onHover(callback)`
Register a callback `(row, col, value) => void` for hover events.

### `viewer.destroy()`
Clean up GPU resources and event listeners.

## Available Colormaps

| Name | Description |
|------|-------------|
| `viridis` | Perceptually uniform, colorblind-friendly (purple → teal → yellow) |
| `inferno` | Dark to bright (black → purple → orange → yellow) |
| `magma` | Dark to bright (black → purple → pink → light yellow) |
| `plasma` | Blue → purple → orange → yellow |
| `cividis` | Optimized for color vision deficiency (blue → gray-green → yellow) |
| `grayscale` | Simple black to white |

## Development Setup

### 1. Install Prerequisites

**Rust** (if not already installed):
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source "$HOME/.cargo/env"
```

**WASM target + wasm-pack**:
```bash
rustup target add wasm32-unknown-unknown
cargo install wasm-pack
```

**Node.js 18+** — install via your package manager or [nvm](https://github.com/nvm-sh/nvm):
```bash
# Arch Linux
sudo pacman -S nodejs npm

# Ubuntu/Debian
sudo apt install nodejs npm

# macOS
brew install node

# Or use nvm (any OS)
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.0/install.sh | bash
nvm install 22
```

### 2. Install Node Dependencies

```bash
cd LeibnizFast
npm install
```

### 3. Build & Run the Example

```bash
# Build the WASM package
wasm-pack build --target web --release

# Serve from repo root (so ../../pkg/ resolves correctly)
npx http-server . -p 8080 -o /examples/basic/
```

Open `http://localhost:8080` in a WebGPU-capable browser (Chrome 113+). You'll see a 1000x1000 sine-wave heatmap. Scroll to zoom, drag to pan, hover for cell values, use the dropdowns to change colormap and matrix size.

### 4. Build Commands

```bash
npm run build          # Full build (WASM + JS bundle)
npm run build:wasm     # WASM only
npm run build:js       # JS/TS only
npm run dev            # Build + serve example at localhost:8080
```

### 5. Test & Lint

```bash
cargo test                        # 31 Rust unit tests
cargo fmt --check                 # Rust formatting
cargo clippy -- -D warnings       # Rust linting
npx prettier --check js/          # TypeScript formatting
npx eslint js/                    # TypeScript linting
```

## Architecture

```
JS (Float32Array) → WASM (Rust) → GPU buffers
                                  ↓
                    Compute shader: apply colormap → RGBA texture
                                  ↓
                    Render pass: textured quad with camera transform → canvas
```

The compute shader only re-runs when data or colormap changes. Pan/zoom only updates a camera uniform in the fragment shader, making viewport changes nearly free.

## License

MIT
