# LeibnizFast

GPU-accelerated 2D matrix visualization for the browser. Renders interactive heatmaps with zoom, pan, and cell-level hover inspection — powered by WebGPU (with WebGL2 fallback) via Rust and WASM.

## Prerequisites

- **Rust** (stable) + `wasm32-unknown-unknown` target
- **wasm-pack**
- **Node.js 18+**

```bash
# Install Rust (if needed)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Add WASM target and wasm-pack
rustup target add wasm32-unknown-unknown
cargo install wasm-pack

# Install Node dependencies
npm install
```

## Build

```bash
npm run build:wasm     # Rust → WASM (outputs to pkg/)
npm run build:js       # TypeScript → ESM bundle (outputs to dist/)
npm run build          # Both
npm run dev            # Build + serve at localhost:8080/examples/basic/
```

## Test & Lint

```bash
npm run test:rs        # cargo test
npm run lint:rs        # cargo fmt --check && cargo clippy -- -D warnings
npm run lint:ts        # prettier --check && eslint
npm run lint           # All linting
```

## Install (as a library)

```bash
npm install leibniz-fast
```

## Usage

```typescript
import { LeibnizFast } from "leibniz-fast";

const canvas = document.getElementById("canvas") as HTMLCanvasElement;
const viewer = await LeibnizFast.create(canvas, { colormap: "viridis" });

const rows = 1000,
  cols = 2000;
const data = new Float32Array(rows * cols);
// ... fill data in row-major order ...

viewer.setData(data, { rows, cols });

// Change colormap at any time
viewer.setColormap("inferno");

// Override auto-detected data range
viewer.setRange(0.0, 1.0);

// Cell hover callback
viewer.onHover((row, col, value) => {
  console.log(`[${row}, ${col}] = ${value}`);
});

// Clean up when done
viewer.destroy();
```

Zoom, pan, and hover are handled automatically by the TypeScript wrapper.

### Streaming API

For very large matrices or incremental data:

```typescript
viewer.beginData({ rows: 8000, cols: 8000 });

for (let startRow = 0; startRow < 8000; startRow += 1000) {
  const chunk = generateRows(startRow, 1000, 8000); // Float32Array
  viewer.appendChunk(chunk, startRow);
}

viewer.endData();
```

### Colormaps

`viridis` `inferno` `magma` `plasma` `cividis` `grayscale`

## Examples

All examples require the WASM build first:

```bash
npm run build:wasm
```

### Basic

Generates a sine-wave matrix in JavaScript. Demonstrates colormap switching, size selection, hover tooltips, and optional streaming mode.

```bash
npm run dev
# Open http://localhost:8080/examples/basic/
```

### GPU Generation

Generates the matrix entirely on the GPU via a WebGPU compute shader, then passes the result to LeibnizFast. Eliminates the JS CPU bottleneck for large matrices.

```bash
npm run dev
# Open http://localhost:8080/examples/gpu-gen/
```

### Waterfall

Live scrolling waterfall display. A C++ generator simulates spatial-temporal sensor data at a fixed sampling rate, streams it via ZeroMQ to a Python bridge, which broadcasts over WebSocket to the browser.

**Dependencies:** `libzmq`, Python packages `pyzmq` and `websockets`.

```bash
# Terminal 1 — Python bridge
pip install pyzmq websockets
python examples/waterfall/bridge.py

# Terminal 2 — C++ generator
g++ -std=c++17 -O2 -o generator examples/waterfall/generator.cpp -lzmq
./generator

# Terminal 3 — Web server
npm run dev
# Open http://localhost:8080/examples/waterfall/
```

### C++ Live Stream (Wave Equation)

Solves a 2D wave equation in C++ and streams live frames to the browser. Supports chunked transmission and optional zlib compression.

**Dependencies:** `libzmq`, `zlib`, Python packages `pyzmq` and `websockets`.

```bash
# Terminal 1 — Python bridge
pip install pyzmq websockets
python examples/cpp-stream/bridge.py

# Terminal 2 — C++ generator
g++ -std=c++17 -O2 -o generator examples/cpp-stream/generator.cpp -lzmq -lz
./generator                  # plain mode
./generator --compress       # zlib compression (~4x smaller, ~2x FPS)
./generator --size 4096      # larger grid

# Terminal 3 — Web server
npm run dev
# Open http://localhost:8080/examples/cpp-stream/
```

## Browser Support

WebGPU (compute shader path): Chrome 113+, Edge 113+, Firefox Nightly (flag), Safari Technology Preview.

Falls back to WebGL2 with CPU-side colormapping when WebGPU is unavailable.

## License

MIT
