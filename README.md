# LeibnizFast

WebGPU-only 2D matrix visualization for browsers. LeibnizFast renders large heatmaps with zoom, pan, chart overlays, streaming updates, and hover inspection through a Rust/WASM core and TypeScript API.

There is no CPU or WebGL2 fallback. If WebGPU is unavailable, creation fails with a clear error that the app can display to the user.

## Quick Start

Prerequisites:

- Rust stable with `wasm32-unknown-unknown`
- `wasm-pack`
- Node.js 18+

```bash
rustup target add wasm32-unknown-unknown
cargo install wasm-pack
npm install
npm run build
```

## Examples

Every example is runnable through one npm command. The runner installs Node dependencies if `node_modules` is missing, builds WASM/JS, starts a local static server, and prints the URL.

```bash
npm run example:chart
npm run example:waterfall
npm run example:cpp-stream
```

The streaming examples also compile their native generator, create/use `venv`, install `examples/requirements.txt`, start the Python bridge, and clean up child processes on Ctrl+C.

Waterfall example system dependencies:

```bash
# Debian/Ubuntu
sudo apt install g++ libzmq3-dev python3-venv
```

The cpp-stream example is CUDA-only. It requires `nvcc`, `libzmq`, an NVIDIA CUDA-capable GPU, and a working NVIDIA driver. The runner compiles the generator with `nvcc` and runs a CUDA preflight before serving the page; if CUDA is unavailable, it exits with an informative error instead of falling back to CPU simulation.

```bash
# Debian/Ubuntu package names vary by distro/CUDA setup
sudo apt install nvidia-cuda-toolkit libzmq3-dev python3-venv
```

The runner does not auto-open a browser because the right browser depends on WebGPU support on your platform. Open the printed `http://localhost:...` URL in a supported browser.

## Browser Support

LeibnizFast requires:

- `navigator.gpu`
- a non-null WebGPU adapter
- successful WebGPU device creation
- a secure context: HTTPS or localhost

A website cannot enable browser flags for the user. The only way to avoid flags is to use a browser/platform where WebGPU is already enabled and serve from HTTPS or localhost.

No-flag targets to try first:

- Chrome/Edge on supported desktop platforms. Chrome shipped WebGPU by default in Chrome 113 on ChromeOS, macOS, and Windows; current Chromium platform details are tracked by the GPUWeb wiki.
- Firefox 141+ on Windows; other Firefox platforms are still rolling out according to the GPUWeb implementation status.
- Safari 26+ on macOS/iOS/iPadOS/visionOS according to the GPUWeb implementation status.

References:

- Chrome WebGPU launch: <https://developer.chrome.com/blog/webgpu-release>
- MDN WebGPU API and secure-context requirement: <https://developer.mozilla.org/en-US/docs/Web/API/WebGPU_API>
- Current cross-browser status: <https://github.com/gpuweb/gpuweb/wiki/Implementation-Status>
- Firefox 141 Windows announcement: <https://mozillagfx.wordpress.com/2025/07/15/shipping-webgpu-on-windows-in-firefox-141/>

Optional troubleshooting:

- Make sure hardware acceleration is enabled in the browser.
- Serve the app from `https://` or `http://localhost`.
- On some Linux/ARM/blocked-GPU setups, the browser may still require WebGPU flags or newer GPU drivers. LeibnizFast will not fall back to WebGL or CPU rendering.

## Usage

```ts
import { LeibnizFast } from 'leibniz-fast';

const canvas = document.getElementById('canvas') as HTMLCanvasElement;

const support = await LeibnizFast.checkSupport();
if (!support.supported) {
  throw new Error(support.reason);
}

const viewer = await LeibnizFast.create(canvas, { colormap: 'viridis' });

const rows = 1000;
const cols = 2000;
const data = new Float32Array(rows * cols);

viewer.setData(data, { rows, cols });
viewer.setRange(0, 1);
viewer.setColormap('inferno');

viewer.onHover((info) => {
  if (info.valueAvailable) {
    console.log(`[${info.row}, ${info.col}] = ${info.value}`);
  }
});
```

## Large Data

For normal matrices, `setData()` is the simplest path:

```ts
viewer.setData(data, { rows, cols });
```

For large matrices, use explicit ranges whenever possible. This avoids a min/max scan and is the fastest path:

```ts
viewer.setRange(-1, 1);
viewer.setData(data, { rows, cols });
```

For matrices that should not exist as one giant JavaScript `Float32Array`, use chunked upload. Chunks are sequential row-major records. With `retainData: false`, data is uploaded into tiled GPU `R32Float` textures immediately and CPU-side hover values are not retained.

```ts
async function* chunks() {
  for (let startRow = 0; startRow < rows; startRow += 1024) {
    yield {
      startRow,
      data: await loadRows(startRow, Math.min(1024, rows - startRow)),
    };
  }
}

await viewer.setDataChunks(chunks(), {
  rows,
  cols,
  retainData: false,
  range: { min: -1, max: 1 },
});
```

When `retainData: false`, hover callbacks still include `row`, `col`, `x`, and `y`, but `valueAvailable` is `false` and `value` is `NaN`.

For waterfall-style streaming, call `setRange()` once before the first `setDataScrolled()`. Without a fixed range, `setDataScrolled()` throws because the library no longer falls back to a full upload.

```ts
viewer.setRange(-0.5, 1.0);
viewer.setDataScrolled(buffer.data, {
  rows: buffer.rows,
  cols: buffer.cols,
  newCols,
  xOffset: totalColsReceived,
});
```

## Scripts

```bash
npm run build:wasm       # Rust -> WASM in pkg/
npm run build:js         # TypeScript -> ESM in dist/
npm run build            # WASM + JS

npm run example:chart
npm run example:waterfall
npm run example:cpp-stream

npm run test:rs          # cargo test
npm run lint:rs          # cargo fmt --check && cargo clippy -- -D warnings
npm run lint:ts          # prettier --check && eslint
npm run lint             # Rust + TypeScript lint

npm run docs:dev
npm run docs:build
npm run docs:preview
```

## Troubleshooting

`navigator.gpu is not available`

Use a WebGPU-capable browser, and make sure the page is served from HTTPS or localhost.

`No WebGPU adapter was returned`

Enable hardware acceleration, update GPU drivers, and check whether your browser blocks WebGPU for the current GPU/OS combination.

`setDataScrolled() requires a fixed range`

Call `viewer.setRange(min, max)` before the first scrolled update.

`g++` or `zmq.h` not found when running the waterfall example

Install the waterfall example system dependencies. On Debian/Ubuntu: `sudo apt install g++ libzmq3-dev python3-venv`.

`nvcc` missing, no CUDA-capable GPU, or CUDA driver errors when running cpp-stream

The cpp-stream example runs its wave generator on CUDA and has no CPU fallback. Install the NVIDIA CUDA Toolkit, use an NVIDIA CUDA-capable GPU, and make sure the NVIDIA driver is loaded and compatible with the installed toolkit.

## License

MIT
