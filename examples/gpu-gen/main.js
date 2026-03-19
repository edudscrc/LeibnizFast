/**
 * GPU data generation example: generates the sine-wave matrix entirely on the
 * GPU via a WebGPU compute shader, then reads the result back as Float32Array
 * and feeds it to LeibnizFast. This removes the JS loop from the hot path.
 *
 * Strategy:
 *   1. Acquire a GPUDevice (separate from LeibnizFast's internal device).
 *   2. Compile a compute shader that writes sin/cos values into a storage buffer.
 *   3. Copy the storage buffer into a mappable readback buffer.
 *   4. Map and hand the Float32Array to LeibnizFast (setData / appendChunk).
 *
 * For large matrices the generation + readback happens in 1000-row chunks to
 * keep peak GPU buffer allocations bounded (same ceiling as the basic example's
 * streaming path).
 */
import init, { LeibnizFast } from '../../pkg/leibniz_fast.js';

const canvas = document.getElementById('canvas');
const colormapSelect = document.getElementById('colormap');
const sizeSelect = document.getElementById('size');
const debugCheckbox = document.getElementById('debug');
const tooltip = document.getElementById('tooltip');
const errorBanner = document.getElementById('error-banner');

/** Whether debug timing is enabled. */
let debugEnabled = debugCheckbox.checked;

/**
 * Show an error message in the banner for a few seconds.
 * @param {string} msg
 */
function showError(msg) {
  errorBanner.textContent = msg;
  errorBanner.style.display = 'block';
  clearTimeout(errorBanner._timer);
  errorBanner._timer = setTimeout(() => {
    errorBanner.style.display = 'none';
  }, 6000);
}

// ---------------------------------------------------------------------------
// GPU sine-wave generator
// ---------------------------------------------------------------------------

/**
 * Holds a compiled compute pipeline and its associated reusable resources.
 * Call generateChunk() to produce rows of the sine-wave pattern on the GPU.
 */
class GpuSineGenerator {
  /**
   * @param {GPUDevice} device
   */
  constructor(device) {
    this._device = device;

    // Uniforms: cols (u32), totalRows (u32), rowOffset (u32), _pad (u32)
    this._uniformBuffer = device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    this._pipeline = device.createComputePipeline({
      layout: 'auto',
      compute: {
        module: device.createShaderModule({ code: SINE_SHADER }),
        entryPoint: 'main',
      },
    });
  }

  /**
   * Generate `rows` rows of the sine-wave pattern starting at `rowOffset`,
   * within a matrix of `totalRows × cols`. Returns a Float32Array (CPU copy).
   *
   * @param {number} rows       - Number of rows to generate in this call
   * @param {number} cols       - Matrix width
   * @param {number} rowOffset  - First row index within the full matrix
   * @param {number} totalRows  - Full matrix height (for normalisation)
   * @returns {Promise<Float32Array>}
   */
  async generateChunk(rows, cols, rowOffset, totalRows) {
    const device = this._device;
    const elementCount = rows * cols;
    const byteSize = elementCount * 4; // f32 = 4 bytes

    // Storage buffer the compute shader writes into
    const storageBuffer = device.createBuffer({
      size: byteSize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });

    // Mappable readback buffer
    const readbackBuffer = device.createBuffer({
      size: byteSize,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });

    // Upload uniforms: [cols, totalRows, rowOffset, _pad]
    device.queue.writeBuffer(
      this._uniformBuffer,
      0,
      new Uint32Array([cols, totalRows, rowOffset, 0]),
    );

    const bindGroup = device.createBindGroup({
      layout: this._pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this._uniformBuffer } },
        { binding: 1, resource: { buffer: storageBuffer } },
      ],
    });

    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(this._pipeline);
    pass.setBindGroup(0, bindGroup);
    // Each workgroup covers 16×16 elements; dispatch enough to cover rows×cols.
    pass.dispatchWorkgroups(Math.ceil(cols / 16), Math.ceil(rows / 16));
    pass.end();

    encoder.copyBufferToBuffer(storageBuffer, 0, readbackBuffer, 0, byteSize);
    device.queue.submit([encoder.finish()]);

    // Map and copy out
    await readbackBuffer.mapAsync(GPUMapMode.READ);
    const result = new Float32Array(readbackBuffer.getMappedRange().slice(0));
    readbackBuffer.unmap();

    // Release GPU memory immediately — we have the CPU copy
    storageBuffer.destroy();
    readbackBuffer.destroy();

    return result;
  }

  destroy() {
    this._uniformBuffer.destroy();
  }
}

/**
 * WGSL compute shader.
 * Computes: sin(x*20)*cos(y*20) + sin((x+y)*10)*0.5 + sin(sqrt(x²+y²)*30)*0.3
 * where x = col/cols, y = (row + rowOffset) / totalRows.
 *
 * Layout:
 *   group(0) binding(0) — uniforms  { cols, totalRows, rowOffset, _pad }
 *   group(0) binding(1) — output[]  f32 storage buffer
 *
 * Workgroup size: 16×16 threads (256 invocations per workgroup).
 * global_invocation_id.x → column index
 * global_invocation_id.y → local row index (0..rows-1)
 */
const SINE_SHADER = /* wgsl */ `
struct Uniforms {
  cols      : u32,
  totalRows : u32,
  rowOffset : u32,
  _pad      : u32,
}

@group(0) @binding(0) var<uniform> u : Uniforms;
@group(0) @binding(1) var<storage, read_write> output : array<f32>;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let col = gid.x;
  let row = gid.y; // local row within this chunk

  // Bounds check — dispatched workgroups may exceed buffer extents
  if col >= u.cols || row * u.cols >= arrayLength(&output) {
    return;
  }

  let x = f32(col) / f32(u.cols);
  let y = f32(row + u.rowOffset) / f32(u.totalRows);

  // Same multi-frequency pattern as the basic JS example
  let v = sin(x * 20.0) * cos(y * 20.0)
        + sin((x + y) * 10.0) * 0.5
        + sin(sqrt(x * x + y * y) * 30.0) * 0.3;

  output[row * u.cols + col] = v;
}
`;

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

async function main() {
  // -- WebGPU availability check -------------------------------------------
  if (!navigator.gpu) {
    showError('WebGPU is not supported in this browser.');
    return;
  }

  // -- Acquire GPU device for data generation --------------------------------
  // This device is separate from the one LeibnizFast creates internally.
  let t0 = performance.now();
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) {
    showError('No WebGPU adapter available.');
    return;
  }
  const gpuDevice = await adapter.requestDevice();
  const generator = new GpuSineGenerator(gpuDevice);
  if (debugEnabled) console.log(`[perf] GPU device + shader init: ${(performance.now() - t0).toFixed(2)}ms`);

  // -- WASM init -------------------------------------------------------------
  t0 = performance.now();
  await init();
  if (debugEnabled) console.log(`[perf] WASM init: ${(performance.now() - t0).toFixed(2)}ms`);

  // -- LeibnizFast viewer ----------------------------------------------------
  t0 = performance.now();
  const viewer = await LeibnizFast.create(canvas, colormapSelect.value, debugEnabled);
  if (debugEnabled) console.log(`[perf] LeibnizFast.create: ${(performance.now() - t0).toFixed(2)}ms`);

  // Annotate options that exceed the device texture limit with "(tiled)"
  const maxDim = viewer.getMaxTextureDimension();
  console.log(`Device maxTextureDimension2D: ${maxDim}`);
  for (const option of sizeSelect.options) {
    if (parseInt(option.value) > maxDim) option.text += ' (tiled)';
  }

  // -- Data loading ----------------------------------------------------------

  /**
   * Generate and load the sine-wave matrix using the GPU compute shader.
   *
   * Large matrices (>1 GB) are processed in 1000-row chunks via the streaming
   * API to keep peak GPU + JS memory bounded. Each chunk is generated on the
   * GPU, read back, then immediately handed to LeibnizFast before the next
   * chunk begins.
   *
   * @param {number} size - Matrix side length
   */
  async function loadData(size) {
    const matrixBytes = size * size * 4;
    const useStreaming = matrixBytes > 1e9;
    if (useStreaming) {
      console.log(`Using chunked path for ${size}×${size} (${(matrixBytes / 1e9).toFixed(1)} GB)`);
    }

    try {
      const tLoad = debugEnabled ? performance.now() : 0;

      if (useStreaming) {
        viewer.beginData(size, size);
        const chunkRows = Math.min(1000, size);
        for (let startRow = 0; startRow < size; startRow += chunkRows) {
          const rows = Math.min(chunkRows, size - startRow);
          const tGen = debugEnabled ? performance.now() : 0;
          const chunk = await generator.generateChunk(rows, size, startRow, size);
          if (debugEnabled)
            console.log(`[perf] GPU generateChunk (${rows}×${size}): ${(performance.now() - tGen).toFixed(2)}ms`);
          viewer.appendChunk(chunk, startRow);
        }
        viewer.endData();
      } else {
        const tGen = debugEnabled ? performance.now() : 0;
        const data = await generator.generateChunk(size, size, 0, size);
        if (debugEnabled)
          console.log(`[perf] GPU generateChunk (${size}×${size}): ${(performance.now() - tGen).toFixed(2)}ms`);
        viewer.setData(data, size, size);
      }

      if (debugEnabled)
        console.log(`[perf] loadData total (${size}×${size}): ${(performance.now() - tLoad).toFixed(2)}ms`);
    } catch (err) {
      showError(`Failed to load ${size}×${size}: ${err}`);
    }
  }

  // Initial load
  let size = parseInt(sizeSelect.options[sizeSelect.selectedIndex].value);
  await loadData(size);

  // -- Event handlers --------------------------------------------------------

  viewer.onHover((row, col, value) => {
    tooltip.style.display = 'block';
    tooltip.textContent = `[${row}, ${col}] = ${value.toFixed(4)}`;
  });

  canvas.addEventListener('mousedown', (e) => {
    const rect = canvas.getBoundingClientRect();
    viewer.onMouseDown(e.clientX - rect.left, e.clientY - rect.top);
  });

  canvas.addEventListener('mousemove', (e) => {
    const rect = canvas.getBoundingClientRect();
    viewer.onMouseMove(e.clientX - rect.left, e.clientY - rect.top);
    tooltip.style.left = `${e.clientX + 12}px`;
    tooltip.style.top = `${e.clientY + 12}px`;
  });

  window.addEventListener('mouseup', () => viewer.onMouseUp());

  canvas.addEventListener('mouseleave', () => {
    tooltip.style.display = 'none';
  });

  canvas.addEventListener(
    'wheel',
    (e) => {
      e.preventDefault();
      const rect = canvas.getBoundingClientRect();
      viewer.onWheel(e.clientX - rect.left, e.clientY - rect.top, -e.deltaY);
    },
    { passive: false },
  );

  colormapSelect.addEventListener('change', () => {
    const t = debugEnabled ? performance.now() : 0;
    viewer.setColormap(colormapSelect.value);
    if (debugEnabled) console.log(`[perf] setColormap: ${(performance.now() - t).toFixed(2)}ms`);
  });

  sizeSelect.addEventListener('change', async () => {
    size = parseInt(sizeSelect.value);
    await loadData(size);
  });

  debugCheckbox.addEventListener('change', () => {
    debugEnabled = debugCheckbox.checked;
    console.log(`[perf] Debug timing ${debugEnabled ? 'enabled' : 'disabled'}`);
  });
}

main().catch(console.error);
