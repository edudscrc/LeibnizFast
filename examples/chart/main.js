/**
 * Chart example: static heatmap with GPU-generated data, configurable axes,
 * colormap, and value range.
 *
 * Data is generated on the GPU via a WebGPU compute shader (ported from the
 * former gpu-gen example), then read back and fed to LeibnizFast.
 */
import { LeibnizFast } from '../../dist/index.js';

// ---- DOM refs ------------------------------------------------------------
const canvas = document.getElementById('chart-canvas');
const colormapSelect = document.getElementById('colormap');
const reloadBtn = document.getElementById('reload');
const tooltip = document.getElementById('tooltip');
const errorBanner = document.getElementById('error-banner');

const rowsInput = document.getElementById('rows');
const colsInput = document.getElementById('cols');
const xLabelInput = document.getElementById('x-label');
const xUnitInput = document.getElementById('x-unit');
const xMinInput = document.getElementById('x-min');
const xMaxInput = document.getElementById('x-max');
const yLabelInput = document.getElementById('y-label');
const yUnitInput = document.getElementById('y-unit');
const yMinInput = document.getElementById('y-min');
const yMaxInput = document.getElementById('y-max');
const valueUnitInput = document.getElementById('value-unit');
const vminInput = document.getElementById('vmin');
const vmaxInput = document.getElementById('vmax');

// ---- Constants -----------------------------------------------------------

/** Maximum rows per GPU compute chunk (keeps storage buffer bounded). */
const GPU_CHUNK_ROWS = 1000;

// ---- GPU sine-wave generator ---------------------------------------------

/**
 * WGSL compute shader that generates a multi-frequency sine-wave pattern.
 *
 * Layout:
 *   group(0) binding(0) - uniforms { cols, totalRows, rowOffset, _pad }
 *   group(0) binding(1) - output[] f32 storage buffer
 *
 * Workgroup size: 16x16 threads (256 invocations per workgroup).
 * global_invocation_id.x -> column index
 * global_invocation_id.y -> local row index (0..rows-1)
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
  let row = gid.y;

  if col >= u.cols || row * u.cols >= arrayLength(&output) {
    return;
  }

  let x = f32(col) / f32(u.cols);
  let y = f32(row + u.rowOffset) / f32(u.totalRows);

  let v = sin(x * 20.0) * cos(y * 20.0)
        + sin((x + y) * 10.0) * 0.5
        + sin(sqrt(x * x + y * y) * 30.0) * 0.3;

  output[row * u.cols + col] = v;
}
`;

/**
 * Holds a compiled compute pipeline and reusable uniform buffer.
 * Call generateChunk() to produce rows of the sine-wave pattern on the GPU.
 */
class GpuSineGenerator {
  /** @param {GPUDevice} device */
  constructor(device) {
    this._device = device;

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
   * Generate `rows` rows starting at `rowOffset` within a totalRows x cols
   * matrix. Returns a Float32Array (CPU copy).
   *
   * @param {number} rows
   * @param {number} cols
   * @param {number} rowOffset
   * @param {number} totalRows
   * @returns {Promise<Float32Array>}
   */
  async generateChunk(rows, cols, rowOffset, totalRows) {
    const device = this._device;
    const byteSize = rows * cols * 4;

    const storageBuffer = device.createBuffer({
      size: byteSize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });

    const readbackBuffer = device.createBuffer({
      size: byteSize,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });

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
    pass.dispatchWorkgroups(Math.ceil(cols / 16), Math.ceil(rows / 16));
    pass.end();

    encoder.copyBufferToBuffer(storageBuffer, 0, readbackBuffer, 0, byteSize);
    device.queue.submit([encoder.finish()]);

    await readbackBuffer.mapAsync(GPUMapMode.READ);
    const result = new Float32Array(readbackBuffer.getMappedRange().slice(0));
    readbackBuffer.unmap();

    storageBuffer.destroy();
    readbackBuffer.destroy();

    return result;
  }

  destroy() {
    this._uniformBuffer.destroy();
  }
}

// ---- Utilities -----------------------------------------------------------

/**
 * Show a transient error message in the banner.
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

/**
 * Read chart config from DOM inputs.
 * @returns {{ rows: number, cols: number, chart: object }}
 */
function readConfig() {
  return {
    rows: Math.max(1, parseInt(rowsInput.value) || 500),
    cols: Math.max(1, parseInt(colsInput.value) || 1000),
    chart: {
      title: 'Spectrogram Analysis',
      xAxis: {
        label: xLabelInput.value || undefined,
        unit: xUnitInput.value || undefined,
        min: parseFloat(xMinInput.value),
        max: parseFloat(xMaxInput.value),
      },
      yAxis: {
        label: yLabelInput.value || undefined,
        unit: yUnitInput.value || undefined,
        min: parseFloat(yMinInput.value),
        max: parseFloat(yMaxInput.value),
      },
      valueUnit: valueUnitInput.value || undefined,
    },
  };
}

// ---- Main ----------------------------------------------------------------

/** @type {LeibnizFast|null} */
let viewer = null;

/** @type {GpuSineGenerator|null} */
let generator = null;

/**
 * Generate data on the GPU and load it into the viewer.
 * Large matrices are processed in chunks to keep GPU memory bounded.
 *
 * @param {number} rows
 * @param {number} cols
 */
async function generateAndLoad(rows, cols) {
  const matrixBytes = rows * cols * 4;
  const maxBindingBytes = generator._device.limits.maxStorageBufferBindingSize;
  const maxRowsByBinding = Math.floor(maxBindingBytes / (cols * 4));
  const chunkRows = Math.max(1, Math.min(GPU_CHUNK_ROWS, rows, maxRowsByBinding));

  const useStreaming = matrixBytes > 1e9;

  if (useStreaming) {
    viewer.beginData({ rows, cols });
    for (let startRow = 0; startRow < rows; startRow += chunkRows) {
      const chunkSize = Math.min(chunkRows, rows - startRow);
      const chunk = await generator.generateChunk(chunkSize, cols, startRow, rows);
      viewer.appendChunk(chunk, startRow);
    }
    viewer.endData();
  } else {
    // For smaller matrices, generate all chunks and concatenate
    if (rows <= chunkRows) {
      const data = await generator.generateChunk(rows, cols, 0, rows);
      viewer.setData(data, { rows, cols });
    } else {
      const data = new Float32Array(rows * cols);
      for (let startRow = 0; startRow < rows; startRow += chunkRows) {
        const chunkSize = Math.min(chunkRows, rows - startRow);
        const chunk = await generator.generateChunk(chunkSize, cols, startRow, rows);
        data.set(chunk, startRow * cols);
      }
      viewer.setData(data, { rows, cols });
    }
  }
}

/**
 * Create (or recreate) the viewer with current config, generate data, and
 * apply vmin/vmax range.
 */
async function load() {
  const config = readConfig();

  if (viewer) viewer.destroy();

  viewer = await LeibnizFast.create(canvas, {
    colormap: colormapSelect.value,
    chart: config.chart,
  });

  try {
    await generateAndLoad(config.rows, config.cols);
  } catch (err) {
    showError(`Failed to generate ${config.rows}x${config.cols}: ${err}`);
    return;
  }

  // Apply vmin/vmax if set
  const vmin = parseFloat(vminInput.value);
  const vmax = parseFloat(vmaxInput.value);
  if (isFinite(vmin) && isFinite(vmax) && vmax > vmin) {
    viewer.setRange(vmin, vmax);
  }

  viewer.onHover((info) => {
    tooltip.style.display = 'block';
    tooltip.innerHTML =
      `Y: ${info.y?.toFixed(1) ?? info.row} ${info.yUnit ?? ''}<br>` +
      `X: ${info.x?.toFixed(2) ?? info.col} ${info.xUnit ?? ''}<br>` +
      `Value: ${info.value.toFixed(4)}${info.valueUnit ? ' ' + info.valueUnit : ''}`;
  });
}

async function main() {
  // ---- WebGPU availability check -----------------------------------------
  if (!navigator.gpu) {
    showError(
      'WebGPU is not supported in this browser. ' +
      'Chrome 113+, Edge 113+, or Firefox Nightly required.',
    );
    return;
  }

  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) {
    showError('No WebGPU adapter available.');
    return;
  }
  const gpuDevice = await adapter.requestDevice();
  generator = new GpuSineGenerator(gpuDevice);

  // ---- Initial load ------------------------------------------------------
  await load();

  // ---- Event listeners ---------------------------------------------------

  reloadBtn.addEventListener('click', () => load().catch(console.error));

  // Colormap change is live (no reload needed)
  colormapSelect.addEventListener('change', () => {
    if (viewer) viewer.setColormap(colormapSelect.value);
  });

  // vmin/vmax change is live (no reload needed)
  function applyRange() {
    const vmin = parseFloat(vminInput.value);
    const vmax = parseFloat(vmaxInput.value);
    if (viewer && isFinite(vmin) && isFinite(vmax) && vmax > vmin) {
      viewer.setRange(vmin, vmax);
    }
  }
  vminInput.addEventListener('input', applyRange);
  vmaxInput.addEventListener('input', applyRange);
  vminInput.addEventListener('change', applyRange);
  vmaxInput.addEventListener('change', applyRange);

  canvas.addEventListener('mousemove', (e) => {
    tooltip.style.left = `${e.clientX + 12}px`;
    tooltip.style.top = `${e.clientY + 12}px`;
  });

  canvas.addEventListener('mouseleave', () => {
    tooltip.style.display = 'none';
  });
}

main().catch(console.error);
