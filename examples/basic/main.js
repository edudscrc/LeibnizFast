/**
 * Basic example: renders a sine-wave matrix with interactive controls.
 */
import init, { LeibnizFast } from '../../pkg/leibniz_fast.js';

const canvas = document.getElementById('canvas');
const colormapSelect = document.getElementById('colormap');
const sizeSelect = document.getElementById('size');
const streamingCheckbox = document.getElementById('streaming');
const tooltip = document.getElementById('tooltip');
const errorBanner = document.getElementById('error-banner');

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

/**
 * Generate a sine-wave test matrix (or a sub-region for streaming).
 * Creates interesting visual patterns for testing zoom/pan/colormap.
 *
 * @param {number} rows - Number of rows to generate
 * @param {number} cols - Number of columns
 * @param {number} [rowOffset=0] - Starting row index (for streaming chunks)
 * @param {number} [totalRows=rows] - Total matrix height (for consistent pattern)
 * @returns {Float32Array}
 */
function generateSineWave(rows, cols, rowOffset = 0, totalRows = rows) {
  const data = new Float32Array(rows * cols);
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const x = c / cols;
      const y = (r + rowOffset) / totalRows;
      // Combine multiple frequencies for visual interest
      data[r * cols + c] =
        Math.sin(x * 20) * Math.cos(y * 20) +
        Math.sin((x + y) * 10) * 0.5 +
        Math.sin(Math.sqrt(x * x + y * y) * 30) * 0.3;
    }
  }
  return data;
}

async function main() {
  // Initialize WASM module
  await init();

  // Create the viewer
  const viewer = await LeibnizFast.create(canvas, colormapSelect.value);

  // Tiling handles matrices larger than maxTextureDimension — all sizes are selectable.
  // Very large sizes (≥16000) auto-enable streaming to avoid OOM from JS allocation.
  const maxDim = viewer.getMaxTextureDimension();
  console.log(`Device maxTextureDimension2D: ${maxDim} (tiling handles larger matrices)`);
  for (const option of sizeSelect.options) {
    const optSize = parseInt(option.value);
    if (optSize > maxDim) {
      option.text += ' (tiled)';
    }
  }

  // Annotate streaming checkbox with memory context
  const streamingLabel = streamingCheckbox.parentElement;
  streamingLabel.title =
    'Streaming API sends data in ~16 MB chunks instead of one large allocation. ' +
    'Required for matrices too large to fit in a single JS allocation (~2 GB).';

  /**
   * Load data using either the standard or streaming API.
   *
   * The streaming path (beginData → appendChunk loop → endData) generates and
   * uploads data in 1000-row slices, keeping peak JS memory proportional to
   * one chunk rather than the full matrix. This is the only viable path for
   * very large matrices (e.g. 32000×32000 = 4 GB if allocated at once).
   *
   * Tiling handles matrices larger than the GPU's maxTextureDimension2D.
   *
   * @param {number} size
   * @param {boolean} useStreaming
   */
  function loadData(size, useStreaming) {
    // Auto-enable streaming for large matrices to avoid JS OOM from single allocation
    const matrixBytes = size * size * 4;
    if (matrixBytes > 1e9 && !useStreaming) {
      console.log(`Auto-enabling streaming for ${size}×${size} (${(matrixBytes / 1e9).toFixed(1)} GB)`);
      useStreaming = true;
      streamingCheckbox.checked = true;
    }

    try {
      if (useStreaming) {
        // Streaming API: allocate GPU buffer upfront, then push 1000-row chunks
        viewer.beginData(size, size);
        const chunkRows = Math.min(1000, size);
        for (let startRow = 0; startRow < size; startRow += chunkRows) {
          const endRow = Math.min(startRow + chunkRows, size);
          const rows = endRow - startRow;
          const chunk = generateSineWave(rows, size, startRow, size);
          viewer.appendChunk(chunk, startRow);
        }
        viewer.endData();
      } else {
        // Standard path: allocates the full Float32Array at once in JS
        const data = generateSineWave(size, size, 0, size);
        viewer.setData(data, size, size);
      }
    } catch (err) {
      showError(`Failed to load ${size}×${size}: ${err}`);
    }
  }

  // Load initial data
  let size = parseInt(sizeSelect.options[sizeSelect.selectedIndex].value);
  loadData(size, streamingCheckbox.checked);

  // Set up hover tooltip
  viewer.onHover((row, col, value) => {
    tooltip.style.display = 'block';
    tooltip.textContent = `[${row}, ${col}] = ${value.toFixed(4)}`;
  });

  // Mouse interaction: pan (drag) and hover (tooltip)
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

  window.addEventListener('mouseup', () => {
    viewer.onMouseUp();
  });

  canvas.addEventListener('mouseleave', () => {
    tooltip.style.display = 'none';
  });

  // Scroll to zoom (cursor-anchored)
  canvas.addEventListener(
    'wheel',
    (e) => {
      e.preventDefault();
      const rect = canvas.getBoundingClientRect();
      viewer.onWheel(e.clientX - rect.left, e.clientY - rect.top, -e.deltaY);
    },
    { passive: false },
  );

  // Colormap change
  colormapSelect.addEventListener('change', () => {
    viewer.setColormap(colormapSelect.value);
  });

  // Size change
  sizeSelect.addEventListener('change', () => {
    size = parseInt(sizeSelect.value);
    loadData(size, streamingCheckbox.checked);
  });

  // Streaming toggle — reload current data through chosen path
  streamingCheckbox.addEventListener('change', () => {
    loadData(size, streamingCheckbox.checked);
  });
}

main().catch(console.error);
