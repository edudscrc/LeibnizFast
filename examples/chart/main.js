/**
 * Chart example: demonstrates axes, labels, units, and title.
 *
 * Uses the TypeScript wrapper (dist/) which includes the chart overlay.
 */
import { LeibnizFast } from '../../dist/index.js';

const canvas = document.getElementById('chart-canvas');
const colormapSelect = document.getElementById('colormap');
const modeSelect = document.getElementById('mode');
const reloadBtn = document.getElementById('reload');
const tooltip = document.getElementById('tooltip');

const ROWS = 500;
const COLS = 1000;

/** Number of columns added per streaming frame. */
const STREAM_CHUNK_COLS = 20;
/** Delay between streaming frames in ms. */
const STREAM_DELAY_MS = 50;
/** Total number of streaming frames to send. */
const STREAM_TOTAL_FRAMES = 80;

/**
 * Generate a sine-wave test matrix representing a spectrogram-like pattern.
 *
 * @param {number} rows
 * @param {number} cols
 * @param {number} [rowOffset=0]
 * @param {number} [totalRows=rows]
 * @returns {Float32Array}
 */
function generateSpectrogram(rows, cols, rowOffset = 0, totalRows = rows) {
  const data = new Float32Array(rows * cols);
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const time = c / cols;
      const freq = (r + rowOffset) / totalRows;
      data[r * cols + c] =
        Math.sin(time * 30) * Math.cos(freq * 15) +
        Math.sin((time + freq) * 20) * 0.5 +
        Math.cos(freq * 40) * 0.3 * Math.sin(time * 10);
    }
  }
  return data;
}

/** @type {LeibnizFast | null} */
let viewer = null;

/**
 * Create a static chart with fixed X and Y axis ranges.
 */
async function createStaticChart() {
  if (viewer) viewer.destroy();

  viewer = await LeibnizFast.create(canvas, {
    colormap: colormapSelect.value,
    debug: true,
    chart: {
      title: 'Spectrogram Analysis',
      xAxis: { label: 'Time', unit: 's', min: 0, max: 10 },
      yAxis: { label: 'Frequency', unit: 'Hz', min: 0, max: 22050 },
    },
  });

  const data = generateSpectrogram(ROWS, COLS);
  viewer.setData(data, { rows: ROWS, cols: COLS });

  viewer.onHover((row, col, value) => {
    tooltip.style.display = 'block';
    tooltip.textContent = `[${row}, ${col}] = ${value.toFixed(4)}`;
  });
}

/**
 * Generate a single column of spectrogram data.
 *
 * @param {number} rows - Number of spatial samples (Y axis)
 * @param {number} colIndex - Global column index (time step)
 * @param {number} totalCols - Total columns for normalization
 * @returns {Float32Array}
 */
function generateColumn(rows, colIndex, totalCols) {
  const col = new Float32Array(rows);
  const time = colIndex / totalCols;
  for (let r = 0; r < rows; r++) {
    const freq = r / rows;
    col[r] =
      Math.sin(time * 30) * Math.cos(freq * 15) +
      Math.sin((time + freq) * 20) * 0.5 +
      Math.cos(freq * 40) * 0.3 * Math.sin(time * 10);
  }
  return col;
}

/**
 * Create a streaming chart where columns arrive right-to-left like a
 * waterfall. The X axis auto-scrolls to track the current time window.
 */
async function createStreamingChart() {
  if (viewer) viewer.destroy();

  viewer = await LeibnizFast.create(canvas, {
    colormap: colormapSelect.value,
    debug: true,
    chart: {
      title: 'Live Spectrogram (Streaming)',
      xAxis: { label: 'Time', unit: 's', unitsPerCell: 0.01 },
      yAxis: { label: 'Frequency', unit: 'Hz', min: 0, max: 22050 },
    },
  });

  // Sliding window buffer: ROWS × COLS, new columns enter on the right
  const buffer = new Float32Array(ROWS * COLS);
  let totalColsReceived = 0;
  const totalColsToSend = STREAM_TOTAL_FRAMES * STREAM_CHUNK_COLS;

  /**
   * Shift every row left by `n` columns, then write new column data
   * at the right edge.
   */
  function pushColumns(newCols) {
    // Shift each row left
    for (let r = 0; r < ROWS; r++) {
      const rowStart = r * COLS;
      buffer.copyWithin(rowStart, rowStart + newCols.length, rowStart + COLS);
    }
    // Write new columns at the right edge
    for (let c = 0; c < newCols.length; c++) {
      const col = newCols[c];
      const dstColIdx = COLS - newCols.length + c;
      for (let r = 0; r < ROWS; r++) {
        buffer[r * COLS + dstColIdx] = col[r];
      }
    }
  }

  function streamFrame() {
    if (!viewer || totalColsReceived >= totalColsToSend) return;

    // Generate a batch of new columns
    const newCols = [];
    for (let i = 0; i < STREAM_CHUNK_COLS; i++) {
      newCols.push(
        generateColumn(ROWS, totalColsReceived + i, totalColsToSend),
      );
    }
    totalColsReceived += STREAM_CHUNK_COLS;

    pushColumns(newCols);

    // Pass xOffset so the library knows the total columns received,
    // enabling the X axis to show a sliding time window.
    viewer.setData(buffer, {
      rows: ROWS,
      cols: COLS,
      xOffset: totalColsReceived,
    });

    if (totalColsReceived < totalColsToSend) {
      setTimeout(streamFrame, STREAM_DELAY_MS);
    }
  }

  streamFrame();

  viewer.onHover((row, col, value) => {
    tooltip.style.display = 'block';
    tooltip.textContent = `[${row}, ${col}] = ${value.toFixed(4)}`;
  });
}

async function load() {
  const mode = modeSelect.value;
  if (mode === 'static') {
    await createStaticChart();
  } else {
    await createStreamingChart();
  }
}

// Event listeners
reloadBtn.addEventListener('click', load);
modeSelect.addEventListener('change', load);
colormapSelect.addEventListener('change', () => {
  if (viewer) viewer.setColormap(colormapSelect.value);
});

canvas.addEventListener('mousemove', (e) => {
  tooltip.style.left = `${e.clientX + 12}px`;
  tooltip.style.top = `${e.clientY + 12}px`;
});

canvas.addEventListener('mouseleave', () => {
  tooltip.style.display = 'none';
});

// Initial load
load().catch(console.error);
