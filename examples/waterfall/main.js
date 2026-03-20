/**
 * main.js — WebSocket client for the DAS waterfall example.
 *
 * Connects to the Python bridge (bridge.py) over WebSocket, receives
 * binary column data from the C++ DAS generator, and renders a scrolling
 * waterfall plot using LeibnizFast.
 *
 * Protocol — waterfall v1 (magic 0x4C465A10, 16-byte header):
 *   Offset  0: magic      = 0x4C465A10
 *   Offset  4: rows       (spatial samples per column)
 *   Offset  8: new_cols   (columns in this message)
 *   Offset 12: msg_id     (monotonic counter)
 *   Offset 16: float32[rows × new_cols]  column data
 *
 * Waterfall logic:
 *   - Pre-allocate Float32Array(rows × displayCols) as a sliding window
 *   - On each message: shift every row left by new_cols, write new data at right edge
 *   - Render at rAF rate (decoupled from network receive)
 */
import init, { LeibnizFast } from '../../pkg/leibniz_fast.js';

// ---- DOM refs ----------------------------------------------------------
const canvas           = document.getElementById('canvas');
const colormapSelect   = document.getElementById('colormap');
const spatialDsInput   = document.getElementById('spatial-downsample');
const displayColsSel   = document.getElementById('display-cols');
const rangeMinInput  = document.getElementById('range-min');
const rangeMaxInput  = document.getElementById('range-max');
const debugCheckbox  = document.getElementById('debug');
const statusBadge    = document.getElementById('status-badge');
const statusText     = document.getElementById('status-text');
const fpsCounter     = document.getElementById('fps-counter');
const dataRateEl     = document.getElementById('data-rate');
const tooltip        = document.getElementById('tooltip');
const errorBanner    = document.getElementById('error-banner');

// ---- Constants ---------------------------------------------------------

const WATERFALL_MAGIC = 0x4C465A10;
const HEADER_BYTES    = 16;  // 4 × uint32

const WS_URL       = 'ws://localhost:8765';
const RECONNECT_MS = 2000;
const FPS_WINDOW   = 10;

// ---- DAS Physics Constants (must match generator CLI defaults) ----------
// NOTE: Keep these in sync with generator.cpp defaults.

const C_LIGHT            = 3e8;            // m/s
const N_FIBER            = 1.4682;         // silica fiber refractive index
const V_FIBER            = C_LIGHT / N_FIBER;  // ~2.0432e8 m/s
const FIBER_START_M      = 10000;          // m  (matches --fiber-start default)
const FIBER_END_M        = 20000;          // m  (matches --fiber-end default)
const SAMPLING_RATE_MHZ  = 400;            // MHz (matches --sampling-rate default)
const REPETITION_RATE_HZ = 10000;          // Hz (matches --repetition-rate default)
const TIME_BUFFER_S      = 0.2;            // s  (matches --time-buffer default)
const BRIDGE_MAX_ROWS    = 65536;

/**
 * Compute spatial row count from spatial downsampling factor.
 * @param {number} ds - integer downsampling step (>= 1)
 * @returns {number}
 */
function computeDasRows(ds) {
  const fiberLength   = FIBER_END_M - FIBER_START_M;
  const roundTripTime = 2.0 * fiberLength / V_FIBER;
  const pointsPerSeg  = Math.round(SAMPLING_RATE_MHZ * 1e6 * roundTripTime);
  return Math.ceil(pointsPerSeg / ds);
}

/**
 * Compute derived DAS stats for a given downsampling factor.
 * @param {number} ds
 * @returns {{ rows: number, colsPerMsg: number, rateMbs: number }}
 */
function computeDasStats(ds) {
  const rows       = computeDasRows(ds);
  const colsPerMsg = Math.round(REPETITION_RATE_HZ * TIME_BUFFER_S);
  const rateMbs    = (rows * colsPerMsg * 4 * (1.0 / TIME_BUFFER_S)) / 1e6;
  return { rows, colsPerMsg, rateMbs };
}

/**
 * Update the DAS computed stats display in the UI.
 * @param {number} ds
 */
function updateDasStats(ds) {
  const { rows, colsPerMsg, rateMbs } = computeDasStats(ds);
  document.getElementById('stat-rows').textContent = `Rows: ${rows.toLocaleString()}`;
  document.getElementById('stat-cols').textContent = `Cols/msg: ${colsPerMsg.toLocaleString()}`;
  document.getElementById('stat-rate').textContent = `Rate: ${rateMbs.toFixed(1)} MB/s`;
}

// ---- State -------------------------------------------------------------
/** @type {LeibnizFast|null} */
let viewer = null;
/** @type {WebSocket|null} */
let ws = null;
let reconnectTimer = null;
let debugEnabled = debugCheckbox.checked;

// FPS tracking
const frameTimes = [];

// Data rate tracking (bytes received in a sliding 1-second window)
let bytesThisSecond = 0;
let lastRateUpdate = performance.now();

// Waterfall buffer
let displayCols = parseInt(displayColsSel.value);
let spatialDownsample = parseInt(spatialDsInput.value);
let rows = computeDasRows(spatialDownsample);

// ---- Waterfall Buffer --------------------------------------------------

class WaterfallBuffer {
  /**
   * @param {number} rows - spatial samples per column
   * @param {number} cols - display width (time window)
   */
  constructor(rows, cols) {
    this.rows = rows;
    this.cols = cols;
    this.data = new Float32Array(rows * cols);
  }

  /**
   * Shift all rows left by `newCols` and write new column data at the right edge.
   * @param {Float32Array} newData - float32[rows × newCols], columns contiguous
   * @param {number} newCols - number of new columns
   */
  pushColumns(newData, newCols) {
    const { rows: r, cols: c, data } = this;

    // Clamp: if more new columns than display width, only keep the rightmost
    const effectiveCols = Math.min(newCols, c);
    const dataOffset = (newCols - effectiveCols) * r;

    if (effectiveCols >= c) {
      // New data fills entire buffer — just copy the rightmost displayCols columns
      for (let col = 0; col < c; col++) {
        const srcCol = dataOffset / r + col;
        for (let row = 0; row < r; row++) {
          data[row * c + col] = newData[srcCol * r + row];
        }
      }
      return;
    }

    // Shift each row left by effectiveCols
    for (let row = 0; row < r; row++) {
      const rowStart = row * c;
      data.copyWithin(rowStart, rowStart + effectiveCols, rowStart + c);
    }

    // Write new columns at the right edge
    // newData layout: columns are contiguous blocks of `rows` floats
    for (let col = 0; col < effectiveCols; col++) {
      const srcOffset = dataOffset + col * r;
      const dstColIdx = c - effectiveCols + col;
      for (let row = 0; row < r; row++) {
        data[row * c + dstColIdx] = newData[srcOffset + row];
      }
    }
  }
}

/** @type {WaterfallBuffer|null} */
let buffer = null;
let dirty = false;

// ---- Utilities ---------------------------------------------------------

function showError(msg) {
  errorBanner.textContent = msg;
  errorBanner.style.display = 'block';
  clearTimeout(errorBanner._timer);
  errorBanner._timer = setTimeout(() => {
    errorBanner.style.display = 'none';
  }, 6000);
}

function setStatus(state) {
  statusBadge.className = `status-badge ${state}`;
  statusText.textContent =
    state === 'connecting'   ? 'Connecting\u2026' :
    state === 'connected'    ? 'Connected'         :
                               'Disconnected';
}

function updateFps() {
  frameTimes.push(performance.now());
  if (frameTimes.length > FPS_WINDOW) frameTimes.shift();
  if (frameTimes.length >= 2) {
    const elapsed = frameTimes[frameTimes.length - 1] - frameTimes[0];
    const fps = ((frameTimes.length - 1) / elapsed) * 1000;
    fpsCounter.textContent = `${fps.toFixed(1)} FPS`;
  }
}

function updateDataRate(bytes) {
  bytesThisSecond += bytes;
  const now = performance.now();
  const elapsed = now - lastRateUpdate;
  if (elapsed >= 1000) {
    const mbps = (bytesThisSecond / elapsed) * 1000 / 1e6;
    dataRateEl.textContent = `${mbps.toFixed(1)} MB/s`;
    bytesThisSecond = 0;
    lastRateUpdate = now;
  }
}

function applyRange() {
  const mn = parseFloat(rangeMinInput.value);
  const mx = parseFloat(rangeMaxInput.value);
  if (viewer && isFinite(mn) && isFinite(mx) && mx > mn) {
    viewer.setRange(mn, mx);
  }
}

// ---- Protocol parsing --------------------------------------------------

/**
 * Parse a waterfall message header.
 * @param {ArrayBuffer} buf
 * @returns {{ rows: number, newCols: number, msgId: number }|null}
 */
function parseHeader(buf) {
  if (buf.byteLength < HEADER_BYTES) return null;

  const view  = new DataView(buf);
  const magic = view.getUint32(0, true);

  if (magic !== WATERFALL_MAGIC) {
    console.warn(`[waterfall] Unknown magic: 0x${magic.toString(16)}`);
    return null;
  }

  const msgRows  = view.getUint32(4, true);
  const newCols  = view.getUint32(8, true);
  const msgId    = view.getUint32(12, true);

  const expectedBytes = HEADER_BYTES + msgRows * newCols * 4;
  if (buf.byteLength < expectedBytes) {
    console.warn(`[waterfall] Message too short: ${buf.byteLength} < ${expectedBytes}`);
    return null;
  }

  return { rows: msgRows, newCols, msgId };
}

// ---- Message processing ------------------------------------------------

function processMessage(buf) {
  if (!viewer || !buffer) return;

  const h = parseHeader(buf);
  if (!h) return;

  // If generator rows changed (e.g. via resize), recreate buffer
  if (h.rows !== buffer.rows) {
    rows = h.rows;
    buffer = new WaterfallBuffer(rows, displayCols);
    updateDasStats(spatialDownsample);
  }

  // Extract column data from after the header
  const colData = new Float32Array(buf, HEADER_BYTES, h.rows * h.newCols);

  buffer.pushColumns(colData, h.newCols);
  dirty = true;

  updateDataRate(buf.byteLength);

  if (debugEnabled) {
    console.log(`[waterfall] msg_id=${h.msgId} rows=${h.rows} newCols=${h.newCols} bytes=${buf.byteLength}`);
  }
}

// ---- Render loop (decoupled from network) ------------------------------

function renderLoop() {
  if (dirty && viewer && buffer) {
    if (debugEnabled) {
      const t0 = performance.now();
      viewer.setData(buffer.data, buffer.rows, buffer.cols);
      console.log(`[perf] setData (${buffer.rows}x${buffer.cols}): ${(performance.now() - t0).toFixed(2)}ms`);
    } else {
      viewer.setData(buffer.data, buffer.rows, buffer.cols);
    }
    dirty = false;
    updateFps();
  }
  requestAnimationFrame(renderLoop);
}

// ---- WebSocket lifecycle -----------------------------------------------

function connect() {
  clearTimeout(reconnectTimer);
  setStatus('connecting');

  try {
    ws = new WebSocket(WS_URL);
  } catch (e) {
    showError(`WebSocket error: ${e.message}`);
    scheduleReconnect();
    return;
  }

  ws.binaryType = 'arraybuffer';

  ws.addEventListener('open', () => {
    setStatus('connected');
    frameTimes.length = 0;
    fpsCounter.textContent = '-- FPS';
    bytesThisSecond = 0;
    lastRateUpdate = performance.now();
    dataRateEl.textContent = '-- MB/s';
    // Sync row count with generator on reconnect
    sendResize(rows);
  });

  ws.addEventListener('message', (e) => {
    processMessage(/** @type {ArrayBuffer} */ (e.data));
  });

  ws.addEventListener('error', () => {
    // 'error' is always followed by 'close'; reconnect logic lives there.
  });

  ws.addEventListener('close', () => {
    ws = null;
    setStatus('disconnected');
    scheduleReconnect();
  });
}

function scheduleReconnect() {
  clearTimeout(reconnectTimer);
  reconnectTimer = setTimeout(connect, RECONNECT_MS);
}

function sendResize(newRows) {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: 'resize', rows: newRows }));
  }
}

// ---- Main --------------------------------------------------------------

async function main() {
  const t0 = performance.now();
  await init();
  if (debugEnabled) console.log(`[perf] WASM init: ${(performance.now() - t0).toFixed(2)}ms`);

  const t1 = performance.now();
  viewer = await LeibnizFast.create(canvas, colormapSelect.value, debugEnabled);
  if (debugEnabled) console.log(`[perf] LeibnizFast.create: ${(performance.now() - t1).toFixed(2)}ms`);

  // Initialize waterfall buffer and do an initial setData so the canvas isn't blank
  buffer = new WaterfallBuffer(rows, displayCols);
  viewer.setData(buffer.data, rows, displayCols);
  applyRange();
  updateDasStats(spatialDownsample);

  // Start decoupled render loop
  requestAnimationFrame(renderLoop);

  // ---- Hover tooltip ---------------------------------------------------
  viewer.onHover((row, col, value) => {
    tooltip.style.display = 'block';
    tooltip.textContent = `[${row}, ${col}] = ${value.toFixed(4)}`;
  });

  // ---- Pan / zoom interactions -----------------------------------------
  canvas.addEventListener('mousedown', (e) => {
    const rect = canvas.getBoundingClientRect();
    viewer.onMouseDown(e.clientX - rect.left, e.clientY - rect.top);
  });

  canvas.addEventListener('mousemove', (e) => {
    const rect = canvas.getBoundingClientRect();
    viewer.onMouseMove(e.clientX - rect.left, e.clientY - rect.top);
    tooltip.style.left = `${e.clientX + 12}px`;
    tooltip.style.top  = `${e.clientY + 12}px`;
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

  // ---- Controls --------------------------------------------------------

  colormapSelect.addEventListener('change', () => {
    const t = debugEnabled ? performance.now() : 0;
    viewer.setColormap(colormapSelect.value);
    if (debugEnabled) console.log(`[perf] setColormap: ${(performance.now() - t).toFixed(2)}ms`);
  });

  spatialDsInput.addEventListener('change', () => {
    const ds = Math.max(1, parseInt(spatialDsInput.value) || 1);
    spatialDsInput.value = ds;
    spatialDownsample = ds;
    rows = computeDasRows(ds);
    buffer = new WaterfallBuffer(rows, displayCols);
    viewer.setData(buffer.data, rows, displayCols);
    updateDasStats(ds);
    if (rows <= BRIDGE_MAX_ROWS) {
      sendResize(rows);
    } else {
      showError(
        `Computed rows (${rows.toLocaleString()}) exceeds bridge limit (${BRIDGE_MAX_ROWS.toLocaleString()}). ` +
        `Restart generator with --spatial-downsample ${ds} to apply.`
      );
    }
  });

  displayColsSel.addEventListener('change', () => {
    displayCols = parseInt(displayColsSel.value);
    buffer = new WaterfallBuffer(rows, displayCols);
    viewer.setData(buffer.data, rows, displayCols);
  });

  rangeMinInput.addEventListener('input', applyRange);
  rangeMaxInput.addEventListener('input', applyRange);
  rangeMinInput.addEventListener('change', applyRange);
  rangeMaxInput.addEventListener('change', applyRange);

  debugCheckbox.addEventListener('change', () => {
    debugEnabled = debugCheckbox.checked;
    console.log(`[perf] Debug timing ${debugEnabled ? 'enabled' : 'disabled'}`);
  });

  // ---- Start WebSocket connection --------------------------------------
  connect();
}

main().catch(console.error);
