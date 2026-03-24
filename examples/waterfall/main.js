/**
 * main.js — WebSocket client for the waterfall example.
 *
 * Connects to the Python bridge (bridge.py) over WebSocket, receives
 * binary column data from the C++ generator, and renders a scrolling
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
import { LeibnizFast } from '../../dist/index.js';

// ---- DOM refs ----------------------------------------------------------
const canvas           = document.getElementById('canvas');
const colormapSelect   = document.getElementById('colormap');
const rangeMinInput    = document.getElementById('range-min');
const rangeMaxInput    = document.getElementById('range-max');
const debugCheckbox    = document.getElementById('debug');
const statusBadge      = document.getElementById('status-badge');
const statusText       = document.getElementById('status-text');
const fpsCounter       = document.getElementById('fps-counter');
const dataRateEl       = document.getElementById('data-rate');
const tooltip          = document.getElementById('tooltip');
const errorBanner      = document.getElementById('error-banner');

// Generator parameter inputs
const genSpatialStartInput  = document.getElementById('gen-spatial-start');
const genSpatialEndInput    = document.getElementById('gen-spatial-end');
const genSamplingRateInput  = document.getElementById('gen-sampling-rate');
const genRepetitionRateInput = document.getElementById('gen-repetition-rate');
const genSpatialDsInput     = document.getElementById('gen-spatial-downsample');
const genTimeBufferInput    = document.getElementById('gen-time-buffer');
const genTimeWindowInput    = document.getElementById('gen-time-window');
const genSaveButton         = document.getElementById('gen-save');
const genStatusEl           = document.getElementById('gen-status');

// ---- Constants ---------------------------------------------------------

const WATERFALL_MAGIC    = 0x4C465A10;
const HEADER_BYTES       = 16;  // 4 × uint32
const PROPAGATION_VELOCITY = 2.0432e8;  // m/s (fixed physical constant)
const BRIDGE_MAX_ROWS    = 65536;

const WS_URL       = 'ws://localhost:8765';
const RECONNECT_MS = 2000;
const FPS_WINDOW   = 10;

// ---- Generator parameters (mirrors generator.cpp defaults) -------------

/** @typedef {{ spatialStart: number, spatialEnd: number, samplingRate: number, repetitionRate: number, spatialDownsample: number, timeBuffer: number }} GenParams */

/**
 * Read initial generator parameters from DOM inputs so the chart config
 * always reflects the values the user sees, not stale JS constants.
 * @returns {GenParams}
 */
function readInitialGenParams() {
  return {
    spatialStart:      parseFloat(genSpatialStartInput.value)   || 10000,
    spatialEnd:        parseFloat(genSpatialEndInput.value)     || 15000,
    samplingRate:      parseInt(genSamplingRateInput.value)     || 400,
    repetitionRate:    parseInt(genRepetitionRateInput.value)   || 1000,
    spatialDownsample: parseInt(genSpatialDsInput.value)        || 5,
    timeBuffer:        parseFloat(genTimeBufferInput.value)     || 0.15,
  };
}

/** @type {GenParams} */
let genParams = readInitialGenParams();

/**
 * Compute spatial row count from generator parameters.
 * @param {GenParams} p
 * @returns {number}
 */
function computeSpatialRows(p) {
  const extent        = p.spatialEnd - p.spatialStart;
  const roundTripTime = 2.0 * extent / PROPAGATION_VELOCITY;
  const pointsPerSeg  = Math.round(p.samplingRate * 1e6 * roundTripTime);
  return Math.ceil(pointsPerSeg / p.spatialDownsample);
}

/**
 * Compute derived stats from generator parameters.
 * @param {GenParams} p
 * @returns {{ rows: number, colsPerMsg: number, rateMbs: number }}
 */
function computeStats(p) {
  const rows       = computeSpatialRows(p);
  const colsPerMsg = Math.round(p.repetitionRate * p.timeBuffer);
  const rateMbs    = (rows * colsPerMsg * 4 * (1.0 / p.timeBuffer)) / 1e6;
  return { rows, colsPerMsg, rateMbs };
}

/**
 * Update the computed stats display in the UI.
 * @param {GenParams} p
 */
function updateStats(p) {
  const { rows, colsPerMsg, rateMbs } = computeStats(p);
  document.getElementById('stat-rows').textContent = `Rows: ${rows.toLocaleString()}`;
  document.getElementById('stat-cols').textContent = `Cols/msg: ${colsPerMsg.toLocaleString()}`;
  document.getElementById('stat-rate').textContent = `Rate: ${rateMbs.toFixed(1)} MB/s`;
}

// ---- State -------------------------------------------------------------
/** @type {LeibnizFast|null} */
let viewer = null;
/** Total columns received from the WebSocket since last buffer reset. */
let totalColsReceived = 0;
/**
 * When true, discard incoming messages whose row count doesn't match the
 * expected value from genParams. This prevents stale messages from a
 * still-dying old generator from reverting the buffer dimensions after Save.
 */
let pendingRestart = false;
/** @type {WebSocket|null} */
let ws = null;
let reconnectTimer = null;
let debugEnabled = debugCheckbox.checked;

// FPS tracking
const frameTimes = [];

// Data rate tracking (bytes received in a sliding 1-second window)
let bytesThisSecond = 0;
let lastRateUpdate = performance.now();

// Inter-message arrival tracking for debug timing
let tLastMessage = 0;  // performance.now() of the last processMessage call

// Waterfall buffer
let displayCols = Math.round(parseFloat(genTimeWindowInput.value) * genParams.repetitionRate);
let rows = computeSpatialRows(genParams);

// ---- Waterfall Buffer --------------------------------------------------

class WaterfallBuffer {
  /**
   * Column-major ring buffer — no per-frame shifting.
   *
   * Layout: data[col * rows .. (col+1) * rows] = all rows for column col.
   * New columns are written at ringCursor (mod cols) and the cursor advances.
   * The GPU texture mirrors this ring layout; the render shader applies
   * ring_offset to unwrap it visually so oldest data appears on the left.
   *
   * @param {number} rows - spatial samples per column
   * @param {number} cols - display width (time window)
   */
  constructor(rows, cols) {
    this.rows = rows;
    this.cols = cols;
    // Column-major: column c occupies data[c*rows .. (c+1)*rows]
    this.data = new Float32Array(rows * cols);
    /** Next write position in [0, cols). Mirrors the GPU texture ring cursor. */
    this.ringCursor = 0;
  }

  /**
   * Write new columns into the ring buffer without any data shifting.
   * Cost: O(rows × newCols) — independent of total display width.
   *
   * @param {Float32Array} newData - column-major: column i = newData[i*rows .. (i+1)*rows]
   * @param {number} newCols - number of new columns in newData
   */
  pushColumns(newData, newCols) {
    const { rows: r, cols: c, data } = this;

    // Clamp: if more new columns than display width, keep only the most recent
    const effectiveCols = Math.min(newCols, c);
    const srcColOffset = newCols - effectiveCols;

    for (let i = 0; i < effectiveCols; i++) {
      const dstCol = (this.ringCursor + i) % c;
      const srcStart = (srcColOffset + i) * r;
      // Zero-copy column write: one TypedArray.set() = one memcpy
      data.set(newData.subarray(srcStart, srcStart + r), dstCol * r);
    }

    this.ringCursor = (this.ringCursor + effectiveCols) % c;
  }
}

/** @type {WaterfallBuffer|null} */
let buffer = null;
let dirty = false;
/** Number of new columns added since last render (for scrolled update). */
let pendingNewCols = 0;

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

  const tRecv = performance.now();
  const gapMs = tLastMessage > 0 ? tRecv - tLastMessage : 0;
  tLastMessage = tRecv;

  const h = parseHeader(buf);
  if (!h) return;

  // After a Save+restart, discard stale messages from the old generator
  // until we receive one that matches the expected row count.
  if (pendingRestart) {
    const expectedRows = computeSpatialRows(genParams);
    if (h.rows !== expectedRows) {
      return;  // stale message from old generator — discard
    }
    pendingRestart = false;
  }

  const tAfterParse = performance.now();

  // If generator rows changed (e.g. after restart), recreate buffer
  if (h.rows !== buffer.rows) {
    rows = h.rows;
    buffer = new WaterfallBuffer(rows, displayCols);
    updateStats(genParams);
  }

  // Extract column data from after the header
  const colData = new Float32Array(buf, HEADER_BYTES, h.rows * h.newCols);

  buffer.pushColumns(colData, h.newCols);
  totalColsReceived += h.newCols;
  pendingNewCols += h.newCols;
  dirty = true;

  const tAfterPush = performance.now();

  updateDataRate(buf.byteLength);

  if (debugEnabled) {
    const parseMs = tAfterParse - tRecv;
    const pushMs  = tAfterPush - tAfterParse;
    console.log(
      `[perf] msg_id=${h.msgId}` +
      `  gap=${gapMs.toFixed(2)}ms` +
      `  parse=${parseMs.toFixed(2)}ms` +
      `  push=${pushMs.toFixed(2)}ms` +
      `  bytes=${buf.byteLength}`
    );
  }
}

// ---- Render loop (decoupled from network) ------------------------------

function renderLoop() {
  if (dirty && viewer && buffer) {
    // Use scrolled update: GPU texture shifts left, only new columns are
    // colormapped. Falls back to full setData if no range is set or on
    // the first frame.
    const newCols = Math.min(pendingNewCols, buffer.cols);
    viewer.setDataScrolled(buffer.data, {
      rows: buffer.rows,
      cols: buffer.cols,
      newCols,
      xOffset: totalColsReceived,
    });
    pendingNewCols = 0;
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

// ---- Main --------------------------------------------------------------

/**
 * Build a chart config object from the current generator parameters.
 * @param {GenParams} p
 * @returns {object}
 */
function buildChartConfig(p) {
  return {
    title: 'Live Waterfall',
    xAxis: { label: 'Time', unit: 's', unitsPerCell: 1 / p.repetitionRate },
    yAxis: { label: 'Depth', unit: 'm', min: p.spatialStart, max: p.spatialEnd },
  };
}

/**
 * Read and validate the generator parameter inputs.
 * Returns the validated params or null if invalid.
 * @returns {GenParams|null}
 */
function readGenParamInputs() {
  const spatialStart     = parseFloat(genSpatialStartInput.value);
  const spatialEnd       = parseFloat(genSpatialEndInput.value);
  const samplingRate     = parseInt(genSamplingRateInput.value);
  const repetitionRate   = parseInt(genRepetitionRateInput.value);
  const spatialDownsample = parseInt(genSpatialDsInput.value);
  const timeBuffer       = parseFloat(genTimeBufferInput.value);

  if (!isFinite(spatialStart) || spatialStart < 0 || spatialStart > 1e6) {
    showError('Spatial Start must be 0–1,000,000 m');
    return null;
  }
  if (!isFinite(spatialEnd) || spatialEnd < 1 || spatialEnd > 1e6) {
    showError('Spatial End must be 1–1,000,000 m');
    return null;
  }
  if (spatialEnd <= spatialStart) {
    showError('Spatial End must be greater than Spatial Start');
    return null;
  }
  if (!isFinite(samplingRate) || samplingRate < 1 || samplingRate > 10000) {
    showError('Sampling Rate must be 1–10,000 MHz');
    return null;
  }
  if (!isFinite(repetitionRate) || repetitionRate < 1 || repetitionRate > 1000000) {
    showError('Repetition Rate must be 1–1,000,000 Hz');
    return null;
  }
  if (!isFinite(spatialDownsample) || spatialDownsample < 1 || spatialDownsample > 1000) {
    showError('Spatial Downsample must be 1–1,000');
    return null;
  }
  if (!isFinite(timeBuffer) || timeBuffer < 0.001 || timeBuffer > 60) {
    showError('Time Buffer must be 0.001–60 s');
    return null;
  }

  return { spatialStart, spatialEnd, samplingRate, repetitionRate, spatialDownsample, timeBuffer };
}

/**
 * Send a restart command to the bridge with new generator parameters,
 * then reset local state (buffer, chart axes, stats).
 */
function saveParams() {
  const params = readGenParamInputs();
  if (!params) return;

  const newRows = computeSpatialRows(params);
  if (newRows > BRIDGE_MAX_ROWS) {
    showError(
      `Computed rows (${newRows.toLocaleString()}) exceeds bridge limit (${BRIDGE_MAX_ROWS.toLocaleString()}). ` +
      `Increase Spatial Downsample or reduce the spatial range.`
    );
    return;
  }

  genParams = params;
  pendingRestart = true;

  // Recompute displayCols from time window (seconds) and new repetition rate
  const timeWindowSec = parseFloat(genTimeWindowInput.value) || 1.0;
  displayCols = Math.max(1, Math.round(timeWindowSec * params.repetitionRate));

  // Send restart message to bridge — bridge will kill + respawn the generator
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({
      type: 'restart',
      spatialStart:      params.spatialStart,
      spatialEnd:        params.spatialEnd,
      samplingRate:      params.samplingRate,
      repetitionRate:    params.repetitionRate,
      spatialDownsample: params.spatialDownsample,
      timeBuffer:        params.timeBuffer,
    }));
    genStatusEl.textContent = 'Restart sent…';
    setTimeout(() => { genStatusEl.textContent = ''; }, 3000);
  } else {
    genStatusEl.textContent = 'Not connected — parameters saved locally';
    setTimeout(() => { genStatusEl.textContent = ''; }, 3000);
  }

  // Reset buffer and counters for new geometry
  rows = newRows;
  buffer = new WaterfallBuffer(rows, displayCols);
  totalColsReceived = 0;

  if (viewer) {
    viewer.setData(buffer.data, { rows, cols: displayCols, xOffset: 0 });
    viewer.setChart(buildChartConfig(params));
  }

  updateStats(params);
}

async function main() {
  viewer = await LeibnizFast.create(canvas, {
    colormap: colormapSelect.value,
    debug: debugEnabled,
    chart: buildChartConfig(genParams),
  });

  // Initialize waterfall buffer and do an initial setData so the canvas isn't blank
  buffer = new WaterfallBuffer(rows, displayCols);
  viewer.setData(buffer.data, { rows, cols: displayCols, xOffset: 0 });
  applyRange();
  updateStats(genParams);

  // Start decoupled render loop
  requestAnimationFrame(renderLoop);

  // ---- Hover tooltip ---------------------------------------------------
  viewer.onHover((row, col, value) => {
    tooltip.style.display = 'block';
    tooltip.textContent = `[${row}, ${col}] = ${value.toFixed(4)}`;
  });

  canvas.addEventListener('mousemove', (e) => {
    tooltip.style.left = `${e.clientX + 12}px`;
    tooltip.style.top  = `${e.clientY + 12}px`;
  });

  canvas.addEventListener('mouseleave', () => {
    tooltip.style.display = 'none';
  });

  // ---- Controls --------------------------------------------------------

  colormapSelect.addEventListener('change', () => {
    const t = debugEnabled ? performance.now() : 0;
    viewer.setColormap(colormapSelect.value);
    if (debugEnabled) console.log(`[perf] setColormap: ${(performance.now() - t).toFixed(2)}ms`);
  });

  genSaveButton.addEventListener('click', saveParams);

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
