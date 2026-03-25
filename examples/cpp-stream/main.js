/**
 * main.js — WebSocket client for the cpp-stream example.
 *
 * Connects to the Python bridge (bridge.py) over WebSocket, receives
 * binary frames produced by the C++ wave simulator, and feeds each frame
 * to LeibnizFast for live GPU rendering with chart-style axes.
 *
 * Render is decoupled from the network via requestAnimationFrame: network
 * messages accumulate into a reusable Float32Array and set a dirty flag;
 * the rAF loop calls setData() only once per display frame.
 *
 * Protocol v1 — plain float32 chunks (magic 0x4C465A01, 32-byte header):
 *   Offset  0: magic        = 0x4C465A01
 *   Offset  4: total_rows
 *   Offset  8: cols
 *   Offset 12: frame_id
 *   Offset 16: chunk_index
 *   Offset 20: total_chunks
 *   Offset 24: row_start
 *   Offset 28: chunk_rows
 *   Offset 32: float32[chunk_rows x cols]  row-major grid data
 *
 * Resize flow (browser -> C++ generator):
 *   browser sends JSON text  {"type":"resize","size":N}
 *   bridge.py parses it and forwards a 4-byte uint32 to the generator
 *   generator reinitializes the simulation at NxN and starts sending new frames
 */
import { LeibnizFast } from '../../dist/index.js';

// ---- DOM refs ------------------------------------------------------------
const canvas         = document.getElementById('canvas');
const colormapSelect = document.getElementById('colormap');
const sizeSelect     = document.getElementById('size');
const rangeMinInput  = document.getElementById('range-min');
const rangeMaxInput  = document.getElementById('range-max');
const debugCheckbox  = document.getElementById('debug');
const statusBadge    = document.getElementById('status-badge');
const statusText     = document.getElementById('status-text');
const fpsCounter     = document.getElementById('fps-counter');
const dataRateEl     = document.getElementById('data-rate');
const tooltip        = document.getElementById('tooltip');
const errorBanner    = document.getElementById('error-banner');

// Axis config inputs
const xLabelInput    = document.getElementById('x-label');
const xUnitInput     = document.getElementById('x-unit');
const xMinInput      = document.getElementById('x-min');
const xMaxInput      = document.getElementById('x-max');
const yLabelInput    = document.getElementById('y-label');
const yUnitInput     = document.getElementById('y-unit');
const yMinInput      = document.getElementById('y-min');
const yMaxInput      = document.getElementById('y-max');
const valueUnitInput = document.getElementById('value-unit');

// ---- Constants -----------------------------------------------------------

const CHUNK_MAGIC        = 0x4C465A01;
const CHUNK_HEADER_BYTES = 32;  // 8 x uint32

const WS_URL       = 'ws://localhost:8765';
const RECONNECT_MS = 2000;
const FPS_WINDOW   = 10;

// ---- State ---------------------------------------------------------------
/** @type {LeibnizFast|null} */
let viewer = null;
/** @type {WebSocket|null} */
let ws = null;
let reconnectTimer = null;
let debugEnabled = debugCheckbox.checked;

// FPS tracking
const frameTimes = [];

// Data rate tracking
let bytesThisSecond = 0;
let lastRateUpdate = performance.now();

// Frame accumulation (reusable buffer, dirty flag for rAF decoupling)
/** @type {Float32Array|null} */
let frameBuffer = null;
let frameBufferRows = 0;
let frameBufferCols = 0;
let dirty = false;

// Chunk accumulation for multi-chunk frames
let accumFrameId = null;
let accumReceived = 0;
let accumTotalChunks = 0;

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
 * Update the connection status badge.
 * @param {'connecting'|'connected'|'disconnected'} state
 */
function setStatus(state) {
  statusBadge.className = `status-badge ${state}`;
  statusText.textContent =
    state === 'connecting'   ? 'Connecting\u2026' :
    state === 'connected'    ? 'Connected'         :
                               'Disconnected';
}

/**
 * Record a frame render and update the FPS counter (rolling average).
 */
function updateFps() {
  frameTimes.push(performance.now());
  if (frameTimes.length > FPS_WINDOW) frameTimes.shift();
  if (frameTimes.length >= 2) {
    const elapsed = frameTimes[frameTimes.length - 1] - frameTimes[0];
    const fps = ((frameTimes.length - 1) / elapsed) * 1000;
    fpsCounter.textContent = `${fps.toFixed(1)} FPS`;
  }
}

/**
 * Track data rate (bytes/second).
 * @param {number} bytes
 */
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

/**
 * Read min/max from the inputs and apply to the viewer.
 */
function applyRange() {
  const mn = parseFloat(rangeMinInput.value);
  const mx = parseFloat(rangeMaxInput.value);
  if (viewer && isFinite(mn) && isFinite(mx) && mx > mn) {
    viewer.setRange(mn, mx);
  }
}

/**
 * Build a ChartConfig from the axis UI inputs.
 * @returns {object}
 */
function buildChartConfig() {
  return {
    title: 'C++ Wave Simulation',
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
  };
}

// ---- Binary protocol -----------------------------------------------------

/**
 * Parse a v1 chunk message header.
 * Returns null if magic is unrecognised or the buffer is too short.
 *
 * @param {ArrayBuffer} buf
 * @returns {{ totalRows: number, cols: number, frameId: number,
 *             chunkIndex: number, totalChunks: number, rowStart: number,
 *             chunkRows: number }|null}
 */
function parseChunkHeader(buf) {
  if (buf.byteLength < CHUNK_HEADER_BYTES) return null;

  const view  = new DataView(buf);
  const magic = view.getUint32(0, true);

  if (magic !== CHUNK_MAGIC) {
    console.warn(`[cpp-stream] Unknown magic: 0x${magic.toString(16)}`);
    return null;
  }

  const totalRows   = view.getUint32(4,  true);
  const cols        = view.getUint32(8,  true);
  const frameId     = view.getUint32(12, true);
  const chunkIndex  = view.getUint32(16, true);
  const totalChunks = view.getUint32(20, true);
  const rowStart    = view.getUint32(24, true);
  const chunkRows   = view.getUint32(28, true);

  const expectedBytes = CHUNK_HEADER_BYTES + chunkRows * cols * 4;
  if (buf.byteLength < expectedBytes) {
    console.warn(
      `[cpp-stream] Chunk too short: ${buf.byteLength} bytes, ` +
      `expected ${expectedBytes} (chunkRows=${chunkRows} cols=${cols})`,
    );
    return null;
  }
  return { totalRows, cols, frameId, chunkIndex, totalChunks, rowStart, chunkRows };
}

// ---- Frame processing (network -> buffer, decoupled from render) ---------

/**
 * Handle one received binary WebSocket message (a single frame chunk).
 * Accumulates chunks into a reusable Float32Array and sets the dirty flag
 * when a complete frame is ready. Rendering happens in the rAF loop.
 *
 * @param {ArrayBuffer} buf
 */
function processFrame(buf) {
  if (!viewer) return;

  const h = parseChunkHeader(buf);
  if (!h) return;

  // Frame dropping: if a newer frame starts while accumulating, discard the old one.
  if (accumFrameId !== null && h.frameId > accumFrameId) {
    if (debugEnabled) console.log(`[v1] dropping incomplete frame ${accumFrameId} for ${h.frameId}`);
    accumFrameId = null;
  }

  // Skip chunks from old/stale frames
  if (accumFrameId !== null && h.frameId < accumFrameId) return;

  // Start accumulating a new frame
  if (accumFrameId === null || h.frameId !== accumFrameId) {
    if (!frameBuffer || frameBufferRows !== h.totalRows || frameBufferCols !== h.cols) {
      frameBuffer = new Float32Array(h.totalRows * h.cols);
      frameBufferRows = h.totalRows;
      frameBufferCols = h.cols;
    }
    accumFrameId = h.frameId;
    accumReceived = 0;
    accumTotalChunks = h.totalChunks;
  }

  // Copy chunk into the reusable frame buffer
  const chunkData = new Float32Array(buf, CHUNK_HEADER_BYTES, h.chunkRows * h.cols);
  frameBuffer.set(chunkData, h.rowStart * h.cols);
  accumReceived++;

  if (accumReceived === accumTotalChunks) {
    // Full frame ready — mark dirty for the rAF loop
    dirty = true;
    accumFrameId = null;
  }

  updateDataRate(buf.byteLength);
}

// ---- Render loop (decoupled from network) --------------------------------

function renderLoop() {
  if (dirty && viewer && frameBuffer) {
    if (debugEnabled) {
      const t0 = performance.now();
      viewer.setData(frameBuffer, { rows: frameBufferRows, cols: frameBufferCols });
      console.log(`[perf] setData (${frameBufferRows}x${frameBufferCols}): ${(performance.now() - t0).toFixed(2)}ms`);
    } else {
      viewer.setData(frameBuffer, { rows: frameBufferRows, cols: frameBufferCols });
    }
    dirty = false;
    updateFps();
  }
  requestAnimationFrame(renderLoop);
}

// ---- WebSocket lifecycle -------------------------------------------------

/**
 * Open a WebSocket connection and wire up event handlers.
 * Automatically reconnects after RECONNECT_MS on close/error.
 */
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
    accumFrameId = null;
    // Sync the size selector with the generator on reconnect
    sendResize(parseInt(sizeSelect.value));
  });

  ws.addEventListener('message', (e) => {
    processFrame(/** @type {ArrayBuffer} */ (e.data));
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

/** Schedule a reconnect attempt after RECONNECT_MS. */
function scheduleReconnect() {
  clearTimeout(reconnectTimer);
  reconnectTimer = setTimeout(connect, RECONNECT_MS);
}

/**
 * Send a resize command to the C++ generator via bridge.py.
 * @param {number} size
 */
function sendResize(size) {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: 'resize', size }));
  }
}

// ---- Main ----------------------------------------------------------------

async function main() {
  viewer = await LeibnizFast.create(canvas, {
    colormap: colormapSelect.value,
    debug: debugEnabled,
    chart: buildChartConfig(),
  });

  applyRange();

  // Start decoupled render loop
  requestAnimationFrame(renderLoop);

  // ---- Hover tooltip -----------------------------------------------------
  viewer.onHover((info) => {
    tooltip.style.display = 'block';
    tooltip.innerHTML =
      `Y: ${info.y?.toFixed(1) ?? info.row} ${info.yUnit ?? ''}<br>` +
      `X: ${info.x?.toFixed(2) ?? info.col} ${info.xUnit ?? ''}<br>` +
      `Value: ${info.value.toFixed(4)}${info.valueUnit ? ' ' + info.valueUnit : ''}`;
  });

  canvas.addEventListener('mousemove', (e) => {
    tooltip.style.left = `${e.clientX + 12}px`;
    tooltip.style.top  = `${e.clientY + 12}px`;
  });

  canvas.addEventListener('mouseleave', () => {
    tooltip.style.display = 'none';
  });

  // ---- Controls ----------------------------------------------------------

  colormapSelect.addEventListener('change', () => {
    if (viewer) viewer.setColormap(colormapSelect.value);
  });

  sizeSelect.addEventListener('change', () => {
    accumFrameId = null;
    sendResize(parseInt(sizeSelect.value));
  });

  rangeMinInput.addEventListener('input', applyRange);
  rangeMaxInput.addEventListener('input', applyRange);
  rangeMinInput.addEventListener('change', applyRange);
  rangeMaxInput.addEventListener('change', applyRange);

  debugCheckbox.addEventListener('change', () => {
    debugEnabled = debugCheckbox.checked;
    console.log(`[perf] Debug timing ${debugEnabled ? 'enabled' : 'disabled'}`);
  });

  // Start WebSocket connection to the Python bridge
  connect();
}

main().catch(console.error);
