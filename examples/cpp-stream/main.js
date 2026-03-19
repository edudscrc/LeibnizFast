/**
 * main.js — WebSocket client for the cpp-stream example.
 *
 * Connects to the Python bridge (bridge.py) over WebSocket, receives
 * binary frames produced by the C++ wave simulator, parses the header,
 * and feeds each frame to LeibnizFast for live GPU rendering.
 *
 * Binary frame format (little-endian, all uint32):
 *   Offset  0: magic    = 0x4C465A00  ("LFZ\0")
 *   Offset  4: rows
 *   Offset  8: cols
 *   Offset 12: frame_id
 *   Offset 16: float32[rows × cols]  row-major grid data
 *
 * Resize flow (browser → C++ generator):
 *   browser sends JSON text  {"type":"resize","size":N}
 *   bridge.py parses it and forwards a 4-byte uint32 to the generator
 *   generator reinitializes the simulation at N×N and starts sending new frames
 */
import init, { LeibnizFast } from '../../pkg/leibniz_fast.js';

// ---- DOM refs --------------------------------------------------------
const canvas          = document.getElementById('canvas');
const colormapSelect  = document.getElementById('colormap');
const sizeSelect      = document.getElementById('size');
const rangeMinInput   = document.getElementById('range-min');
const rangeMaxInput   = document.getElementById('range-max');
const debugCheckbox   = document.getElementById('debug');
const statusBadge     = document.getElementById('status-badge');
const statusText      = document.getElementById('status-text');
const fpsCounter      = document.getElementById('fps-counter');
const tooltip         = document.getElementById('tooltip');
const errorBanner     = document.getElementById('error-banner');

// ---- Constants -------------------------------------------------------
const CHUNK_MAGIC        = 0x4C465A01;
const CHUNK_HEADER_BYTES = 32;     // 8 × uint32
const WS_URL       = 'ws://localhost:8765';
const RECONNECT_MS = 2000;   // retry interval on disconnect
const FPS_WINDOW   = 10;     // rolling FPS average over last N frames

// ---- State -----------------------------------------------------------
/** @type {LeibnizFast|null} */
let viewer = null;
/** @type {WebSocket|null} */
let ws = null;
let reconnectTimer = null;

/** Whether debug timing is enabled. */
let debugEnabled = debugCheckbox.checked;

// FPS tracking: timestamps (ms) of the last FPS_WINDOW frames
const frameTimes = [];

// Chunk accumulation: frameId → { totalChunks, totalRows, cols, full: Float32Array, received }
const pendingFrames = new Map();

// ---- Utilities -------------------------------------------------------

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
 * Record a frame arrival and update the FPS counter (rolling average).
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

// ---- Binary protocol -------------------------------------------------

/**
 * Parse the 32-byte chunk message header.
 * Returns null if magic is wrong or the buffer is too short.
 *
 * @param {ArrayBuffer} buf
 * @returns {{ totalRows: number, cols: number, frameId: number,
 *             chunkIndex: number, totalChunks: number,
 *             rowStart: number, chunkRows: number }|null}
 */
function parseChunkHeader(buf) {
  if (buf.byteLength < CHUNK_HEADER_BYTES) return null;

  // DataView reads little-endian fields portably regardless of CPU endianness.
  const view  = new DataView(buf);
  const magic = view.getUint32(0, /* littleEndian */ true);

  if (magic !== CHUNK_MAGIC) {
    console.warn(
      `[cpp-stream] Bad magic: 0x${magic.toString(16)} ` +
      `(expected 0x${CHUNK_MAGIC.toString(16)})`
    );
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
      `expected ${expectedBytes} (chunkRows=${chunkRows} cols=${cols})`
    );
    return null;
  }

  return { totalRows, cols, frameId, chunkIndex, totalChunks, rowStart, chunkRows };
}

// ---- Frame processing ------------------------------------------------

/**
 * Read min/max from the inputs and apply to the viewer.
 * Safe to call with no data loaded yet (viewer.setRange is a no-op then).
 */
function applyRange() {
  const mn = parseFloat(rangeMinInput.value);
  const mx = parseFloat(rangeMaxInput.value);
  if (viewer && isFinite(mn) && isFinite(mx) && mx > mn) {
    viewer.setRange(mn, mx);
  }
}

/**
 * Handle one received binary WebSocket message (a single frame chunk).
 *
 * Chunks are accumulated by frameId. When all chunks for a frame arrive,
 * the rows are already assembled in-place (each chunk is copied into its
 * correct position as it arrives), so no second pass is needed.
 * setData() is called once per complete frame.
 *
 * @param {ArrayBuffer} buf
 */
function processFrame(buf) {
  if (!viewer) return; // WASM not ready yet

  const h = parseChunkHeader(buf);
  if (!h) return;

  // Look up or create the accumulation entry for this frame.
  let pending = pendingFrames.get(h.frameId);
  if (!pending) {
    pending = {
      totalChunks: h.totalChunks,
      totalRows:   h.totalRows,
      cols:        h.cols,
      // Pre-allocate the full assembled buffer once on the first chunk.
      full:        new Float32Array(h.totalRows * h.cols),
      received:    0,
    };
    pendingFrames.set(h.frameId, pending);
  }

  // Copy this chunk's rows into the right position in the assembled buffer.
  // CHUNK_HEADER_BYTES = 32 is 4-byte aligned — TypedArray requirement satisfied.
  const src = new Float32Array(buf, CHUNK_HEADER_BYTES, h.chunkRows * h.cols);
  pending.full.set(src, h.rowStart * h.cols);
  pending.received++;

  // All chunks received: render the complete frame and clean up.
  if (pending.received === pending.totalChunks) {
    pendingFrames.delete(h.frameId);

    if (debugEnabled) {
      const t0 = performance.now();
      viewer.setData(pending.full, h.totalRows, h.cols);
      console.log(`[perf] setData (${h.totalRows}×${h.cols}): ${(performance.now() - t0).toFixed(2)}ms`);
    } else {
      viewer.setData(pending.full, h.totalRows, h.cols);
    }

    updateFps();

    // Discard any incomplete frames older than this one (lost chunks).
    for (const [id] of pendingFrames) {
      if (id < h.frameId) pendingFrames.delete(id);
    }
  }
}

// ---- WebSocket lifecycle ---------------------------------------------

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

  // Must be set before any messages arrive to receive ArrayBuffer, not Blob.
  ws.binaryType = 'arraybuffer';

  ws.addEventListener('open', () => {
    setStatus('connected');
    // Clear stale FPS history from any previous connection session.
    frameTimes.length = 0;
    fpsCounter.textContent = '-- FPS';
    // Sync the size selector with the generator on reconnect
    sendResize(parseInt(sizeSelect.value));
  });

  ws.addEventListener('message', (e) => {
    // e.data is an ArrayBuffer because we set binaryType = 'arraybuffer'.
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

// ---- Main ------------------------------------------------------------

async function main() {
  const t0 = performance.now();
  await init();
  if (debugEnabled) console.log(`[perf] WASM init: ${(performance.now() - t0).toFixed(2)}ms`);

  const t1 = performance.now();
  viewer = await LeibnizFast.create(canvas, colormapSelect.value, debugEnabled);
  if (debugEnabled) console.log(`[perf] LeibnizFast.create: ${(performance.now() - t1).toFixed(2)}ms`);

  // Set the initial sticky range so the first setData skips the min/max scan.
  applyRange();

  // Hover tooltip
  viewer.onHover((row, col, value) => {
    tooltip.style.display = 'block';
    tooltip.textContent = `[${row}, ${col}] = ${value.toFixed(4)}`;
  });

  // Pan / zoom interactions (same pattern as basic and gpu-gen examples)
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

  // Colormap selector
  colormapSelect.addEventListener('change', () => {
    const t = debugEnabled ? performance.now() : 0;
    viewer.setColormap(colormapSelect.value);
    if (debugEnabled) console.log(`[perf] setColormap: ${(performance.now() - t).toFixed(2)}ms`);
  });

  // Size selector: send resize command to C++ generator
  sizeSelect.addEventListener('change', () => {
    sendResize(parseInt(sizeSelect.value));
  });

  // Range inputs — respond to both 'input' (while typing) and
  // 'change' (on blur/Enter) for immediate feedback
  rangeMinInput.addEventListener('input', applyRange);
  rangeMaxInput.addEventListener('input', applyRange);
  rangeMinInput.addEventListener('change', applyRange);
  rangeMaxInput.addEventListener('change', applyRange);

  // Debug toggle — updates JS-side flag; WASM debug requires recreating the viewer
  debugCheckbox.addEventListener('change', () => {
    debugEnabled = debugCheckbox.checked;
    console.log(`[perf] Debug timing ${debugEnabled ? 'enabled' : 'disabled'}`);
  });

  // Start WebSocket connection to the Python bridge
  connect();
}

main().catch(console.error);
