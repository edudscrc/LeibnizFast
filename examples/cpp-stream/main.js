/**
 * main.js — WebSocket client for the cpp-stream example.
 *
 * Connects to the Python bridge (bridge.py) over WebSocket, receives
 * binary frames produced by the C++ wave simulator, and feeds each frame
 * to LeibnizFast for live GPU rendering.
 *
 * Supports two on-wire protocols from the generator:
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
 *   Offset 32: float32[chunk_rows × cols]  row-major grid data
 *
 * Protocol v2 — compressed / delta chunks (magic 0x4C465A02, 40-byte header):
 *   Offset  0: magic          = 0x4C465A02
 *   Offset  4..28: same as v1
 *   Offset 32: flags          bit0=compressed  bit1=int8_delta  bit2=keyframe
 *   Offset 36: payload_bytes  byte count of data following this header
 *   Offset 40: zlib deflate-raw compressed payload
 *              keyframe → float32[chunk_rows × cols]
 *              delta    → int8[chunk_rows × cols]  (reconstruct: prev + delta * 1/128)
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

// Protocol v1: plain float32 chunks
const CHUNK_MAGIC        = 0x4C465A01;
const CHUNK_HEADER_BYTES = 32;     // 8 × uint32

// Protocol v2: compressed / delta chunks
const ENHANCED_MAGIC        = 0x4C465A02;
const ENHANCED_HEADER_BYTES = 40;  // 10 × uint32
const FLAG_COMPRESSED = 0x1;       // payload is zlib deflate-raw compressed
const FLAG_DELTA      = 0x2;       // payload is int8 delta (after decompression)
const FLAG_KEYFRAME   = 0x4;       // full float32 resync frame
const DELTA_SCALE     = 1.0 / 128.0; // int8 → float32: float = prev + int8 * DELTA_SCALE

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

// Chunk accumulation: frameId → pending frame entry (structure depends on protocol version)
const pendingFrames = new Map();

// Reconstructed previous frame for delta decoding (protocol v2 --delta mode).
// Reset to null on reconnect/resize so the first keyframe initialises it cleanly.
/** @type {Float32Array|null} */
let prevFrame = null;

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
 * Parse a chunk message header (v1 or v2).
 * Returns null if magic is unrecognised or the buffer is too short.
 *
 * @param {ArrayBuffer} buf
 * @returns {{ version: number, totalRows: number, cols: number, frameId: number,
 *             chunkIndex: number, totalChunks: number, rowStart: number,
 *             chunkRows: number,
 *             // v2 only:
 *             flags?: number, payloadBytes?: number }|null}
 */
function parseChunkHeader(buf) {
  if (buf.byteLength < CHUNK_HEADER_BYTES) return null;

  // DataView reads little-endian fields portably regardless of host endianness.
  const view  = new DataView(buf);
  const magic = view.getUint32(0, /* littleEndian */ true);

  if (magic !== CHUNK_MAGIC && magic !== ENHANCED_MAGIC) {
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

  if (magic === ENHANCED_MAGIC) {
    if (buf.byteLength < ENHANCED_HEADER_BYTES) return null;
    const flags        = view.getUint32(32, true);
    const payloadBytes = view.getUint32(36, true);
    if (buf.byteLength < ENHANCED_HEADER_BYTES + payloadBytes) {
      console.warn(`[cpp-stream] Enhanced chunk too short: ${buf.byteLength} bytes`);
      return null;
    }
    return { version: 2, totalRows, cols, frameId, chunkIndex, totalChunks,
             rowStart, chunkRows, flags, payloadBytes };
  }

  // v1: validate payload size
  const expectedBytes = CHUNK_HEADER_BYTES + chunkRows * cols * 4;
  if (buf.byteLength < expectedBytes) {
    console.warn(
      `[cpp-stream] Chunk too short: ${buf.byteLength} bytes, ` +
      `expected ${expectedBytes} (chunkRows=${chunkRows} cols=${cols})`
    );
    return null;
  }
  return { version: 1, totalRows, cols, frameId, chunkIndex, totalChunks,
           rowStart, chunkRows };
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
 * Decompress a zlib-format payload (RFC 1950, as produced by zlib compress2)
 * using the native browser DecompressionStream API.
 *
 * Reading and writing run concurrently to avoid backpressure deadlock on
 * large frames — the readable must be consumed while writing, otherwise the
 * stream's internal queue fills up and writer.write() never resolves.
 *
 * @param {Uint8Array} data
 * @returns {Promise<ArrayBuffer>}
 */
async function inflate(data) {
  const ds     = new DecompressionStream('deflate');
  const writer = ds.writable.getWriter();
  const reader = ds.readable.getReader();

  // Start consuming the readable before writing so backpressure never blocks.
  const readAll = (async () => {
    const parts = [];
    // eslint-disable-next-line no-constant-condition
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      parts.push(value);
    }
    return parts;
  })();

  // Write input, close to signal EOF, then wait for all output.
  await writer.write(data);
  await writer.close();
  const parts = await readAll;

  // Always copy into a fresh buffer to guarantee byteOffset === 0.
  const total  = parts.reduce((n, p) => n + p.byteLength, 0);
  const result = new Uint8Array(total);
  let off = 0;
  for (const p of parts) { result.set(p, off); off += p.byteLength; }
  return result.buffer;
}

/**
 * Assemble a complete v2 frame: decompress each chunk, reconstruct delta if needed,
 * then call setData(). Called fire-and-forget once all chunks have arrived.
 *
 * @param {{ totalChunks: number, totalRows: number, cols: number,
 *           full: Float32Array, flags: number,
 *           rawChunks: Uint8Array[], rowStarts: number[], chunkRowCounts: number[] }} pending
 * @param {number} frameId  (used only for stale-frame cleanup)
 */
async function assembleFrame(pending, frameId) {
  const isKeyframe = !!(pending.flags & FLAG_KEYFRAME);
  const isDelta    = !!(pending.flags & FLAG_DELTA);

  if (debugEnabled) console.log(`[v2] assembleFrame frameId=${frameId} flags=0x${pending.flags.toString(16)} isKeyframe=${isKeyframe} isDelta=${isDelta} totalChunks=${pending.totalChunks}`);

  for (let c = 0; c < pending.totalChunks; c++) {
    let payload = pending.rawChunks[c]; // Uint8Array of compressed bytes

    if (pending.flags & FLAG_COMPRESSED) {
      const inputBytes = payload.byteLength;
      payload = new Uint8Array(await inflate(payload));
      if (debugEnabled) console.log(`[v2] chunk ${c}: inflate ${inputBytes}B → ${payload.byteLength}B`);
    }

    const rowOff    = pending.rowStarts[c] * pending.cols;
    const cellCount = pending.chunkRowCounts[c] * pending.cols;

    if (isDelta && !isKeyframe) {
      // Delta frame: reconstruct float = prevFrame + int8 * DELTA_SCALE.
      if (!prevFrame || prevFrame.length !== pending.full.length) {
        // Guard: delta arrived before any keyframe (shouldn't happen since
        // frame_id=0 is always a keyframe, but be defensive).
        console.warn('[cpp-stream] Delta frame arrived before keyframe — discarding');
        return;
      }
      const int8s = new Int8Array(payload.buffer, payload.byteOffset, cellCount);
      for (let i = 0; i < cellCount; i++) {
        pending.full[rowOff + i] = prevFrame[rowOff + i] + int8s[i] * DELTA_SCALE;
      }
    } else {
      // Keyframe or compress-only: float32 payload.
      const f32 = new Float32Array(payload.buffer, payload.byteOffset, cellCount);
      pending.full.set(f32, rowOff);
    }
  }

  // Update prevFrame for the next delta (keyframe or fully-reconstructed delta).
  if (!prevFrame || prevFrame.length !== pending.full.length) {
    prevFrame = new Float32Array(pending.full);
  } else {
    prevFrame.set(pending.full);
  }

  if (debugEnabled) {
    const t0 = performance.now();
    viewer.setData(pending.full, pending.totalRows, pending.cols);
    console.log(`[perf] setData (${pending.totalRows}×${pending.cols}): ${(performance.now() - t0).toFixed(2)}ms`);
  } else {
    viewer.setData(pending.full, pending.totalRows, pending.cols);
  }
  updateFps();

  // Discard any incomplete frames older than this one.
  for (const [id] of pendingFrames) {
    if (id < frameId) pendingFrames.delete(id);
  }
}

/**
 * Handle one received binary WebSocket message (a single frame chunk).
 *
 * Protocol v1: chunks are assembled in-place (sync, fast path).
 * Protocol v2: compressed raw bytes are stored per-chunk; assembleFrame()
 * is called asynchronously once all chunks have arrived.
 *
 * @param {ArrayBuffer} buf
 */
function processFrame(buf) {
  if (!viewer) return; // WASM not ready yet

  const h = parseChunkHeader(buf);
  if (!h) return;

  if (h.version === 1) {
    // ---- Protocol v1: plain float32, sync in-place assembly ----------

    let pending = pendingFrames.get(h.frameId);
    if (!pending) {
      pending = {
        version:     1,
        totalChunks: h.totalChunks,
        totalRows:   h.totalRows,
        cols:        h.cols,
        full:        new Float32Array(h.totalRows * h.cols),
        received:    0,
      };
      pendingFrames.set(h.frameId, pending);
    }

    // CHUNK_HEADER_BYTES = 32 is 4-byte aligned — TypedArray requirement met.
    const src = new Float32Array(buf, CHUNK_HEADER_BYTES, h.chunkRows * h.cols);
    pending.full.set(src, h.rowStart * h.cols);
    pending.received++;

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
      for (const [id] of pendingFrames) {
        if (id < h.frameId) pendingFrames.delete(id);
      }
    }

  } else {
    // ---- Protocol v2: compressed / delta, async assembly -------------

    let pending = pendingFrames.get(h.frameId);
    if (!pending) {
      pending = {
        version:        2,
        totalChunks:    h.totalChunks,
        totalRows:      h.totalRows,
        cols:           h.cols,
        full:           new Float32Array(h.totalRows * h.cols),
        flags:          h.flags,
        rawChunks:      new Array(h.totalChunks).fill(null),
        rowStarts:      new Array(h.totalChunks).fill(0),
        chunkRowCounts: new Array(h.totalChunks).fill(0),
        received:       0,
      };
      pendingFrames.set(h.frameId, pending);
    }

    // Store this chunk's compressed payload (copy from the WS message buffer).
    if (pending.rawChunks[h.chunkIndex] === null) {
      pending.rawChunks[h.chunkIndex]      = new Uint8Array(buf, ENHANCED_HEADER_BYTES, h.payloadBytes).slice();
      pending.rowStarts[h.chunkIndex]      = h.rowStart;
      pending.chunkRowCounts[h.chunkIndex] = h.chunkRows;
      pending.received++;
      if (debugEnabled) console.log(`[v2] stored chunk ${h.chunkIndex}/${h.totalChunks} frameId=${h.frameId} payloadBytes=${h.payloadBytes} flags=0x${h.flags.toString(16)}`);
    }

    if (pending.received === pending.totalChunks) {
      pendingFrames.delete(h.frameId);
      // Fire-and-forget: assembleFrame decompresses and calls setData asynchronously.
      // JS is single-threaded so microtask ordering ensures no concurrent mutation.
      assembleFrame(pending, h.frameId).catch((err) => {
        console.error('[cpp-stream] assembleFrame error:', err);
        showError(`Frame decode error: ${err?.message ?? String(err)}`);
      });
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
    // Clear stale FPS history and delta state from any previous session.
    frameTimes.length = 0;
    fpsCounter.textContent = '-- FPS';
    pendingFrames.clear();
    prevFrame = null; // ensure first v2 keyframe initialises delta reconstruction
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

  // Size selector: send resize command to C++ generator.
  // Reset delta state since grid dimensions will change.
  sizeSelect.addEventListener('change', () => {
    prevFrame = null;
    pendingFrames.clear();
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
