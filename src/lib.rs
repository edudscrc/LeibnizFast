//! # LeibnizFast
//!
//! GPU-accelerated 2D matrix visualization library for the browser.
//!
//! This crate provides the core Rust/WASM implementation that leverages wgpu
//! to render large matrices as colored heatmaps with interactive zoom, pan,
//! and hover inspection.
//!
//! # Architecture
//!
//! Modules are split into pure-logic (testable without GPU) and GPU wrappers:
//! - `camera`: `CameraState` (pure math) + `Camera` (GPU uniform buffer)
//! - `matrix`: `MatrixData` (CPU) + `MatrixView` (GPU buffer)
//! - `colormap` / `colormap_data`: colormap tables and `ColormapProvider` trait
//! - `interaction`: mouse event state machine (pure logic)
//! - `renderer`, `pipeline`: GPU setup (WASM-only)
//!
//! WebGPU is required. The crate does not provide CPU or WebGL fallbacks.

// Pure-logic modules — always compiled, testable on native
pub mod camera;
pub mod chunked_upload;
pub mod colormap;
pub mod colormap_data;
pub mod interaction;
pub mod matrix;
pub mod tile_grid;

// GPU/WASM modules — only compiled for wasm32 target
#[cfg(target_arch = "wasm32")]
mod perf;
#[cfg(target_arch = "wasm32")]
mod pipeline;
#[cfg(target_arch = "wasm32")]
mod renderer;

#[cfg(target_arch = "wasm32")]
mod wasm_entry {
    use wasm_bindgen::prelude::*;

    use crate::camera;
    use crate::colormap;
    use crate::interaction;
    use crate::matrix;
    use crate::perf::PerfTimer;
    use crate::renderer;

    /// Initialize panic hook for better error messages in the browser console.
    fn init_logging() {
        console_error_panic_hook::set_once();
        console_log::init_with_level(log::Level::Info).ok();
    }

    fn validate_nonzero_dimensions(rows: u32, cols: u32) -> Result<(), JsValue> {
        if rows == 0 || cols == 0 {
            return Err(JsValue::from_str(
                "Matrix dimensions must be greater than zero.",
            ));
        }
        Ok(())
    }

    fn validate_dimensions(rows: u32, cols: u32) -> Result<(), JsValue> {
        validate_nonzero_dimensions(rows, cols)?;
        checked_element_count(rows, cols).map(|_| ())
    }

    fn checked_element_count(rows: u32, cols: u32) -> Result<u32, JsValue> {
        let total = (rows as u64)
            .checked_mul(cols as u64)
            .ok_or_else(|| JsValue::from_str("Matrix dimensions overflow."))?;
        u32::try_from(total).map_err(|_| {
            JsValue::from_str(
                "Matrix is too large for a single JavaScript Float32Array. Use setDataChunks() with retainData: false.",
            )
        })
    }

    fn checked_row_offset(row: u32, cols: u32) -> Result<u32, JsValue> {
        row.checked_mul(cols)
            .ok_or_else(|| JsValue::from_str("Matrix row offset overflow."))
    }

    fn validate_range(min: f32, max: f32) -> Result<(), JsValue> {
        if !min.is_finite() || !max.is_finite() || max <= min {
            return Err(JsValue::from_str(
                "Range must contain finite values with max greater than min.",
            ));
        }
        Ok(())
    }

    fn scan_float32_array(data: &js_sys::Float32Array, debug: bool) -> (f32, f32) {
        let _timer = PerfTimer::new("scan_float32_array", debug);
        const SCAN_CHUNK_ELEMENTS: usize = 1024 * 1024;
        let total = data.length() as usize;
        let mut min_val = f32::INFINITY;
        let mut max_val = f32::NEG_INFINITY;
        let chunk_size = SCAN_CHUNK_ELEMENTS.min(total);
        if chunk_size > 0 {
            let mut buf = vec![0.0f32; chunk_size];
            let mut offset = 0;
            while offset < total {
                let end = (offset + chunk_size).min(total);
                let len = end - offset;
                data.subarray(offset as u32, end as u32)
                    .copy_to(&mut buf[..len]);
                for &v in &buf[..len] {
                    if v.is_finite() {
                        min_val = min_val.min(v);
                        max_val = max_val.max(v);
                    }
                }
                offset = end;
            }
        }
        (min_val, max_val)
    }

    fn finalize_range(min_val: f32, max_val: f32) -> (f32, f32) {
        if min_val.is_infinite() || max_val.is_infinite() {
            (0.0, 1.0)
        } else {
            (min_val, max_val)
        }
    }

    /// Tracks an in-progress streaming upload.
    struct PendingUpload {
        /// JS-heap-backed data source being built incrementally
        js_data: matrix::JsDataSource,
        /// GPU staging buffer
        matrix_view: matrix::MatrixView,
        /// Total rows expected
        rows: u32,
        /// Total columns expected
        cols: u32,
        /// Next expected row index (enforces sequential appends)
        next_row: u32,
    }

    /// Tracks a chunked upload that may avoid retaining CPU-side data.
    struct PendingChunkUpload {
        /// Optional JS-heap-backed cache for hover values.
        js_data: Option<matrix::JsDataSource>,
        /// GPU staging buffer
        matrix_view: matrix::MatrixView,
        rows: u32,
        cols: u32,
        next_row: u32,
        fixed_range: Option<(f32, f32)>,
        min_val: f32,
        max_val: f32,
    }

    /// Main entry point for the library. Owns all GPU state and provides
    /// the public API for matrix visualization.
    ///
    #[wasm_bindgen]
    pub struct LeibnizFast {
        renderer: renderer::Renderer,
        camera: camera::Camera,
        /// GPU-side staging resources for the active matrix.
        matrix: Option<matrix::MatrixView>,
        colormap_texture: Option<colormap::ColormapTexture>,
        interaction: interaction::InteractionState,
        /// JS-heap-backed matrix data for tooltip lookups and colormap re-dispatch
        js_data: Option<matrix::JsDataSource>,
        /// JavaScript callback for hover events
        hover_callback: Option<js_sys::Function>,
        /// Current colormap name
        current_colormap: String,
        /// Active data range used for colormap normalization.
        active_range: Option<(f32, f32)>,
        /// In-progress streaming upload, if any
        pending_upload: Option<PendingUpload>,
        /// In-progress chunked upload, if any
        pending_chunk_upload: Option<PendingChunkUpload>,
        /// Fixed range for colormap mapping set via setRange().
        ///
        /// When Some, `set_data` skips the min/max scan and uses this range
        /// directly, eliminating both the scan cost and the second GPU dispatch.
        sticky_range: Option<(f32, f32)>,
        /// Enable performance timing logs
        debug: bool,
    }

    #[wasm_bindgen]
    impl LeibnizFast {
        /// Create a new LeibnizFast instance attached to the given canvas element.
        #[wasm_bindgen]
        pub async fn create(
            canvas: web_sys::HtmlCanvasElement,
            colormap: Option<String>,
            debug: Option<bool>,
        ) -> Result<LeibnizFast, JsValue> {
            init_logging();
            let debug = debug.unwrap_or(false);
            let _timer = PerfTimer::new("LeibnizFast::create", debug);
            log::info!("LeibnizFast: initializing...");

            let colormap_name = colormap.unwrap_or_else(|| "viridis".to_string());

            let renderer = renderer::Renderer::new(&canvas, debug)
                .await
                .map_err(|e| JsValue::from_str(&e))?;

            let camera = camera::Camera::new(
                &renderer.device,
                canvas.width() as f32,
                canvas.height() as f32,
            );

            log::info!("LeibnizFast: initialized successfully");

            Ok(LeibnizFast {
                renderer,
                camera,
                matrix: None,
                colormap_texture: None,
                interaction: interaction::InteractionState::new(),
                js_data: None,
                hover_callback: None,
                current_colormap: colormap_name,
                active_range: None,
                pending_upload: None,
                pending_chunk_upload: None,
                sticky_range: None,
                debug,
            })
        }

        /// Set the matrix data to visualize.
        ///
        /// `data` is a Float32Array in row-major order (kept in JS heap — no copy
        /// into WASM memory). `rows` and `cols` specify the matrix dimensions.
        /// Min/max is scanned in small chunks. Tooltips and colormap changes work
        /// at any matrix size.
        ///
        /// When called repeatedly with the same dimensions (e.g. real-time streaming),
        /// the GPU pipeline is reused and only the colormap compute shader is
        /// re-dispatched — no resource allocation or pipeline rebuild occurs.
        #[wasm_bindgen(js_name = setData)]
        pub fn set_data(
            &mut self,
            data: js_sys::Float32Array,
            rows: u32,
            cols: u32,
        ) -> Result<(), JsValue> {
            let _timer = PerfTimer::new("set_data", self.debug);
            if self.pending_upload.is_some() || self.pending_chunk_upload.is_some() {
                return Err(JsValue::from_str("An upload is already in progress."));
            }
            validate_nonzero_dimensions(rows, cols)?;
            let expected_len = checked_element_count(rows, cols)?;
            if data.length() != expected_len {
                return Err(JsValue::from_str(&format!(
                    "Data length {} does not match rows x cols = {} x {} = {}",
                    data.length(),
                    rows,
                    cols,
                    expected_len
                )));
            }

            // Create JsDataSource.
            // When a sticky range is set (manual range mode), skip the min/max scan and
            // use the pre-supplied range directly — saves ~70ms per frame for large matrices.
            let js_data = if let Some((min, max)) = self.sticky_range {
                matrix::JsDataSource::new_with_range(data, rows, cols, min, max)
            } else {
                matrix::JsDataSource::new(data, rows, cols, self.debug)
            };
            self.active_range = Some(js_data.range());
            self.js_data = Some(js_data);

            // Fast path: if the GPU pipeline already exists for the same dimensions,
            // skip MatrixView allocation and rebuild_pipelines(). Just re-dispatch the
            // colormap compute shader with the new data. This eliminates per-frame GPU
            // resource churn that causes visible flickering in real-time streaming.
            let same_dims = self
                .matrix
                .as_ref()
                .is_some_and(|m| m.rows() == rows && m.cols() == cols);

            if same_dims {
                // Fast path: data copy only — no pipeline rebuild
                if let (Some(ref matrix_view), Some(ref jd)) = (&self.matrix, &self.js_data) {
                    let (min_val, max_val) = jd.range();
                    self.renderer.update_range_buffer(min_val, max_val);
                    let read_fn = |start: usize, buf: &mut [f32]| {
                        jd.read_range(start, buf);
                    };
                    self.renderer
                        .apply_colormap_tiled(matrix_view, &read_fn, cols);
                }
            } else {
                // Slow path: first call or dimension change — full pipeline rebuild
                if self.colormap_texture.is_none() {
                    self.set_colormap_internal(&self.current_colormap.clone())?;
                }

                self.camera.state.set_matrix_size(rows, cols);

                let matrix_view = matrix::MatrixView::with_empty_buffer(
                    &self.renderer.device,
                    rows,
                    cols,
                    self.debug,
                )
                .map_err(|e| JsValue::from_str(&e))?;
                self.matrix = Some(matrix_view);

                self.renderer
                    .rebuild_pipelines(
                        &self.matrix,
                        &self.colormap_texture,
                        &self.camera,
                        rows,
                        cols,
                    )
                    .map_err(|e| JsValue::from_str(&e))?;

                if let (Some(ref matrix_view), Some(ref jd)) = (&self.matrix, &self.js_data) {
                    let (min_val, max_val) = jd.range();
                    self.renderer.update_range_buffer(min_val, max_val);
                    let read_fn = |start: usize, buf: &mut [f32]| {
                        jd.read_range(start, buf);
                    };
                    self.renderer
                        .apply_colormap_tiled(matrix_view, &read_fn, cols);
                }
            }
            // setData always performs a full colormap from scratch, so the ring
            // cursor must be at 0 — any subsequent setDataScrolled call will use
            // a fresh JS WaterfallBuffer whose ringCursor is also 0.
            self.renderer.reset_ring_cursor();

            self.render_frame()?;

            Ok(())
        }

        /// Scrolled streaming update: shift existing pixels left and only colormap new columns.
        ///
        /// Use this instead of `setData` when the JS buffer shifts left by `new_cols`
        /// and writes new data at the right edge (waterfall/scrolling pattern).
        /// Reduces per-frame GPU work from O(rows × cols) to O(rows × new_cols).
        ///
        /// **Requires `setRange()` to be called first** to set a fixed colormap range.
        #[wasm_bindgen(js_name = setDataScrolled)]
        pub fn set_data_scrolled(
            &mut self,
            data: js_sys::Float32Array,
            rows: u32,
            cols: u32,
            new_cols: u32,
        ) -> Result<(), JsValue> {
            let _timer = PerfTimer::new("set_data_scrolled", self.debug);
            if self.pending_upload.is_some() || self.pending_chunk_upload.is_some() {
                return Err(JsValue::from_str("An upload is already in progress."));
            }
            validate_dimensions(rows, cols)?;

            // Require sticky_range — without it we'd need a full min/max scan
            // which defeats the purpose of the scrolled path.
            let (min_val, max_val) = match self.sticky_range {
                Some(range) => range,
                None => {
                    return Err(JsValue::from_str(
                        "setDataScrolled() requires a fixed range. Call setRange(min, max) before using the scrolled streaming path.",
                    ));
                }
            };
            self.active_range = Some((min_val, max_val));

            let expected_len = checked_element_count(rows, cols)?;
            if data.length() != expected_len {
                return Err(JsValue::from_str(&format!(
                    "Data length {} does not match rows x cols = {} x {} = {}",
                    data.length(),
                    rows,
                    cols,
                    expected_len
                )));
            }

            // Store as JsDataSource with the fixed range (skip scan)
            let js_data = matrix::JsDataSource::new_with_range(data, rows, cols, min_val, max_val);
            self.js_data = Some(js_data);

            // Check if pipeline already exists for these dimensions
            let same_dims = self
                .matrix
                .as_ref()
                .is_some_and(|m| m.rows() == rows && m.cols() == cols);

            if !same_dims {
                // First call or dimension change: initialize textures with a full upload.
                if self.colormap_texture.is_none() {
                    self.set_colormap_internal(&self.current_colormap.clone())?;
                }
                self.camera.state.set_matrix_size(rows, cols);

                let matrix_view = matrix::MatrixView::with_empty_buffer(
                    &self.renderer.device,
                    rows,
                    cols,
                    self.debug,
                )
                .map_err(|e| JsValue::from_str(&e))?;
                self.matrix = Some(matrix_view);

                self.renderer
                    .rebuild_pipelines(
                        &self.matrix,
                        &self.colormap_texture,
                        &self.camera,
                        rows,
                        cols,
                    )
                    .map_err(|e| JsValue::from_str(&e))?;

                self.renderer.reset_ring_cursor();
                self.renderer.update_range_buffer(min_val, max_val);
                if let (Some(ref matrix_view), Some(ref jd)) = (&self.matrix, &self.js_data) {
                    let read_fn = |start: usize, buf: &mut [f32]| {
                        jd.read_range(start, buf);
                    };
                    self.renderer
                        .apply_colormap_tiled(matrix_view, &read_fn, cols);
                }
                self.render_frame()?;
                return Ok(());
            }

            // Ring-buffer fast path: only copy new columns at the cursor,
            // O(rows × new_cols) regardless of total column count.
            if let (Some(ref matrix_view), Some(ref jd)) = (&self.matrix, &self.js_data) {
                let read_fn = |start: usize, buf: &mut [f32]| {
                    jd.read_range(start, buf);
                };
                self.renderer
                    .apply_colormap_ring(matrix_view, &read_fn, cols, new_cols);
            }

            self.render_frame()?;
            Ok(())
        }

        /// Set the colormap used for visualization.
        ///
        /// Instant O(1) operation: updates the colormap LUT texture and rebuilds
        /// the shared render bind group. No data re-read or compute re-dispatch.
        /// The fragment shader applies the new colormap on the next render.
        #[wasm_bindgen(js_name = setColormap)]
        pub fn set_colormap(&mut self, name: &str) -> Result<(), JsValue> {
            let _timer = PerfTimer::new("set_colormap", self.debug);
            self.set_colormap_internal(name)?;

            if self.matrix.is_some() {
                // Rebuild the shared colormap bind group (render group 1) with the new LUT
                if let Some(ref colormap) = self.colormap_texture {
                    self.renderer
                        .rebuild_colormap_bind_group(colormap)
                        .map_err(|e| JsValue::from_str(&e))?;
                }

                self.render_frame()?;
            }

            Ok(())
        }

        /// Set the data range for colormap mapping.
        ///
        /// Instant O(1) operation: updates the range uniform buffer (16 bytes).
        /// No data re-read or compute re-dispatch. The fragment shader applies
        /// the new range on the next render.
        #[wasm_bindgen(js_name = setRange)]
        pub fn set_range(&mut self, min: f32, max: f32) -> Result<(), JsValue> {
            let _timer = PerfTimer::new("set_range", self.debug);
            validate_range(min, max)?;
            // Persist as sticky range so future setData calls use it without rescanning
            self.sticky_range = Some((min, max));
            self.active_range = Some((min, max));
            if let Some(ref mut jd) = self.js_data {
                jd.set_range(min, max);
            }

            if self.matrix.is_some() {
                // Update the range uniform buffer — fragment shader picks it up immediately
                self.renderer.update_range_buffer(min, max);

                self.render_frame()?;
            }

            Ok(())
        }

        /// Begin a streaming data upload.
        ///
        /// Allocates a JS-heap Float32Array for `rows × cols` elements and a GPU
        /// staging buffer. Builds pipelines early so the compute shader is available
        /// for per-chunk processing in `append_chunk()`.
        /// Use `append_chunk()` to upload data, then `end_data()` to finalize.
        /// Errors if an upload is already in progress.
        #[wasm_bindgen(js_name = beginData)]
        pub fn begin_data(&mut self, rows: u32, cols: u32) -> Result<(), JsValue> {
            let _timer = PerfTimer::new("begin_data", self.debug);
            if self.pending_upload.is_some() {
                return Err(JsValue::from_str(
                    "A streaming upload is already in progress. Call endData() first.",
                ));
            }
            if self.pending_chunk_upload.is_some() {
                return Err(JsValue::from_str(
                    "A chunked upload is already in progress.",
                ));
            }
            validate_dimensions(rows, cols)?;

            // Allocate JS-heap accumulator — no WASM memory pressure
            let js_data =
                matrix::JsDataSource::from_empty(rows, cols).map_err(|e| JsValue::from_str(&e))?;

            let matrix_view = matrix::MatrixView::with_empty_buffer(
                &self.renderer.device,
                rows,
                cols,
                self.debug,
            )
            .map_err(|e| JsValue::from_str(&e))?;

            // Build pipelines early so compute shader is available for per-chunk dispatch
            if self.colormap_texture.is_none() {
                self.set_colormap_internal(&self.current_colormap.clone())?;
            }

            self.matrix = Some(matrix_view);
            self.renderer
                .rebuild_pipelines(
                    &self.matrix,
                    &self.colormap_texture,
                    &self.camera,
                    rows,
                    cols,
                )
                .map_err(|e| JsValue::from_str(&e))?;

            let matrix_view = self
                .matrix
                .take()
                .ok_or_else(|| JsValue::from_str("Internal error: matrix missing after setup."))?;
            self.pending_upload = Some(PendingUpload {
                js_data,
                matrix_view,
                rows,
                cols,
                next_row: 0,
            });

            Ok(())
        }

        /// Begin a streaming update, reusing GPU resources when dimensions match.
        ///
        /// This is the fast path for real-time streaming: when called with the
        /// same dimensions as the previous frame, it reuses the existing
        /// `JsDataSource` (Float32Array in JS heap) and `MatrixView` (GPU staging
        /// buffer), avoiding per-frame allocation and pipeline rebuild.
        ///
        /// Falls back to `begin_data()` on first use or when dimensions change.
        /// Follow with `append_chunk()` calls and finalize with `end_data()`.
        #[wasm_bindgen(js_name = beginUpdate)]
        pub fn begin_update(&mut self, rows: u32, cols: u32) -> Result<(), JsValue> {
            let _timer = PerfTimer::new("begin_update", self.debug);
            if self.pending_upload.is_some() {
                return Err(JsValue::from_str(
                    "A streaming upload is already in progress. Call endData() or abortData() first.",
                ));
            }
            if self.pending_chunk_upload.is_some() {
                return Err(JsValue::from_str(
                    "A chunked upload is already in progress.",
                ));
            }

            validate_dimensions(rows, cols)?;

            let same_dims = self
                .matrix
                .as_ref()
                .is_some_and(|m| m.rows() == rows && m.cols() == cols)
                && self.js_data.is_some();

            if same_dims {
                // Fast path: reuse existing resources — no allocation, no pipeline rebuild
                let mut js_data = self.js_data.take().ok_or_else(|| {
                    JsValue::from_str("begin_update: js_data missing despite same_dims check")
                })?;
                if let Some((min, max)) = self.sticky_range {
                    js_data.set_range(min, max);
                } else {
                    js_data.reset_range();
                }
                let matrix_view = self.matrix.take();
                let matrix_view = matrix_view.ok_or_else(|| {
                    JsValue::from_str("begin_update: matrix missing despite same_dims check")
                })?;
                self.pending_upload = Some(PendingUpload {
                    js_data,
                    matrix_view,
                    rows,
                    cols,
                    next_row: 0,
                });
                Ok(())
            } else {
                // Slow path: first call or dimension change — full setup
                self.begin_data(rows, cols)
            }
        }

        /// Abort an in-progress streaming upload.
        ///
        /// Discards the pending upload and restores reusable resources
        /// (`JsDataSource`, `MatrixView`) back to the viewer so the next
        /// `begin_update()` can reuse them. No-op if no upload is in progress.
        #[wasm_bindgen(js_name = abortData)]
        pub fn abort_data(&mut self) {
            if let Some(pending) = self.pending_upload.take() {
                self.js_data = Some(pending.js_data);
                self.matrix = Some(pending.matrix_view);
            }
        }

        /// Abort an in-progress chunked upload.
        #[wasm_bindgen(js_name = abortDataChunks)]
        pub fn abort_data_chunks(&mut self) {
            self.pending_chunk_upload = None;
        }

        /// Render a single frame without modifying data.
        ///
        /// Useful for decoupled rendering: ingest data via `append_chunk()`
        /// at the data source rate, then call `render()` at the display
        /// refresh rate via `requestAnimationFrame`.
        #[wasm_bindgen]
        pub fn render(&mut self) -> Result<(), JsValue> {
            let _timer = PerfTimer::new("render", self.debug);
            self.render_frame()
        }

        /// Append a chunk of rows to the in-progress streaming upload.
        ///
        /// `chunk` must contain a whole number of rows. `start_row` must
        /// match the next expected row (sequential ordering required).
        ///
        /// Copies chunk data to the JS-heap accumulator for tooltip lookup and
        /// final GPU upload. The min/max range uses a running estimate finalized
        /// in `end_data()`.
        #[wasm_bindgen(js_name = appendChunk)]
        pub fn append_chunk(&mut self, chunk: &[f32], start_row: u32) -> Result<(), JsValue> {
            let _timer = PerfTimer::new("append_chunk", self.debug);
            let fixed_range = self.sticky_range.is_some();
            let pending = self.pending_upload.as_mut().ok_or_else(|| {
                JsValue::from_str("No streaming upload in progress. Call beginData() first.")
            })?;

            if start_row != pending.next_row {
                return Err(JsValue::from_str(&format!(
                    "Expected start_row={}, got {start_row}. Chunks must be sequential.",
                    pending.next_row
                )));
            }

            let cols = pending.cols as usize;
            if cols > 0 && !chunk.len().is_multiple_of(cols) {
                return Err(JsValue::from_str(&format!(
                    "Chunk length {} is not divisible by cols {}",
                    chunk.len(),
                    cols
                )));
            }

            let chunk_rows = if cols > 0 {
                (chunk.len() / cols) as u32
            } else {
                0
            };

            let end_row = start_row
                .checked_add(chunk_rows)
                .ok_or_else(|| JsValue::from_str("Chunk row range overflow."))?;
            if end_row > pending.rows {
                return Err(JsValue::from_str(&format!(
                    "Chunk would exceed total rows: start_row={start_row} + chunk_rows={chunk_rows} > {}",
                    pending.rows
                )));
            }

            // Copy chunk to JS-heap accumulator and track running min/max.
            // Colormap is applied once in end_data() with the final range,
            // ensuring consistent coloring across all chunks.
            let element_offset = checked_row_offset(start_row, pending.cols)?;
            pending.js_data.write_range(element_offset, chunk);
            if !fixed_range {
                pending.js_data.update_min_max(chunk);
            }

            pending.next_row = end_row;

            Ok(())
        }

        /// Finalize a streaming upload.
        ///
        /// Finalizes min/max, stores the JS-heap data source, and renders.
        /// Errors if the upload is incomplete (not all rows uploaded).
        #[wasm_bindgen(js_name = endData)]
        pub fn end_data(&mut self) -> Result<(), JsValue> {
            let _timer = PerfTimer::new("end_data", self.debug);
            let mut pending = self.pending_upload.take().ok_or_else(|| {
                JsValue::from_str("No streaming upload in progress. Call beginData() first.")
            })?;

            if pending.next_row != pending.rows {
                // Put it back so user can continue
                let next_row = pending.next_row;
                let total = pending.rows;
                self.pending_upload = Some(pending);
                return Err(JsValue::from_str(&format!(
                    "Upload incomplete: {next_row}/{total} rows uploaded."
                )));
            }

            let rows = pending.rows;
            let cols = pending.cols;

            if let Some((min, max)) = self.sticky_range {
                pending.js_data.set_range(min, max);
            } else {
                pending.js_data.finalize();
            }
            self.active_range = Some(pending.js_data.range());

            self.js_data = Some(pending.js_data);
            self.matrix = Some(pending.matrix_view);

            if self.colormap_texture.is_none() {
                self.set_colormap_internal(&self.current_colormap.clone())?;
            }

            self.camera.state.set_matrix_size(rows, cols);

            // Apply colormap in a single pass with the final min/max range.
            // This ensures consistent coloring across all rows (no per-chunk
            // range drift). Pipelines were already built in begin_data().
            if let (Some(ref matrix_view), Some(ref jd)) = (&self.matrix, &self.js_data) {
                let (min_val, max_val) = jd.range();
                self.renderer.update_range_buffer(min_val, max_val);
                let read_fn = |start: usize, buf: &mut [f32]| {
                    jd.read_range(start, buf);
                };
                self.renderer
                    .apply_colormap_tiled(matrix_view, &read_fn, cols);
            }

            self.render_frame()?;

            Ok(())
        }

        /// Begin a chunked matrix upload.
        ///
        /// This path uploads row-major chunks directly into tiled GPU textures.
        /// When `retain_data` is false, hover callbacks still report row/column
        /// coordinates but do not retain CPU-side values.
        #[wasm_bindgen(js_name = beginDataChunks)]
        pub fn begin_data_chunks(
            &mut self,
            rows: u32,
            cols: u32,
            retain_data: bool,
            range_min: Option<f32>,
            range_max: Option<f32>,
        ) -> Result<(), JsValue> {
            let _timer = PerfTimer::new("begin_data_chunks", self.debug);
            if self.pending_upload.is_some() || self.pending_chunk_upload.is_some() {
                return Err(JsValue::from_str("An upload is already in progress."));
            }
            validate_nonzero_dimensions(rows, cols)?;

            let fixed_range = match (range_min, range_max) {
                (Some(min), Some(max)) => {
                    validate_range(min, max)?;
                    Some((min, max))
                }
                (None, None) => None,
                _ => {
                    return Err(JsValue::from_str(
                        "Chunked range must provide both min and max, or neither.",
                    ));
                }
            };

            let js_data = if retain_data {
                Some(
                    matrix::JsDataSource::from_empty(rows, cols)
                        .map_err(|e| JsValue::from_str(&e))?,
                )
            } else {
                None
            };

            if self.colormap_texture.is_none() {
                self.set_colormap_internal(&self.current_colormap.clone())?;
            }
            self.camera.state.set_matrix_size(rows, cols);

            let matrix_view = matrix::MatrixView::with_empty_buffer(
                &self.renderer.device,
                rows,
                cols,
                self.debug,
            )
            .map_err(|e| JsValue::from_str(&e))?;

            self.matrix = Some(matrix_view);
            self.js_data = None;
            self.renderer.reset_ring_cursor();
            self.renderer
                .rebuild_pipelines(
                    &self.matrix,
                    &self.colormap_texture,
                    &self.camera,
                    rows,
                    cols,
                )
                .map_err(|e| JsValue::from_str(&e))?;

            let matrix_view = self
                .matrix
                .take()
                .ok_or_else(|| JsValue::from_str("Internal error: matrix missing after setup."))?;
            self.pending_chunk_upload = Some(PendingChunkUpload {
                js_data,
                matrix_view,
                rows,
                cols,
                next_row: 0,
                fixed_range,
                min_val: f32::INFINITY,
                max_val: f32::NEG_INFINITY,
            });

            Ok(())
        }

        /// Append one row-major chunk to an active chunked matrix upload.
        #[wasm_bindgen(js_name = appendDataChunk)]
        pub fn append_data_chunk(
            &mut self,
            data: js_sys::Float32Array,
            start_row: u32,
        ) -> Result<(), JsValue> {
            let _timer = PerfTimer::new("append_data_chunk", self.debug);
            let mut pending = self.pending_chunk_upload.take().ok_or_else(|| {
                JsValue::from_str("No chunked upload in progress. Call beginDataChunks() first.")
            })?;

            let result = (|| -> Result<(), JsValue> {
                if start_row != pending.next_row {
                    return Err(JsValue::from_str(&format!(
                        "Expected start_row={}, got {start_row}. Chunks must be sequential.",
                        pending.next_row
                    )));
                }

                let len = data.length();
                if len == 0 || !len.is_multiple_of(pending.cols) {
                    return Err(JsValue::from_str(&format!(
                        "Chunk length {} is not a positive multiple of cols {}.",
                        len, pending.cols
                    )));
                }

                let chunk_rows = len / pending.cols;
                let end_row = start_row
                    .checked_add(chunk_rows)
                    .ok_or_else(|| JsValue::from_str("Chunk row range overflow."))?;
                if end_row > pending.rows {
                    return Err(JsValue::from_str(&format!(
                        "Chunk would exceed total rows: start_row={start_row} + chunk_rows={chunk_rows} > {}",
                        pending.rows
                    )));
                }

                if let Some(ref jd) = pending.js_data {
                    let element_offset = checked_row_offset(start_row, pending.cols)?;
                    let mut offset = 0;
                    const COPY_CHUNK: u32 = 1024 * 1024;
                    while offset < len {
                        let end = (offset + COPY_CHUNK).min(len);
                        let mut buf = vec![0.0f32; (end - offset) as usize];
                        data.subarray(offset, end).copy_to(&mut buf);
                        let dst_offset = element_offset
                            .checked_add(offset)
                            .ok_or_else(|| JsValue::from_str("Chunk copy offset overflow."))?;
                        jd.write_range(dst_offset, &buf);
                        offset = end;
                    }
                }

                if pending.fixed_range.is_none() {
                    let (chunk_min, chunk_max) = scan_float32_array(&data, self.debug);
                    if chunk_min < pending.min_val {
                        pending.min_val = chunk_min;
                    }
                    if chunk_max > pending.max_val {
                        pending.max_val = chunk_max;
                    }
                }

                let read_fn = |start: usize, buf: &mut [f32]| {
                    let end = start + buf.len();
                    data.subarray(start as u32, end as u32).copy_to(buf);
                };
                self.renderer
                    .upload_rows_tiled(
                        &pending.matrix_view,
                        start_row,
                        chunk_rows,
                        pending.cols,
                        &read_fn,
                    )
                    .map_err(|e| JsValue::from_str(&e))?;

                pending.next_row = end_row;
                Ok(())
            })();

            if result.is_err() {
                self.pending_chunk_upload = Some(pending);
                return result;
            }

            self.pending_chunk_upload = Some(pending);
            Ok(())
        }

        /// Finalize a chunked matrix upload and render it.
        #[wasm_bindgen(js_name = endDataChunks)]
        pub fn end_data_chunks(&mut self) -> Result<(), JsValue> {
            let _timer = PerfTimer::new("end_data_chunks", self.debug);
            let mut pending = self.pending_chunk_upload.take().ok_or_else(|| {
                JsValue::from_str("No chunked upload in progress. Call beginDataChunks() first.")
            })?;

            if pending.next_row != pending.rows {
                let next_row = pending.next_row;
                let total = pending.rows;
                self.pending_chunk_upload = Some(pending);
                return Err(JsValue::from_str(&format!(
                    "Upload incomplete: {next_row}/{total} rows uploaded."
                )));
            }

            let (min_val, max_val) = pending
                .fixed_range
                .unwrap_or_else(|| finalize_range(pending.min_val, pending.max_val));
            self.active_range = Some((min_val, max_val));

            if let Some(ref mut jd) = pending.js_data {
                jd.set_range(min_val, max_val);
            }

            self.renderer.update_range_buffer(min_val, max_val);
            self.js_data = pending.js_data;
            self.matrix = Some(pending.matrix_view);
            self.render_frame()?;

            Ok(())
        }

        /// Get the maximum number of matrix elements supported by this device.
        #[wasm_bindgen(js_name = getMaxMatrixElements)]
        pub fn get_max_matrix_elements(&self) -> f64 {
            // Return as f64 since JS numbers are doubles and u64 can't cross wasm boundary
            (self.renderer.max_buffer_size() / 4) as f64
        }

        /// Get the maximum matrix dimension (rows or cols) supported by this device.
        ///
        /// Matrices with rows or cols exceeding this will fail at pipeline build time.
        #[wasm_bindgen(js_name = getMaxTextureDimension)]
        pub fn get_max_texture_dimension(&self) -> u32 {
            self.renderer.max_texture_dimension()
        }

        /// Get the current camera visible range as UV coordinates.
        ///
        /// Returns `[uv_offset_x, uv_offset_y, uv_scale_x, uv_scale_y]`.
        /// The TypeScript layer maps these to data-space axis coordinates.
        #[wasm_bindgen(js_name = getVisibleRange)]
        pub fn get_visible_range(&self) -> Vec<f32> {
            let uniforms = self.camera.state.get_uniforms();
            vec![
                uniforms.uv_offset[0],
                uniforms.uv_offset[1],
                uniforms.uv_scale[0],
                uniforms.uv_scale[1],
            ]
        }

        /// Get the active data range used for colormap normalization.
        ///
        /// Returns an empty array when no explicit range or uploaded data has
        /// established the range yet.
        #[wasm_bindgen(js_name = getColorRange)]
        pub fn get_color_range(&self) -> Vec<f32> {
            self.active_range
                .map(|(min, max)| vec![min, max])
                .unwrap_or_default()
        }

        /// Get the active colormap as packed RGB bytes.
        ///
        /// The returned array contains 256 entries in RGBRGB... order and is
        /// sourced from the same lookup table used to build the GPU LUT.
        #[wasm_bindgen(js_name = getColormapLut)]
        pub fn get_colormap_lut(&self) -> Vec<u8> {
            use colormap::ColormapProvider;
            let provider = colormap::BuiltinColormaps;
            let Some(rgb_data) = provider.get_colormap_rgb(&self.current_colormap) else {
                return Vec::new();
            };

            let mut lut = Vec::with_capacity(rgb_data.len() * 3);
            for rgb in rgb_data {
                lut.extend_from_slice(rgb);
            }
            lut
        }

        /// Register a callback for hover events.
        #[wasm_bindgen(js_name = onHover)]
        pub fn on_hover(&mut self, callback: js_sys::Function) {
            self.hover_callback = Some(callback);
        }

        /// Handle mouse down event. Called from JS event listeners.
        #[wasm_bindgen(js_name = onMouseDown)]
        pub fn on_mouse_down(&mut self, x: f32, y: f32) {
            self.interaction.mouse_down(x, y);
        }

        /// Handle mouse move event. Called from JS event listeners.
        #[wasm_bindgen(js_name = onMouseMove)]
        pub fn on_mouse_move(&mut self, x: f32, y: f32) -> Result<(), JsValue> {
            let _timer = PerfTimer::new("on_mouse_move", self.debug);
            match self.interaction.mouse_move(x, y) {
                interaction::InteractionResult::Pan { dx, dy } => {
                    self.camera.state.pan(dx, dy);
                    self.camera.update_uniform(&self.renderer.queue);
                    self.render_frame()?;
                }
                interaction::InteractionResult::Hover => {
                    self.handle_hover(x, y)?;
                }
                interaction::InteractionResult::None => {}
            }
            Ok(())
        }

        /// Handle mouse up event. Called from JS event listeners.
        #[wasm_bindgen(js_name = onMouseUp)]
        pub fn on_mouse_up(&mut self) {
            self.interaction.mouse_up();
        }

        /// Handle wheel/scroll event for zooming. Called from JS event listeners.
        #[wasm_bindgen(js_name = onWheel)]
        pub fn on_wheel(&mut self, x: f32, y: f32, delta: f32) -> Result<(), JsValue> {
            let _timer = PerfTimer::new("on_wheel", self.debug);
            self.camera.state.zoom_at(x, y, delta);
            self.camera.update_uniform(&self.renderer.queue);
            self.render_frame()?;
            Ok(())
        }

        /// Zoom only the X axis at a screen position. Called from JS for axis-region scroll.
        #[wasm_bindgen(js_name = zoomAtX)]
        pub fn zoom_at_x(&mut self, screen_x: f32, delta: f32) -> Result<(), JsValue> {
            self.camera.state.zoom_at_x(screen_x, delta);
            self.camera.update_uniform(&self.renderer.queue);
            self.render_frame()?;
            Ok(())
        }

        /// Zoom only the Y axis at a screen position. Called from JS for axis-region scroll.
        #[wasm_bindgen(js_name = zoomAtY)]
        pub fn zoom_at_y(&mut self, screen_y: f32, delta: f32) -> Result<(), JsValue> {
            self.camera.state.zoom_at_y(screen_y, delta);
            self.camera.update_uniform(&self.renderer.queue);
            self.render_frame()?;
            Ok(())
        }

        /// Pan only the X axis. Called from JS for axis-region drag.
        #[wasm_bindgen(js_name = panX)]
        pub fn pan_x(&mut self, dx: f32) -> Result<(), JsValue> {
            self.camera.state.pan_x(dx);
            self.camera.update_uniform(&self.renderer.queue);
            self.render_frame()?;
            Ok(())
        }

        /// Pan only the Y axis. Called from JS for axis-region drag.
        #[wasm_bindgen(js_name = panY)]
        pub fn pan_y(&mut self, dy: f32) -> Result<(), JsValue> {
            self.camera.state.pan_y(dy);
            self.camera.update_uniform(&self.renderer.queue);
            self.render_frame()?;
            Ok(())
        }

        /// Zoom to frame a UV rectangle. Called from JS for selection-rect zoom.
        #[wasm_bindgen(js_name = zoomToUvRect)]
        pub fn zoom_to_uv_rect(
            &mut self,
            u_min: f32,
            v_min: f32,
            u_max: f32,
            v_max: f32,
        ) -> Result<(), JsValue> {
            self.camera
                .state
                .zoom_to_uv_rect(u_min, v_min, u_max, v_max);
            self.camera.update_uniform(&self.renderer.queue);
            self.render_frame()?;
            Ok(())
        }

        /// Reset both axes to the default full-matrix view.
        #[wasm_bindgen(js_name = resetZoom)]
        pub fn reset_zoom(&mut self) -> Result<(), JsValue> {
            self.camera.state.reset_zoom();
            self.camera.update_uniform(&self.renderer.queue);
            self.render_frame()?;
            Ok(())
        }

        /// Reset only the X axis zoom, keeping Y unchanged.
        #[wasm_bindgen(js_name = resetZoomX)]
        pub fn reset_zoom_x(&mut self) -> Result<(), JsValue> {
            self.camera.state.reset_zoom_x();
            self.camera.update_uniform(&self.renderer.queue);
            self.render_frame()?;
            Ok(())
        }

        /// Reset only the Y axis zoom, keeping X unchanged.
        #[wasm_bindgen(js_name = resetZoomY)]
        pub fn reset_zoom_y(&mut self) -> Result<(), JsValue> {
            self.camera.state.reset_zoom_y();
            self.camera.update_uniform(&self.renderer.queue);
            self.render_frame()?;
            Ok(())
        }

        /// Resize the canvas and update the rendering surface.
        #[wasm_bindgen]
        pub fn resize(&mut self, width: u32, height: u32) -> Result<(), JsValue> {
            let _timer = PerfTimer::new("resize", self.debug);
            self.camera
                .state
                .set_canvas_size(width as f32, height as f32);
            self.renderer
                .resize(width, height)
                .map_err(|e| JsValue::from_str(&e))?;
            self.camera.update_uniform(&self.renderer.queue);
            if self.matrix.is_some() {
                self.render_frame()?;
            }
            Ok(())
        }

        /// Clean up all GPU resources. Must be called when done.
        #[wasm_bindgen]
        pub fn destroy(self) {
            log::info!("LeibnizFast: destroyed");
        }
    }

    // Private methods (not exposed to JS)
    impl LeibnizFast {
        /// Internal method to set colormap and update the GPU texture.
        fn set_colormap_internal(&mut self, name: &str) -> Result<(), JsValue> {
            use colormap::ColormapProvider;
            let provider = colormap::BuiltinColormaps;
            let rgb_data = provider
                .get_colormap_rgb(name)
                .ok_or_else(|| JsValue::from_str(&format!("Unknown colormap: {name}")))?;

            let texture = colormap::ColormapTexture::new(
                &self.renderer.device,
                &self.renderer.queue,
                rgb_data,
            );
            self.colormap_texture = Some(texture);
            self.current_colormap = name.to_string();
            Ok(())
        }

        /// Render a single frame.
        fn render_frame(&mut self) -> Result<(), JsValue> {
            self.renderer
                .render_frame(&self.colormap_texture, &self.camera)
                .map_err(|e| JsValue::from_str(&e))
        }

        /// Handle hover by looking up the matrix value when retained data exists.
        fn handle_hover(&self, x: f32, y: f32) -> Result<(), JsValue> {
            let Some(ref callback) = self.hover_callback else {
                return Ok(());
            };

            let (rows, cols, value_available) = if let Some(ref jd) = self.js_data {
                (jd.rows(), jd.cols(), true)
            } else if let Some(ref matrix) = self.matrix {
                (matrix.rows(), matrix.cols(), false)
            } else {
                return Ok(());
            };

            if let Some((row, col)) = self.camera.state.screen_to_matrix(x, y, rows, cols) {
                let value = if let Some(ref jd) = self.js_data {
                    jd.get_value(row, col).unwrap_or(f32::NAN)
                } else {
                    f32::NAN
                };
                let this = JsValue::NULL;
                let _ = callback.call4(
                    &this,
                    &JsValue::from(row),
                    &JsValue::from(col),
                    &JsValue::from(value),
                    &JsValue::from_bool(value_available),
                );
            }
            Ok(())
        }
    }
}
