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
//! Supports two rendering paths:
//! - **WebGPU**: compute shader applies colormaps on GPU (fast)
//! - **WebGL2**: CPU applies colormaps, uploads RGBA texture (fallback)

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
    use crate::chunked_upload;
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

    /// Tracks an in-progress streaming upload.
    struct PendingUpload {
        /// JS-heap-backed data source being built incrementally
        js_data: matrix::JsDataSource,
        /// GPU buffer (WebGPU only — `None` on WebGL2)
        matrix_view: Option<matrix::MatrixView>,
        /// Total rows expected
        rows: u32,
        /// Total columns expected
        cols: u32,
        /// Next expected row index (enforces sequential appends)
        next_row: u32,
    }

    /// Main entry point for the library. Owns all GPU state and provides
    /// the public API for matrix visualization.
    ///
    /// Supports both WebGPU (compute shader path) and WebGL2 (CPU fallback).
    #[wasm_bindgen]
    pub struct LeibnizFast {
        renderer: renderer::Renderer,
        camera: camera::Camera,
        /// GPU-side matrix (only used on WebGPU path with compute shaders)
        matrix: Option<matrix::MatrixView>,
        colormap_texture: Option<colormap::ColormapTexture>,
        interaction: interaction::InteractionState,
        /// JS-heap-backed matrix data for tooltip lookups and colormap re-dispatch
        js_data: Option<matrix::JsDataSource>,
        /// JavaScript callback for hover events
        hover_callback: Option<js_sys::Function>,
        /// Current colormap name
        current_colormap: String,
        /// Cached colormap LUT for CPU fallback path
        current_colormap_lut: Option<&'static [[u8; 3]; 256]>,
        /// In-progress streaming upload, if any
        pending_upload: Option<PendingUpload>,
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
                current_colormap_lut: None,
                pending_upload: None,
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
            let expected_len = rows * cols;
            if data.length() != expected_len {
                return Err(JsValue::from_str(&format!(
                    "Data length {} does not match rows×cols = {}×{} = {}",
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
            self.js_data = Some(js_data);

            // Fast path: if the GPU pipeline already exists for the same dimensions,
            // skip MatrixView allocation and rebuild_pipelines(). Just re-dispatch the
            // colormap compute shader with the new data. This eliminates per-frame GPU
            // resource churn that causes visible flickering in real-time streaming.
            let same_dims = self.renderer.has_compute
                && self
                    .matrix
                    .as_ref()
                    .map_or(false, |m| m.rows() == rows && m.cols() == cols);

            if same_dims {
                // Fast path: colormap re-dispatch only — no pipeline rebuild
                if let (Some(ref matrix_view), Some(ref jd)) = (&self.matrix, &self.js_data) {
                    let (min_val, max_val) = jd.range();
                    let read_fn = |start: usize, buf: &mut [f32]| {
                        jd.read_range(start, buf);
                    };
                    self.renderer.apply_colormap_tiled(
                        matrix_view,
                        &read_fn,
                        cols,
                        min_val,
                        max_val,
                    );
                }
            } else {
                // Slow path: first call or dimension change — full pipeline rebuild
                if self.colormap_texture.is_none() || self.current_colormap_lut.is_none() {
                    self.set_colormap_internal(&self.current_colormap.clone())?;
                }

                self.camera.state.set_matrix_size(rows, cols);

                // WebGPU path: create staging buffer, build pipelines, apply colormap
                if self.renderer.has_compute {
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
                        let read_fn = |start: usize, buf: &mut [f32]| {
                            jd.read_range(start, buf);
                        };
                        self.renderer.apply_colormap_tiled(
                            matrix_view,
                            &read_fn,
                            cols,
                            min_val,
                            max_val,
                        );
                    }
                }
            }
            // setData always performs a full colormap from scratch, so the ring
            // cursor must be at 0 — any subsequent setDataScrolled call will use
            // a fresh JS WaterfallBuffer whose ringCursor is also 0.
            self.renderer.reset_ring_cursor();

            // On WebGL2: apply colormap on CPU and upload the colored texture
            self.upload_cpu_colormap_if_needed();

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
        /// Without a fixed range, falls back to full `setData`.
        #[wasm_bindgen(js_name = setDataScrolled)]
        pub fn set_data_scrolled(
            &mut self,
            data: js_sys::Float32Array,
            rows: u32,
            cols: u32,
            new_cols: u32,
        ) -> Result<(), JsValue> {
            let _timer = PerfTimer::new("set_data_scrolled", self.debug);

            // Require sticky_range — without it we'd need a full min/max scan
            // which defeats the purpose of the scrolled path.
            let (min_val, max_val) = match self.sticky_range {
                Some(range) => range,
                None => {
                    // Fall back to full setData if no fixed range
                    return self.set_data(data, rows, cols);
                }
            };

            let expected_len = rows * cols;
            if data.length() != expected_len {
                return Err(JsValue::from_str(&format!(
                    "Data length {} does not match rows×cols = {}×{} = {}",
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
            let same_dims = self.renderer.has_compute
                && self
                    .matrix
                    .as_ref()
                    .map_or(false, |m| m.rows() == rows && m.cols() == cols);

            if !same_dims {
                // First call or dimension change — do a full setData instead
                if self.colormap_texture.is_none() || self.current_colormap_lut.is_none() {
                    self.set_colormap_internal(&self.current_colormap.clone())?;
                }
                self.camera.state.set_matrix_size(rows, cols);

                if self.renderer.has_compute {
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

                    // Full colormap for the first frame — reset ring cursor
                    self.renderer.reset_ring_cursor();
                    if let (Some(ref matrix_view), Some(ref jd)) = (&self.matrix, &self.js_data) {
                        let read_fn = |start: usize, buf: &mut [f32]| {
                            jd.read_range(start, buf);
                        };
                        self.renderer.apply_colormap_tiled(
                            matrix_view,
                            &read_fn,
                            cols,
                            min_val,
                            max_val,
                        );
                    }
                }
                self.upload_cpu_colormap_if_needed();
                self.render_frame()?;
                return Ok(());
            }

            // Ring-buffer fast path: only colormap the new columns at the cursor,
            // O(rows × new_cols) regardless of total column count.
            if let (Some(ref matrix_view), Some(ref jd)) = (&self.matrix, &self.js_data) {
                let read_fn = |start: usize, buf: &mut [f32]| {
                    jd.read_range(start, buf);
                };
                self.renderer.apply_colormap_ring(
                    matrix_view,
                    &read_fn,
                    cols,
                    new_cols,
                    min_val,
                    max_val,
                );
            }

            self.render_frame()?;
            Ok(())
        }

        /// Set the colormap used for visualization.
        ///
        /// Re-dispatches the colormap from JS-heap data — works at any matrix size.
        #[wasm_bindgen(js_name = setColormap)]
        pub fn set_colormap(&mut self, name: &str) -> Result<(), JsValue> {
            let _timer = PerfTimer::new("set_colormap", self.debug);
            self.set_colormap_internal(name)?;

            if let Some(ref jd) = self.js_data {
                let (min_val, max_val) = jd.range();
                self.renderer
                    .rebuild_compute_bind_groups(&self.matrix, &self.colormap_texture)
                    .map_err(|e| JsValue::from_str(&e))?;

                // Re-dispatch colormap from JS-heap data
                if self.renderer.has_compute {
                    if let Some(ref matrix_view) = self.matrix {
                        let cols = jd.cols();
                        let read_fn = |start: usize, buf: &mut [f32]| {
                            jd.read_range(start, buf);
                        };
                        self.renderer.apply_colormap_tiled(
                            matrix_view,
                            &read_fn,
                            cols,
                            min_val,
                            max_val,
                        );
                    }
                }

                self.upload_cpu_colormap_if_needed();
                self.render_frame()?;
            }

            Ok(())
        }

        /// Set the data range for colormap mapping.
        ///
        /// Re-applies the colormap with the new range from JS-heap data.
        #[wasm_bindgen(js_name = setRange)]
        pub fn set_range(&mut self, min: f32, max: f32) -> Result<(), JsValue> {
            let _timer = PerfTimer::new("set_range", self.debug);
            // Persist as sticky range so future setData calls use it without rescanning
            self.sticky_range = Some((min, max));
            if let Some(ref mut jd) = self.js_data {
                jd.set_range(min, max);
            }

            if let Some(ref jd) = self.js_data {
                // Re-apply colormap with new range
                if self.renderer.has_compute {
                    if let Some(ref matrix_view) = self.matrix {
                        let read_fn = |start: usize, buf: &mut [f32]| {
                            jd.read_range(start, buf);
                        };
                        self.renderer.apply_colormap_tiled(
                            matrix_view,
                            &read_fn,
                            jd.cols(),
                            min,
                            max,
                        );
                    }
                }

                self.upload_cpu_colormap_if_needed();
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

            // Allocate JS-heap accumulator — no WASM memory pressure
            let js_data = matrix::JsDataSource::from_empty(rows, cols);

            let matrix_view = if self.renderer.has_compute {
                Some(
                    matrix::MatrixView::with_empty_buffer(
                        &self.renderer.device,
                        rows,
                        cols,
                        self.debug,
                    )
                    .map_err(|e| JsValue::from_str(&e))?,
                )
            } else {
                None
            };

            // Build pipelines early so compute shader is available for per-chunk dispatch
            if matrix_view.is_some() {
                if self.colormap_texture.is_none() || self.current_colormap_lut.is_none() {
                    self.set_colormap_internal(&self.current_colormap.clone())?;
                }

                // Temporarily move the view into self.matrix for rebuild_pipelines
                self.matrix = matrix_view;
                self.renderer
                    .rebuild_pipelines(
                        &self.matrix,
                        &self.colormap_texture,
                        &self.camera,
                        rows,
                        cols,
                    )
                    .map_err(|e| JsValue::from_str(&e))?;

                // Move it back into PendingUpload
                let matrix_view = self.matrix.take();
                self.pending_upload = Some(PendingUpload {
                    js_data,
                    matrix_view,
                    rows,
                    cols,
                    next_row: 0,
                });
            } else {
                self.pending_upload = Some(PendingUpload {
                    js_data,
                    matrix_view,
                    rows,
                    cols,
                    next_row: 0,
                });
            }

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

            let same_dims = self.renderer.has_compute
                && self
                    .matrix
                    .as_ref()
                    .map_or(false, |m| m.rows() == rows && m.cols() == cols)
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
                self.matrix = pending.matrix_view;
            }
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
        /// Copies chunk data to the JS-heap accumulator (for tooltip/colormap)
        /// and immediately dispatches the compute shader for this chunk.
        /// The min/max range uses a running estimate finalized in `end_data()`.
        #[wasm_bindgen(js_name = appendChunk)]
        pub fn append_chunk(&mut self, chunk: &[f32], start_row: u32) -> Result<(), JsValue> {
            let _timer = PerfTimer::new("append_chunk", self.debug);
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
            if cols > 0 && chunk.len() % cols != 0 {
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

            if start_row + chunk_rows > pending.rows {
                return Err(JsValue::from_str(&format!(
                    "Chunk would exceed total rows: start_row={start_row} + chunk_rows={chunk_rows} > {}",
                    pending.rows
                )));
            }

            // Copy chunk to JS-heap accumulator and track running min/max.
            // Colormap is applied once in end_data() with the final range,
            // ensuring consistent coloring across all chunks.
            let element_offset = start_row * pending.cols;
            pending.js_data.write_range(element_offset, chunk);
            pending.js_data.update_min_max(chunk);

            pending.next_row = start_row + chunk_rows;

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

            pending.js_data.finalize();

            self.js_data = Some(pending.js_data);
            self.matrix = pending.matrix_view;

            if self.colormap_texture.is_none() || self.current_colormap_lut.is_none() {
                self.set_colormap_internal(&self.current_colormap.clone())?;
            }

            self.camera.state.set_matrix_size(rows, cols);

            // Apply colormap in a single pass with the final min/max range.
            // This ensures consistent coloring across all rows (no per-chunk
            // range drift). Pipelines were already built in begin_data().
            if self.renderer.has_compute {
                if let (Some(ref matrix_view), Some(ref jd)) = (&self.matrix, &self.js_data) {
                    let (min_val, max_val) = jd.range();
                    let read_fn = |start: usize, buf: &mut [f32]| {
                        jd.read_range(start, buf);
                    };
                    self.renderer.apply_colormap_tiled(
                        matrix_view,
                        &read_fn,
                        cols,
                        min_val,
                        max_val,
                    );
                }
            }

            // CPU colormap upload for WebGL2 fallback
            self.upload_cpu_colormap_if_needed();

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
            if self.js_data.is_some() {
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

            // Cache the LUT for CPU fallback path
            self.current_colormap_lut = Some(rgb_data);

            let texture = colormap::ColormapTexture::new(
                &self.renderer.device,
                &self.renderer.queue,
                rgb_data,
            );
            self.colormap_texture = Some(texture);
            self.current_colormap = name.to_string();
            Ok(())
        }

        /// Apply colormap on CPU and upload to texture (WebGL2 fallback).
        ///
        /// Only runs when compute shaders are unavailable. Uses chunked uploads
        /// for large matrices to avoid allocating the full RGBA buffer at once.
        /// Reads from JS-heap data via `JsDataSource::read_range()`.
        /// Reuses the renderer's chunk_buffer to avoid per-chunk allocations.
        fn upload_cpu_colormap_if_needed(&mut self) {
            if self.renderer.has_compute {
                return; // WebGPU path uses compute shader instead
            }

            if let (Some(ref jd), Some(lut)) = (&self.js_data, self.current_colormap_lut) {
                let (min_val, max_val) = jd.range();
                let rows = jd.rows();
                let cols = jd.cols();

                let mut uploader = chunked_upload::ChunkedUploader::new(
                    rows,
                    cols,
                    chunked_upload::ChunkConfig { chunk_rows: None },
                );

                let mut rgba_buffer: Vec<u8> = Vec::new();

                while let Some((start_row, end_row)) = uploader.next_chunk_range() {
                    let start_idx = (start_row as usize) * (cols as usize);
                    let chunk_len = (end_row - start_row) as usize * (cols as usize);
                    let chunk_buf = self.renderer.chunk_buf_mut(chunk_len);
                    jd.read_range(start_idx, chunk_buf);
                    colormap::apply_colormap_cpu_into(
                        chunk_buf,
                        min_val,
                        max_val,
                        lut,
                        &mut rgba_buffer,
                    );
                    let chunk_rows = end_row - start_row;
                    self.renderer.upload_colored_texture_region(
                        &rgba_buffer,
                        cols,
                        chunk_rows,
                        start_row,
                    );
                    uploader.advance();
                }
            }
        }

        /// Render a single frame.
        fn render_frame(&mut self) -> Result<(), JsValue> {
            self.renderer
                .render_frame(&self.colormap_texture, &self.camera)
                .map_err(|e| JsValue::from_str(&e))
        }

        /// Handle hover by looking up the matrix value at the given screen position.
        ///
        /// Reads a single f32 from the JS-heap Float32Array — negligible overhead
        /// for hover events (~one JS/WASM boundary crossing per mouse move).
        fn handle_hover(&self, x: f32, y: f32) -> Result<(), JsValue> {
            if let (Some(ref callback), Some(ref jd)) = (&self.hover_callback, &self.js_data) {
                if let Some((row, col)) =
                    self.camera
                        .state
                        .screen_to_matrix(x, y, jd.rows(), jd.cols())
                {
                    if let Some(value) = jd.get_value(row, col) {
                        let this = JsValue::NULL;
                        let _ = callback.call3(
                            &this,
                            &JsValue::from(row),
                            &JsValue::from(col),
                            &JsValue::from(value),
                        );
                    }
                }
            }
            Ok(())
        }
    }
}
