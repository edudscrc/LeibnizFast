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

// GPU/WASM modules — only compiled for wasm32 target
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
    use crate::renderer;

    /// Initialize panic hook for better error messages in the browser console.
    fn init_logging() {
        console_error_panic_hook::set_once();
        console_log::init_with_level(log::Level::Info).ok();
    }

    /// Tracks an in-progress streaming upload.
    struct PendingUpload {
        /// CPU-side matrix being built incrementally
        matrix_data: matrix::MatrixData,
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
        /// CPU-side matrix data for tooltip lookups and WebGL2 fallback
        matrix_data: Option<matrix::MatrixData>,
        /// JavaScript callback for hover events
        hover_callback: Option<js_sys::Function>,
        /// Current colormap name
        current_colormap: String,
        /// Cached colormap LUT for CPU fallback path
        current_colormap_lut: Option<&'static [[u8; 3]; 256]>,
        /// In-progress streaming upload, if any
        pending_upload: Option<PendingUpload>,
    }

    #[wasm_bindgen]
    impl LeibnizFast {
        /// Create a new LeibnizFast instance attached to the given canvas element.
        #[wasm_bindgen]
        pub async fn create(
            canvas: web_sys::HtmlCanvasElement,
            colormap: Option<String>,
        ) -> Result<LeibnizFast, JsValue> {
            init_logging();
            log::info!("LeibnizFast: initializing...");

            let colormap_name = colormap.unwrap_or_else(|| "viridis".to_string());

            let renderer = renderer::Renderer::new(&canvas)
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
                matrix_data: None,
                hover_callback: None,
                current_colormap: colormap_name,
                current_colormap_lut: None,
                pending_upload: None,
            })
        }

        /// Set the matrix data to visualize.
        ///
        /// `data` is a flat Float32Array in row-major order.
        /// `rows` and `cols` specify the matrix dimensions.
        /// Internally uses chunked upload for large matrices.
        /// For matrices exceeding `max_buffer_size`, uses a staging buffer
        /// with chunked compute shader processing.
        #[wasm_bindgen(js_name = setData)]
        pub fn set_data(&mut self, data: &[f32], rows: u32, cols: u32) -> Result<(), JsValue> {
            // Determine if CPU-side data should use range-only mode
            const MAX_CPU_MATRIX_BYTES: u64 = 2 * 1024 * 1024 * 1024;
            let data_bytes = (rows as u64) * (cols as u64) * 4;
            let use_range_only = data_bytes > MAX_CPU_MATRIX_BYTES;

            let mut matrix_data = if use_range_only {
                if !self.renderer.has_compute {
                    return Err(JsValue::from_str(
                        "Matrix is too large for CPU-side storage and this device lacks \
                         compute shaders (WebGL2 fallback). WebGPU is required.",
                    ));
                }
                log::warn!(
                    "setData: matrix {}×{} exceeds CPU limit, using range-only mode.",
                    rows, cols
                );
                matrix::MatrixData::range_only(rows, cols)
            } else {
                matrix::MatrixData::with_capacity(rows, cols)
            };

            // Only create GPU storage buffer on WebGPU path (compute shaders available)
            if self.renderer.has_compute {
                let matrix_view =
                    matrix::MatrixView::with_empty_buffer(&self.renderer.device, rows, cols)
                        .map_err(|e| JsValue::from_str(&e))?;

                let is_staging = matrix_view.is_staging();

                // Upload data in chunks
                let mut uploader = chunked_upload::ChunkedUploader::new(
                    rows,
                    cols,
                    chunked_upload::ChunkConfig { chunk_rows: None },
                );

                if is_staging {
                    // Staging path: we need pipelines built first so compute shader
                    // is available for per-chunk dispatch
                    self.matrix = Some(matrix_view);

                    if self.colormap_texture.is_none() || self.current_colormap_lut.is_none() {
                        self.set_colormap_internal(&self.current_colormap.clone())?;
                    }

                    self.renderer
                        .rebuild_pipelines(
                            &self.matrix,
                            &self.colormap_texture,
                            &self.camera,
                            rows,
                            cols,
                        )
                        .map_err(|e| JsValue::from_str(&e))?;

                    // Process via staging: compute min/max first pass, then apply colormap
                    // For setData we have all data, so compute range first
                    while let Some((start_row, end_row)) = uploader.next_chunk_range() {
                        let start_idx = (start_row as usize) * (cols as usize);
                        let end_idx = (end_row as usize) * (cols as usize);
                        let chunk = &data[start_idx..end_idx];
                        matrix_data.append_rows(chunk);
                        uploader.advance();
                    }
                    matrix_data.finalize();
                    let (min_val, max_val) = matrix_data.range();

                    // Now apply colormap chunk by chunk through the staging buffer
                    if let Some(ref matrix_view) = self.matrix {
                        self.renderer.apply_colormap_staged(
                            matrix_view, data, rows, cols, min_val, max_val,
                        );
                    }
                } else {
                    // Standard path: full buffer, upload all chunks
                    while let Some((start_row, end_row)) = uploader.next_chunk_range() {
                        let start_idx = (start_row as usize) * (cols as usize);
                        let end_idx = (end_row as usize) * (cols as usize);
                        let chunk = &data[start_idx..end_idx];
                        matrix_data.append_rows(chunk);
                        matrix_view.write_chunk(
                            &self.renderer.queue, start_row, cols, chunk,
                        );
                        uploader.advance();
                    }

                    matrix_data.finalize();
                    let (min_val, max_val) = matrix_data.range();
                    let mut matrix_view = matrix_view;
                    matrix_view.update_params(min_val, max_val, &self.renderer.queue);
                    self.renderer.set_colormap_applied(false);
                    self.matrix = Some(matrix_view);
                }
            } else {
                // WebGL2 path: just build MatrixData directly (no GPU storage buffer)
                matrix_data.append_rows(data);
                matrix_data.finalize();
            }

            self.matrix_data = Some(matrix_data);

            if self.colormap_texture.is_none() || self.current_colormap_lut.is_none() {
                self.set_colormap_internal(&self.current_colormap.clone())?;
            }

            self.camera.state.set_matrix_size(rows, cols);

            // Only rebuild pipelines if not already done (staging path builds early)
            if !self.matrix.as_ref().map_or(false, |m| m.is_staging()) {
                self.renderer
                    .rebuild_pipelines(
                        &self.matrix,
                        &self.colormap_texture,
                        &self.camera,
                        rows,
                        cols,
                    )
                    .map_err(|e| JsValue::from_str(&e))?;
            }

            // On WebGL2: apply colormap on CPU and upload the colored texture
            self.upload_cpu_colormap_if_needed();

            self.render_frame()?;

            Ok(())
        }

        /// Set the colormap used for visualization.
        #[wasm_bindgen(js_name = setColormap)]
        pub fn set_colormap(&mut self, name: &str) -> Result<(), JsValue> {
            self.set_colormap_internal(name)?;

            // Staged data can't have its colormap reapplied (raw data not retained)
            if self.matrix.as_ref().map_or(false, |m| m.is_staging()) {
                return Err(JsValue::from_str(
                    "Cannot change colormap on a staged matrix. \
                     Reload the data after selecting the desired colormap.",
                ));
            }

            // Reset the colormap_applied flag so render_frame runs the compute pass
            self.renderer.set_colormap_applied(false);

            if let Some(ref md) = self.matrix_data {
                let (rows, cols) = (md.rows(), md.cols());
                self.renderer
                    .rebuild_pipelines(
                        &self.matrix,
                        &self.colormap_texture,
                        &self.camera,
                        rows,
                        cols,
                    )
                    .map_err(|e| JsValue::from_str(&e))?;
                self.upload_cpu_colormap_if_needed();
                self.render_frame()?;
            }

            Ok(())
        }

        /// Set the data range for colormap mapping.
        #[wasm_bindgen(js_name = setRange)]
        pub fn set_range(&mut self, min: f32, max: f32) -> Result<(), JsValue> {
            if let Some(ref mut matrix_data) = self.matrix_data {
                matrix_data.set_range(min, max);
            }
            if let Some(ref mut matrix_view) = self.matrix {
                matrix_view.set_range(min, max, &self.renderer.queue);
            }

            if self.matrix_data.is_some() {
                self.upload_cpu_colormap_if_needed();
                self.render_frame()?;
            }

            Ok(())
        }

        /// Begin a streaming data upload.
        ///
        /// Allocates CPU and GPU buffers for `rows × cols` elements.
        /// For matrices exceeding `max_buffer_size`, creates a staging buffer
        /// and sets up pipelines early for per-chunk compute shader processing.
        /// Use `append_chunk()` to upload data, then `end_data()` to finalize.
        /// Errors if an upload is already in progress.
        #[wasm_bindgen(js_name = beginData)]
        pub fn begin_data(&mut self, rows: u32, cols: u32) -> Result<(), JsValue> {
            if self.pending_upload.is_some() {
                return Err(JsValue::from_str(
                    "A streaming upload is already in progress. Call endData() first.",
                ));
            }

            // 2 GB threshold — beyond this, WASM's 4 GB address space can't hold
            // the full CPU-side Vec<f32> plus other allocations.
            const MAX_CPU_MATRIX_BYTES: u64 = 2 * 1024 * 1024 * 1024;
            let data_bytes = (rows as u64) * (cols as u64) * 4;
            let use_range_only = data_bytes > MAX_CPU_MATRIX_BYTES;

            if use_range_only && !self.renderer.has_compute {
                return Err(JsValue::from_str(
                    "Matrix is too large for CPU-side storage and this device lacks \
                     compute shaders (WebGL2 fallback). WebGPU is required for \
                     matrices exceeding ~2 GB.",
                ));
            }

            let matrix_data = if use_range_only {
                log::warn!(
                    "Matrix {}×{} ({} bytes) exceeds CPU storage limit. \
                     Using range-only mode — tooltip hover will be unavailable.",
                    rows, cols, data_bytes
                );
                matrix::MatrixData::range_only(rows, cols)
            } else {
                matrix::MatrixData::with_capacity(rows, cols)
            };

            let matrix_view = if self.renderer.has_compute {
                Some(
                    matrix::MatrixView::with_empty_buffer(&self.renderer.device, rows, cols)
                        .map_err(|e| JsValue::from_str(&e))?,
                )
            } else {
                None
            };

            // For staging mode: build pipelines early so the compute shader
            // is available for per-chunk processing in append_chunk()
            if matrix_view.as_ref().map_or(false, |m| m.is_staging()) {
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
                    matrix_data,
                    matrix_view,
                    rows,
                    cols,
                    next_row: 0,
                });
            } else {
                self.pending_upload = Some(PendingUpload {
                    matrix_data,
                    matrix_view,
                    rows,
                    cols,
                    next_row: 0,
                });
            }

            Ok(())
        }

        /// Append a chunk of rows to the in-progress streaming upload.
        ///
        /// `chunk` must contain a whole number of rows. `start_row` must
        /// match the next expected row (sequential ordering required).
        ///
        /// In staging mode, immediately dispatches the compute shader for
        /// this chunk (applying the colormap on the fly). The min/max range
        /// uses a running estimate that is finalized in `end_data()`.
        #[wasm_bindgen(js_name = appendChunk)]
        pub fn append_chunk(&mut self, chunk: &[f32], start_row: u32) -> Result<(), JsValue> {
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

            pending.matrix_data.append_rows(chunk);

            if let Some(ref matrix_view) = pending.matrix_view {
                if matrix_view.is_staging() {
                    // Staging mode: process this chunk through the compute shader now.
                    // Use the running min/max (will be corrected at end_data if needed,
                    // but for initial display the running values are good enough).
                    let (min_val, max_val) = pending.matrix_data.range();
                    self.renderer.apply_colormap_staged_chunk(
                        matrix_view,
                        chunk,
                        chunk_rows,
                        start_row,
                        pending.cols,
                        min_val,
                        max_val,
                    );
                } else {
                    // Full buffer mode: write data to the buffer at the correct offset
                    matrix_view.write_chunk(
                        &self.renderer.queue, start_row, pending.cols, chunk,
                    );
                }
            }

            pending.next_row = start_row + chunk_rows;

            Ok(())
        }

        /// Finalize a streaming upload.
        ///
        /// Computes final min/max, rebuilds pipelines, and renders.
        /// Errors if the upload is incomplete (not all rows uploaded).
        #[wasm_bindgen(js_name = endData)]
        pub fn end_data(&mut self) -> Result<(), JsValue> {
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
            let is_staging = pending.matrix_view.as_ref().map_or(false, |m| m.is_staging());

            pending.matrix_data.finalize();
            let (min_val, max_val) = pending.matrix_data.range();

            if let Some(ref mut matrix_view) = pending.matrix_view {
                if !is_staging {
                    // Full buffer: update the params uniform with final range
                    matrix_view.update_params(min_val, max_val, &self.renderer.queue);
                }
                // For staging: colormap was already applied per-chunk during append_chunk
            }

            self.matrix_data = Some(pending.matrix_data);
            self.matrix = pending.matrix_view;

            if self.colormap_texture.is_none() || self.current_colormap_lut.is_none() {
                self.set_colormap_internal(&self.current_colormap.clone())?;
            }

            self.camera.state.set_matrix_size(rows, cols);

            if is_staging {
                // Staging path: pipelines already built in begin_data,
                // colormap already applied per-chunk
                self.renderer.set_colormap_applied(true);
            } else {
                self.renderer.set_colormap_applied(false);
                self.renderer
                    .rebuild_pipelines(
                        &self.matrix,
                        &self.colormap_texture,
                        &self.camera,
                        rows,
                        cols,
                    )
                    .map_err(|e| JsValue::from_str(&e))?;

                // CPU colormap upload only works when we have full data on CPU
                if !self
                    .matrix_data
                    .as_ref()
                    .map_or(false, |md| md.is_range_only())
                {
                    self.upload_cpu_colormap_if_needed();
                }
            }

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
            self.camera.state.zoom_at(x, y, delta);
            self.camera.update_uniform(&self.renderer.queue);
            self.render_frame()?;
            Ok(())
        }

        /// Resize the canvas and update the rendering surface.
        #[wasm_bindgen]
        pub fn resize(&mut self, width: u32, height: u32) -> Result<(), JsValue> {
            self.camera
                .state
                .set_canvas_size(width as f32, height as f32);
            self.renderer
                .resize(width, height)
                .map_err(|e| JsValue::from_str(&e))?;
            self.camera.update_uniform(&self.renderer.queue);
            if self.matrix_data.is_some() {
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
        fn upload_cpu_colormap_if_needed(&self) {
            if self.renderer.has_compute {
                return; // WebGPU path uses compute shader instead
            }

            if let (Some(ref matrix_data), Some(lut)) =
                (&self.matrix_data, self.current_colormap_lut)
            {
                let (min_val, max_val) = matrix_data.range();
                let rows = matrix_data.rows();
                let cols = matrix_data.cols();
                let raw = matrix_data.raw_data();

                let mut uploader = chunked_upload::ChunkedUploader::new(
                    rows,
                    cols,
                    chunked_upload::ChunkConfig { chunk_rows: None },
                );

                while let Some((start_row, end_row)) = uploader.next_chunk_range() {
                    let start_idx = (start_row as usize) * (cols as usize);
                    let end_idx = (end_row as usize) * (cols as usize);
                    let chunk_data = &raw[start_idx..end_idx];
                    let rgba = colormap::apply_colormap_cpu(chunk_data, min_val, max_val, lut);
                    let chunk_rows = end_row - start_row;
                    self.renderer
                        .upload_colored_texture_region(&rgba, cols, chunk_rows, start_row);
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
        fn handle_hover(&self, x: f32, y: f32) -> Result<(), JsValue> {
            if let (Some(ref callback), Some(ref matrix_data)) =
                (&self.hover_callback, &self.matrix_data)
            {
                // Skip lookup in range-only mode (no CPU data stored)
                if matrix_data.is_range_only() {
                    return Ok(());
                }
                if let Some((row, col)) =
                    self.camera
                        .state
                        .screen_to_matrix(x, y, matrix_data.rows(), matrix_data.cols())
                {
                    if let Some(value) = matrix_data.get_value(row, col) {
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
