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
    use crate::colormap;
    use crate::interaction;
    use crate::matrix;
    use crate::renderer;

    /// Initialize panic hook for better error messages in the browser console.
    fn init_logging() {
        console_error_panic_hook::set_once();
        console_log::init_with_level(log::Level::Info).ok();
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
            })
        }

        /// Set the matrix data to visualize.
        ///
        /// `data` is a flat Float32Array in row-major order.
        /// `rows` and `cols` specify the matrix dimensions.
        #[wasm_bindgen(js_name = setData)]
        pub fn set_data(&mut self, data: &[f32], rows: u32, cols: u32) -> Result<(), JsValue> {
            let matrix_data = matrix::MatrixData::new(data.to_vec(), rows, cols);
            self.matrix_data = Some(matrix_data);

            // Only create GPU storage buffer on WebGPU path (compute shaders available)
            if self.renderer.has_compute {
                let matrix_view = matrix::MatrixView::new(
                    &self.renderer.device,
                    &self.renderer.queue,
                    data,
                    rows,
                    cols,
                );
                self.matrix = Some(matrix_view);
            }

            if self.colormap_texture.is_none() || self.current_colormap_lut.is_none() {
                self.set_colormap_internal(&self.current_colormap.clone())?;
            }

            self.camera.state.set_matrix_size(rows, cols);

            self.renderer
                .rebuild_pipelines(&self.matrix, &self.colormap_texture, &self.camera, rows, cols)
                .map_err(|e| JsValue::from_str(&e))?;

            // On WebGL2: apply colormap on CPU and upload the colored texture
            self.upload_cpu_colormap_if_needed();

            self.render_frame()?;

            Ok(())
        }

        /// Set the colormap used for visualization.
        #[wasm_bindgen(js_name = setColormap)]
        pub fn set_colormap(&mut self, name: &str) -> Result<(), JsValue> {
            self.set_colormap_internal(name)?;

            if let Some(ref md) = self.matrix_data {
                let (rows, cols) = (md.rows(), md.cols());
                self.renderer
                    .rebuild_pipelines(&self.matrix, &self.colormap_texture, &self.camera, rows, cols)
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
        /// Only runs when compute shaders are unavailable.
        fn upload_cpu_colormap_if_needed(&self) {
            if self.renderer.has_compute {
                return; // WebGPU path uses compute shader instead
            }

            if let (Some(ref matrix_data), Some(lut)) =
                (&self.matrix_data, self.current_colormap_lut)
            {
                let (min_val, max_val) = matrix_data.range();
                let rgba =
                    colormap::apply_colormap_cpu(matrix_data.raw_data(), min_val, max_val, lut);
                self.renderer
                    .upload_colored_texture(&rgba, matrix_data.cols(), matrix_data.rows());
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
