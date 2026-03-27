//! # Camera
//!
//! 2D viewport management for matrix visualization. Handles zoom, pan, and
//! coordinate transforms between screen space, UV space, and matrix indices.
//!
//! Split into two parts:
//! - `CameraState`: Pure math, no GPU dependency — fully unit-testable
//! - `Camera`: Wraps `CameraState` and manages the GPU uniform buffer

/// Camera uniform data sent to the GPU fragment shader.
///
/// Represents the visible region of the matrix in UV coordinates [0,1]².
/// `uv_offset` is the top-left corner, `uv_scale` is the visible width/height.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CameraUniforms {
    /// UV offset (top-left corner of visible region)
    pub uv_offset: [f32; 2],
    /// UV scale (width/height of visible region in UV space)
    pub uv_scale: [f32; 2],
}

/// Scroll-to-zoom conversion factor (how much one scroll tick zooms).
const ZOOM_SENSITIVITY: f32 = 0.001;

/// Minimum zoom level (1.0 = full matrix visible).
const MIN_ZOOM: f32 = 1.0;

/// Maximum zoom level (1000x magnification).
const MAX_ZOOM: f32 = 1000.0;

/// Pure math camera state — no GPU dependency, fully unit-testable.
///
/// The camera defines a viewport over the matrix in UV coordinates [0,1]².
/// At zoom=1, the entire matrix is visible. Higher zoom shows less of the matrix.
///
/// The `center` is the UV coordinate at the center of the viewport.
/// `zoom` is a multiplier: zoom=2 means each matrix pixel is 2x bigger on screen.
pub struct CameraState {
    /// Center of the viewport in UV coordinates
    center: (f32, f32),
    /// Horizontal zoom level (1.0 = full matrix width visible)
    zoom_x: f32,
    /// Vertical zoom level (1.0 = full matrix height visible)
    zoom_y: f32,
    /// Canvas width in pixels
    canvas_width: f32,
    /// Canvas height in pixels
    canvas_height: f32,
    /// Matrix rows (for aspect ratio and coordinate transforms)
    matrix_rows: u32,
    /// Matrix columns (for aspect ratio and coordinate transforms)
    matrix_cols: u32,
}

impl CameraState {
    /// Create a new camera state showing the full matrix.
    pub fn new(canvas_width: f32, canvas_height: f32) -> Self {
        Self {
            center: (0.5, 0.5),
            zoom_x: 1.0,
            zoom_y: 1.0,
            canvas_width,
            canvas_height,
            matrix_rows: 1,
            matrix_cols: 1,
        }
    }

    /// Set the matrix dimensions (needed for aspect ratio correction).
    pub fn set_matrix_size(&mut self, rows: u32, cols: u32) {
        self.matrix_rows = rows;
        self.matrix_cols = cols;
    }

    /// Set the canvas size (needed for screen-to-UV transforms).
    pub fn set_canvas_size(&mut self, width: f32, height: f32) {
        self.canvas_width = width;
        self.canvas_height = height;
    }

    /// Get the camera uniforms for the GPU shader.
    ///
    /// Returns UV offset (top-left) and UV scale (visible width/height).
    /// At zoom=(1,1), offset=[0,0] and scale=[1,1] (entire matrix visible).
    pub fn get_uniforms(&self) -> CameraUniforms {
        let uv_width = 1.0 / self.zoom_x;
        let uv_height = 1.0 / self.zoom_y;

        let uv_left = self.center.0 - uv_width / 2.0;
        let uv_top = self.center.1 - uv_height / 2.0;

        CameraUniforms {
            uv_offset: [uv_left, uv_top],
            uv_scale: [uv_width, uv_height],
        }
    }

    /// Zoom both axes at a specific screen position, keeping that point fixed.
    ///
    /// `delta` is the scroll amount (positive = zoom in, negative = zoom out).
    /// The point under the cursor stays in place while the rest of the view scales.
    pub fn zoom_at(&mut self, screen_x: f32, screen_y: f32, delta: f32) {
        let uv_before = self.screen_to_uv(screen_x, screen_y);

        let zoom_factor = 1.0 + delta * ZOOM_SENSITIVITY;
        self.zoom_x = (self.zoom_x * zoom_factor).clamp(MIN_ZOOM, MAX_ZOOM);
        self.zoom_y = (self.zoom_y * zoom_factor).clamp(MIN_ZOOM, MAX_ZOOM);

        let uv_after = self.screen_to_uv(screen_x, screen_y);

        self.center.0 += uv_before.0 - uv_after.0;
        self.center.1 += uv_before.1 - uv_after.1;

        self.clamp_center();
    }

    /// Zoom only the X axis at a specific screen X position, keeping that
    /// point fixed horizontally. Y zoom is unchanged.
    pub fn zoom_at_x(&mut self, screen_x: f32, delta: f32) {
        let uv_before = self.screen_to_uv(screen_x, 0.0);

        let zoom_factor = 1.0 + delta * ZOOM_SENSITIVITY;
        self.zoom_x = (self.zoom_x * zoom_factor).clamp(MIN_ZOOM, MAX_ZOOM);

        let uv_after = self.screen_to_uv(screen_x, 0.0);

        self.center.0 += uv_before.0 - uv_after.0;

        self.clamp_center();
    }

    /// Zoom only the Y axis at a specific screen Y position, keeping that
    /// point fixed vertically. X zoom is unchanged.
    pub fn zoom_at_y(&mut self, screen_y: f32, delta: f32) {
        let uv_before = self.screen_to_uv(0.0, screen_y);

        let zoom_factor = 1.0 + delta * ZOOM_SENSITIVITY;
        self.zoom_y = (self.zoom_y * zoom_factor).clamp(MIN_ZOOM, MAX_ZOOM);

        let uv_after = self.screen_to_uv(0.0, screen_y);

        self.center.1 += uv_before.1 - uv_after.1;

        self.clamp_center();
    }

    /// Pan the viewport by a screen-space delta.
    ///
    /// Converts pixel movement to UV movement based on current zoom level.
    pub fn pan(&mut self, dx: f32, dy: f32) {
        let uv_dx = -dx / (self.canvas_width * self.zoom_x);
        let uv_dy = -dy / (self.canvas_height * self.zoom_y);

        self.center.0 += uv_dx;
        self.center.1 += uv_dy;

        self.clamp_center();
    }

    /// Pan only the X axis by a screen-space delta.
    pub fn pan_x(&mut self, dx: f32) {
        let uv_dx = -dx / (self.canvas_width * self.zoom_x);
        self.center.0 += uv_dx;
        self.clamp_center();
    }

    /// Pan only the Y axis by a screen-space delta.
    pub fn pan_y(&mut self, dy: f32) {
        let uv_dy = -dy / (self.canvas_height * self.zoom_y);
        self.center.1 += uv_dy;
        self.clamp_center();
    }

    /// Zoom to frame a specific UV rectangle.
    ///
    /// Sets center and zoom so the given UV region fills the viewport.
    /// Coordinates are in UV space [0,1]².
    pub fn zoom_to_uv_rect(&mut self, u_min: f32, v_min: f32, u_max: f32, v_max: f32) {
        let uv_width = (u_max - u_min).max(1e-6);
        let uv_height = (v_max - v_min).max(1e-6);

        self.zoom_x = (1.0 / uv_width).clamp(MIN_ZOOM, MAX_ZOOM);
        self.zoom_y = (1.0 / uv_height).clamp(MIN_ZOOM, MAX_ZOOM);

        self.center.0 = (u_min + u_max) / 2.0;
        self.center.1 = (v_min + v_max) / 2.0;

        self.clamp_center();
    }

    /// Reset both axes to the default full-matrix view.
    pub fn reset_zoom(&mut self) {
        self.zoom_x = 1.0;
        self.zoom_y = 1.0;
        self.center = (0.5, 0.5);
    }

    /// Reset only the X axis zoom, keeping Y unchanged.
    pub fn reset_zoom_x(&mut self) {
        self.zoom_x = 1.0;
        self.center.0 = 0.5;
        self.clamp_center();
    }

    /// Reset only the Y axis zoom, keeping X unchanged.
    pub fn reset_zoom_y(&mut self) {
        self.zoom_y = 1.0;
        self.center.1 = 0.5;
        self.clamp_center();
    }

    /// Convert screen coordinates to UV coordinates.
    ///
    /// Screen (0,0) is top-left, UV (0,0) is also top-left of the matrix.
    pub fn screen_to_uv(&self, screen_x: f32, screen_y: f32) -> (f32, f32) {
        let uniforms = self.get_uniforms();
        let u = uniforms.uv_offset[0] + (screen_x / self.canvas_width) * uniforms.uv_scale[0];
        let v = uniforms.uv_offset[1] + (screen_y / self.canvas_height) * uniforms.uv_scale[1];
        (u, v)
    }

    /// Convert screen coordinates to matrix (row, col) indices.
    ///
    /// Returns `None` if the screen position is outside the matrix bounds.
    pub fn screen_to_matrix(
        &self,
        screen_x: f32,
        screen_y: f32,
        rows: u32,
        cols: u32,
    ) -> Option<(u32, u32)> {
        let (u, v) = self.screen_to_uv(screen_x, screen_y);

        let col = (u * cols as f32).floor() as i32;
        let row = (v * rows as f32).floor() as i32;

        if row >= 0 && row < rows as i32 && col >= 0 && col < cols as i32 {
            Some((row as u32, col as u32))
        } else {
            None
        }
    }

    /// Clamp the center so the viewport doesn't go out of [0,1] bounds.
    fn clamp_center(&mut self) {
        let half_width = 0.5 / self.zoom_x;
        let half_height = 0.5 / self.zoom_y;

        self.center.0 = self.center.0.clamp(half_width, 1.0 - half_width);
        self.center.1 = self.center.1.clamp(half_height, 1.0 - half_height);
    }
}

/// Camera with GPU uniform buffer. Wraps `CameraState` for rendering.
///
/// Only available when compiling for WASM target.
#[cfg(target_arch = "wasm32")]
pub struct Camera {
    /// Public so the rest of the crate can call pure-math methods
    pub state: CameraState,
    /// GPU uniform buffer containing camera transform
    pub uniform_buffer: wgpu::Buffer,
}

#[cfg(target_arch = "wasm32")]
impl Camera {
    /// Create a new Camera with a GPU uniform buffer.
    pub fn new(device: &wgpu::Device, canvas_width: f32, canvas_height: f32) -> Self {
        use wgpu::util::DeviceExt;

        let state = CameraState::new(canvas_width, canvas_height);
        let uniforms = state.get_uniforms();

        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Camera Uniform Buffer"),
            contents: bytemuck::cast_slice(&[uniforms]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        Self {
            state,
            uniform_buffer,
        }
    }

    /// Write the current camera state to the GPU uniform buffer.
    pub fn update_uniform(&self, queue: &wgpu::Queue) {
        let uniforms = self.state.get_uniforms();
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::cast_slice(&[uniforms]));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_camera_shows_full_matrix() {
        let cam = CameraState::new(800.0, 600.0);
        let u = cam.get_uniforms();
        assert_eq!(u.uv_offset, [0.0, 0.0]);
        assert_eq!(u.uv_scale, [1.0, 1.0]);
    }

    #[test]
    fn test_zoom_2x_at_center() {
        let mut cam = CameraState::new(800.0, 600.0);
        // Manually set zoom to 2x (center stays at 0.5, 0.5)
        cam.zoom_x = 2.0;
        cam.zoom_y = 2.0;
        let u = cam.get_uniforms();
        assert!((u.uv_offset[0] - 0.25).abs() < 1e-6);
        assert!((u.uv_offset[1] - 0.25).abs() < 1e-6);
        assert!((u.uv_scale[0] - 0.5).abs() < 1e-6);
        assert!((u.uv_scale[1] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_zoom_at_keeps_point_fixed() {
        let mut cam = CameraState::new(800.0, 600.0);
        let screen_x = 200.0;
        let screen_y = 150.0;

        let uv_before = cam.screen_to_uv(screen_x, screen_y);
        cam.zoom_at(screen_x, screen_y, 500.0);
        let uv_after = cam.screen_to_uv(screen_x, screen_y);

        assert!(
            (uv_before.0 - uv_after.0).abs() < 1e-4,
            "U shifted: {} → {}",
            uv_before.0,
            uv_after.0
        );
        assert!(
            (uv_before.1 - uv_after.1).abs() < 1e-4,
            "V shifted: {} → {}",
            uv_before.1,
            uv_after.1
        );
    }

    #[test]
    fn test_pan_moves_center() {
        let mut cam = CameraState::new(800.0, 600.0);
        // Must zoom in first — at zoom=1 you see the full matrix and can't pan
        cam.zoom_x = 4.0;
        cam.zoom_y = 4.0;
        let center_before = cam.center;

        // Pan right by 80px → content shifts, center moves left in UV space
        cam.pan(80.0, 0.0);

        assert!(
            cam.center.0 < center_before.0,
            "Center should shift left: {} should be < {}",
            cam.center.0,
            center_before.0
        );
        assert!((cam.center.1 - center_before.1).abs() < 1e-6);
    }

    #[test]
    fn test_pan_clamps_to_bounds() {
        let mut cam = CameraState::new(800.0, 600.0);

        // Try to pan way past the left edge
        cam.pan(10000.0, 10000.0);
        let u = cam.get_uniforms();
        assert!(u.uv_offset[0] >= 0.0);
        assert!(u.uv_offset[1] >= 0.0);

        // Try to pan way past the right edge
        cam.pan(-20000.0, -20000.0);
        let u = cam.get_uniforms();
        assert!(u.uv_offset[0] + u.uv_scale[0] <= 1.0 + 1e-6);
        assert!(u.uv_offset[1] + u.uv_scale[1] <= 1.0 + 1e-6);
    }

    #[test]
    fn test_screen_to_matrix_default_camera() {
        let mut cam = CameraState::new(800.0, 600.0);
        cam.set_matrix_size(100, 200);

        let result = cam.screen_to_matrix(400.0, 300.0, 100, 200);
        assert_eq!(result, Some((50, 100)));

        let result = cam.screen_to_matrix(0.0, 0.0, 100, 200);
        assert_eq!(result, Some((0, 0)));
    }

    #[test]
    fn test_screen_to_matrix_out_of_bounds() {
        let cam = CameraState::new(800.0, 600.0);
        let result = cam.screen_to_matrix(-10.0, -10.0, 100, 200);
        assert_eq!(result, None);
    }

    #[test]
    fn test_screen_to_uv_corners() {
        let cam = CameraState::new(800.0, 600.0);

        let (u, v) = cam.screen_to_uv(0.0, 0.0);
        assert!((u).abs() < 1e-6);
        assert!((v).abs() < 1e-6);

        let (u, v) = cam.screen_to_uv(800.0, 600.0);
        assert!((u - 1.0).abs() < 1e-6);
        assert!((v - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_zoom_clamp_minimum() {
        let mut cam = CameraState::new(800.0, 600.0);
        cam.zoom_at(400.0, 300.0, -10000.0);
        assert!((cam.zoom_x - MIN_ZOOM).abs() < 1e-6);
        assert!((cam.zoom_y - MIN_ZOOM).abs() < 1e-6);
    }

    #[test]
    fn test_zoom_clamp_maximum() {
        let mut cam = CameraState::new(800.0, 600.0);
        // Zoom in with a massive delta to hit the ceiling
        cam.zoom_at(400.0, 300.0, 1_000_000.0);
        assert!((cam.zoom_x - MAX_ZOOM).abs() < 1e-6);
        assert!((cam.zoom_y - MAX_ZOOM).abs() < 1e-6);
    }

    #[test]
    fn test_set_canvas_size() {
        let mut cam = CameraState::new(800.0, 600.0);
        cam.set_canvas_size(1920.0, 1080.0);

        // screen_to_uv at bottom-right of new canvas should map to (1,1)
        let (u, v) = cam.screen_to_uv(1920.0, 1080.0);
        assert!((u - 1.0).abs() < 1e-6);
        assert!((v - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_set_matrix_size() {
        let mut cam = CameraState::new(800.0, 600.0);
        cam.set_matrix_size(500, 1000);

        // screen_to_matrix should use the new dimensions
        let result = cam.screen_to_matrix(400.0, 300.0, 500, 1000);
        assert_eq!(result, Some((250, 500)));
    }

    #[test]
    fn test_pan_zero_delta() {
        let mut cam = CameraState::new(800.0, 600.0);
        cam.zoom_x = 4.0;
        cam.zoom_y = 4.0;
        let center_before = cam.center;

        cam.pan(0.0, 0.0);

        assert!((cam.center.0 - center_before.0).abs() < 1e-6);
        assert!((cam.center.1 - center_before.1).abs() < 1e-6);
    }

    #[test]
    fn test_screen_to_matrix_bottom_right_edge() {
        let cam = CameraState::new(800.0, 600.0);
        // Exactly at the bottom-right edge → should be None (exclusive)
        let result = cam.screen_to_matrix(800.0, 600.0, 100, 200);
        assert_eq!(result, None);
    }

    #[test]
    fn test_independent_zoom_produces_different_scales() {
        let mut cam = CameraState::new(800.0, 600.0);
        cam.zoom_x = 2.0;
        cam.zoom_y = 4.0;
        let u = cam.get_uniforms();
        assert!((u.uv_scale[0] - 0.5).abs() < 1e-6, "X scale should be 0.5");
        assert!(
            (u.uv_scale[1] - 0.25).abs() < 1e-6,
            "Y scale should be 0.25"
        );
    }

    #[test]
    fn test_zoom_at_x_keeps_point_fixed_and_preserves_y() {
        let mut cam = CameraState::new(800.0, 600.0);
        let zoom_y_before = cam.zoom_y;

        let uv_before = cam.screen_to_uv(200.0, 150.0);
        cam.zoom_at_x(200.0, 500.0);
        let uv_after = cam.screen_to_uv(200.0, 150.0);

        assert!(
            (uv_before.0 - uv_after.0).abs() < 1e-4,
            "X UV should stay fixed: {} → {}",
            uv_before.0,
            uv_after.0
        );
        assert!(
            (cam.zoom_y - zoom_y_before).abs() < 1e-6,
            "Y zoom should be unchanged"
        );
    }

    #[test]
    fn test_zoom_at_y_keeps_point_fixed_and_preserves_x() {
        let mut cam = CameraState::new(800.0, 600.0);
        let zoom_x_before = cam.zoom_x;

        let uv_before = cam.screen_to_uv(200.0, 150.0);
        cam.zoom_at_y(150.0, 500.0);
        let uv_after = cam.screen_to_uv(200.0, 150.0);

        assert!(
            (uv_before.1 - uv_after.1).abs() < 1e-4,
            "Y UV should stay fixed: {} → {}",
            uv_before.1,
            uv_after.1
        );
        assert!(
            (cam.zoom_x - zoom_x_before).abs() < 1e-6,
            "X zoom should be unchanged"
        );
    }

    #[test]
    fn test_zoom_to_uv_rect() {
        let mut cam = CameraState::new(800.0, 600.0);
        cam.zoom_to_uv_rect(0.25, 0.25, 0.75, 0.75);

        let u = cam.get_uniforms();
        assert!((u.uv_offset[0] - 0.25).abs() < 1e-6);
        assert!((u.uv_offset[1] - 0.25).abs() < 1e-6);
        assert!((u.uv_scale[0] - 0.5).abs() < 1e-6);
        assert!((u.uv_scale[1] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_zoom_to_uv_rect_asymmetric() {
        let mut cam = CameraState::new(800.0, 600.0);
        cam.zoom_to_uv_rect(0.0, 0.25, 0.5, 0.75);

        let u = cam.get_uniforms();
        assert!((u.uv_scale[0] - 0.5).abs() < 1e-6, "X scale should be 0.5");
        assert!((u.uv_scale[1] - 0.5).abs() < 1e-6, "Y scale should be 0.5");
        assert!(
            (u.uv_offset[0] - 0.0).abs() < 1e-6,
            "X offset should be 0.0"
        );
        assert!(
            (u.uv_offset[1] - 0.25).abs() < 1e-6,
            "Y offset should be 0.25"
        );
    }

    #[test]
    fn test_pan_x_only_moves_horizontal() {
        let mut cam = CameraState::new(800.0, 600.0);
        cam.zoom_x = 4.0;
        cam.zoom_y = 4.0;
        let center_before = cam.center;

        cam.pan_x(80.0);

        assert!(
            (cam.center.0 - center_before.0).abs() > 1e-6,
            "X center should move"
        );
        assert!(
            (cam.center.1 - center_before.1).abs() < 1e-6,
            "Y center should not move"
        );
    }

    #[test]
    fn test_pan_y_only_moves_vertical() {
        let mut cam = CameraState::new(800.0, 600.0);
        cam.zoom_x = 4.0;
        cam.zoom_y = 4.0;
        let center_before = cam.center;

        cam.pan_y(60.0);

        assert!(
            (cam.center.0 - center_before.0).abs() < 1e-6,
            "X center should not move"
        );
        assert!(
            (cam.center.1 - center_before.1).abs() > 1e-6,
            "Y center should move"
        );
    }

    #[test]
    fn test_reset_zoom() {
        let mut cam = CameraState::new(800.0, 600.0);
        cam.zoom_x = 5.0;
        cam.zoom_y = 3.0;
        cam.center = (0.7, 0.3);

        cam.reset_zoom();

        assert!((cam.zoom_x - 1.0).abs() < 1e-6);
        assert!((cam.zoom_y - 1.0).abs() < 1e-6);
        assert!((cam.center.0 - 0.5).abs() < 1e-6);
        assert!((cam.center.1 - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_reset_zoom_x_keeps_y() {
        let mut cam = CameraState::new(800.0, 600.0);
        cam.zoom_x = 5.0;
        cam.zoom_y = 3.0;
        cam.center = (0.7, 0.5);

        cam.reset_zoom_x();

        assert!((cam.zoom_x - 1.0).abs() < 1e-6, "X zoom should reset to 1");
        assert!(
            (cam.zoom_y - 3.0).abs() < 1e-6,
            "Y zoom should be unchanged"
        );
        assert!((cam.center.0 - 0.5).abs() < 1e-6, "X center should reset");
    }

    #[test]
    fn test_reset_zoom_y_keeps_x() {
        let mut cam = CameraState::new(800.0, 600.0);
        cam.zoom_x = 5.0;
        cam.zoom_y = 3.0;
        cam.center = (0.5, 0.7);

        cam.reset_zoom_y();

        assert!(
            (cam.zoom_x - 5.0).abs() < 1e-6,
            "X zoom should be unchanged"
        );
        assert!((cam.zoom_y - 1.0).abs() < 1e-6, "Y zoom should reset to 1");
        assert!((cam.center.1 - 0.5).abs() < 1e-6, "Y center should reset");
    }
}
