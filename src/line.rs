//! Pure line-chart helpers shared by the WASM renderer and unit tests.

/// Line X coordinates are generated from `x_start + index * x_step`.
pub const LINE_X_MODE_LINEAR: u32 = 0;
/// Line X coordinates are read from the shared explicit X buffer.
pub const LINE_X_MODE_EXPLICIT: u32 = 1;

/// GPU uniform block for line rendering.
///
/// Layout must match `LineParams` in `src/shaders/line.wgsl`.
/// The first four fields are `u32`, followed by twelve `f32` values, for a
/// total of 64 bytes and uniform-buffer-friendly 16-byte alignment.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct LineParams {
    pub sample_count: u32,
    pub series_count: u32,
    pub ring_cursor: u32,
    pub x_mode: u32,
    pub x_min: f32,
    pub x_max: f32,
    pub y_min: f32,
    pub y_max: f32,
    pub uv_x_offset: f32,
    pub uv_y_offset: f32,
    pub uv_x_scale: f32,
    pub uv_y_scale: f32,
    pub x_start: f32,
    pub x_step: f32,
    pub _pad0: f32,
    pub _pad1: f32,
}

impl Default for LineParams {
    fn default() -> Self {
        Self {
            sample_count: 0,
            series_count: 0,
            ring_cursor: 0,
            x_mode: LINE_X_MODE_LINEAR,
            x_min: 0.0,
            x_max: 1.0,
            y_min: 0.0,
            y_max: 1.0,
            uv_x_offset: 0.0,
            uv_y_offset: 0.0,
            uv_x_scale: 1.0,
            uv_y_scale: 1.0,
            x_start: 0.0,
            x_step: 1.0,
            _pad0: 0.0,
            _pad1: 0.0,
        }
    }
}

/// Return the physical ring-buffer index for a logical sample index.
pub fn line_ring_physical_index(
    logical_index: u32,
    sample_count: u32,
    ring_cursor: u32,
) -> Option<u32> {
    if sample_count == 0 || logical_index >= sample_count || ring_cursor >= sample_count {
        return None;
    }
    Some((ring_cursor + logical_index) % sample_count)
}

/// Normalize a line color from public API units into WebGPU units.
pub fn normalize_rgba_255_alpha(
    red: f32,
    green: f32,
    blue: f32,
    alpha: f32,
) -> Result<[f32; 4], &'static str> {
    if !(0.0..=255.0).contains(&red)
        || !(0.0..=255.0).contains(&green)
        || !(0.0..=255.0).contains(&blue)
        || !(0.0..=1.0).contains(&alpha)
        || !red.is_finite()
        || !green.is_finite()
        || !blue.is_finite()
        || !alpha.is_finite()
    {
        return Err("RGBA colors must use RGB values in 0..255 and alpha in 0..1.");
    }

    Ok([red / 255.0, green / 255.0, blue / 255.0, alpha])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ring_index_wraps_from_cursor() {
        let mapped: Vec<u32> = (0..5)
            .map(|idx| line_ring_physical_index(idx, 5, 3).unwrap())
            .collect();
        assert_eq!(mapped, vec![3, 4, 0, 1, 2]);
    }

    #[test]
    fn ring_index_rejects_invalid_values() {
        assert_eq!(line_ring_physical_index(0, 0, 0), None);
        assert_eq!(line_ring_physical_index(5, 5, 0), None);
        assert_eq!(line_ring_physical_index(0, 5, 5), None);
    }

    #[test]
    fn color_normalization_uses_byte_rgb_and_unit_alpha() {
        let color = normalize_rgba_255_alpha(128.0, 64.0, 255.0, 0.5).unwrap();
        assert!((color[0] - 128.0 / 255.0).abs() < 1e-6);
        assert!((color[1] - 64.0 / 255.0).abs() < 1e-6);
        assert!((color[2] - 1.0).abs() < 1e-6);
        assert!((color[3] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn color_normalization_rejects_out_of_range_values() {
        assert!(normalize_rgba_255_alpha(-1.0, 0.0, 0.0, 1.0).is_err());
        assert!(normalize_rgba_255_alpha(0.0, 256.0, 0.0, 1.0).is_err());
        assert!(normalize_rgba_255_alpha(0.0, 0.0, 0.0, 1.1).is_err());
        assert!(normalize_rgba_255_alpha(f32::NAN, 0.0, 0.0, 1.0).is_err());
    }

    #[test]
    fn line_params_match_shader_layout_size() {
        assert_eq!(std::mem::size_of::<LineParams>(), 64);
    }
}
