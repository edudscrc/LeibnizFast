//! # Matrix
//!
//! Matrix data management, both CPU-side (for tooltip lookups) and GPU-side
//! (storage buffer for compute shader input).
//!
//! Split into:
//! - `MatrixData`: CPU-side data with value lookup and range computation (always available)
//! - `MatrixView`: GPU buffer management for the matrix data (WASM-only)

/// Uniform parameters for the compute shader.
///
/// Contains matrix dimensions and data range for normalization.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct MatrixParams {
    /// Number of rows in the matrix
    pub rows: u32,
    /// Number of columns in the matrix
    pub cols: u32,
    /// Minimum data value (maps to first colormap color)
    pub min_val: f32,
    /// Maximum data value (maps to last colormap color)
    pub max_val: f32,
}

/// CPU-side matrix data for tooltip lookups and range computation.
///
/// Keeps a copy of the matrix data in memory to avoid async GPU readback
/// when the user hovers over cells.
pub struct MatrixData {
    /// Row-major flat data
    data: Vec<f32>,
    rows: u32,
    cols: u32,
    /// Data range for colormap normalization
    min_val: f32,
    max_val: f32,
}

impl MatrixData {
    /// Create new matrix data from a flat row-major array.
    ///
    /// Automatically computes min/max from the data.
    pub fn new(data: Vec<f32>, rows: u32, cols: u32) -> Self {
        let (min_val, max_val) = Self::compute_range(&data);
        Self {
            data,
            rows,
            cols,
            min_val,
            max_val,
        }
    }

    /// Get the value at (row, col). Returns `None` if out of bounds.
    pub fn get_value(&self, row: u32, col: u32) -> Option<f32> {
        if row < self.rows && col < self.cols {
            let idx = (row * self.cols + col) as usize;
            self.data.get(idx).copied()
        } else {
            None
        }
    }

    /// Get the number of rows.
    pub fn rows(&self) -> u32 {
        self.rows
    }

    /// Get the number of columns.
    pub fn cols(&self) -> u32 {
        self.cols
    }

    /// Get a reference to the raw data (for CPU colormap application).
    pub fn raw_data(&self) -> &[f32] {
        &self.data
    }

    /// Get the current data range (min, max).
    pub fn range(&self) -> (f32, f32) {
        (self.min_val, self.max_val)
    }

    /// Override the auto-computed range with explicit values.
    pub fn set_range(&mut self, min: f32, max: f32) {
        self.min_val = min;
        self.max_val = max;
    }

    /// Compute min and max from the data, ignoring NaN values.
    pub(crate) fn compute_range(data: &[f32]) -> (f32, f32) {
        let mut min = f32::INFINITY;
        let mut max = f32::NEG_INFINITY;
        for &v in data {
            if v.is_finite() {
                if v < min {
                    min = v;
                }
                if v > max {
                    max = v;
                }
            }
        }
        // Handle edge case: all NaN or empty data
        if min.is_infinite() {
            min = 0.0;
            max = 1.0;
        }
        (min, max)
    }
}

/// GPU-side matrix view with storage buffer and parameters uniform.
///
/// Only available when compiling for WASM target.
#[cfg(target_arch = "wasm32")]
pub struct MatrixView {
    /// Storage buffer containing the raw float data
    pub data_buffer: wgpu::Buffer,
    /// Uniform buffer containing matrix dimensions and range
    pub params_buffer: wgpu::Buffer,
    rows: u32,
    cols: u32,
    min_val: f32,
    max_val: f32,
}

#[cfg(target_arch = "wasm32")]
impl MatrixView {
    /// Create a new MatrixView, uploading data to GPU buffers.
    pub fn new(
        device: &wgpu::Device,
        _queue: &wgpu::Queue,
        data: &[f32],
        rows: u32,
        cols: u32,
    ) -> Self {
        use wgpu::util::DeviceExt;

        let (min_val, max_val) = MatrixData::compute_range(data);

        let data_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Matrix Data Buffer"),
            contents: bytemuck::cast_slice(data),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let params = MatrixParams {
            rows,
            cols,
            min_val,
            max_val,
        };
        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Matrix Params Buffer"),
            contents: bytemuck::cast_slice(&[params]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        Self {
            data_buffer,
            params_buffer,
            rows,
            cols,
            min_val,
            max_val,
        }
    }

    /// Get the number of rows.
    pub fn rows(&self) -> u32 {
        self.rows
    }

    /// Get the number of columns.
    pub fn cols(&self) -> u32 {
        self.cols
    }

    /// Update the data range and write to the GPU uniform buffer.
    pub fn set_range(&mut self, min: f32, max: f32, queue: &wgpu::Queue) {
        self.min_val = min;
        self.max_val = max;
        let params = MatrixParams {
            rows: self.rows,
            cols: self.cols,
            min_val: self.min_val,
            max_val: self.max_val,
        };
        queue.write_buffer(&self.params_buffer, 0, bytemuck::cast_slice(&[params]));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_value_valid_indices() {
        // 2x3 matrix: [[1, 2, 3], [4, 5, 6]]
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let m = MatrixData::new(data, 2, 3);

        assert_eq!(m.get_value(0, 0), Some(1.0));
        assert_eq!(m.get_value(0, 2), Some(3.0));
        assert_eq!(m.get_value(1, 0), Some(4.0));
        assert_eq!(m.get_value(1, 2), Some(6.0));
    }

    #[test]
    fn test_get_value_out_of_bounds() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let m = MatrixData::new(data, 2, 3);

        assert_eq!(m.get_value(2, 0), None);
        assert_eq!(m.get_value(0, 3), None);
        assert_eq!(m.get_value(5, 5), None);
    }

    #[test]
    fn test_auto_min_max() {
        let data = vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0];
        let m = MatrixData::new(data, 2, 3);
        let (min, max) = m.range();
        assert!((min - 1.0).abs() < 1e-6);
        assert!((max - 9.0).abs() < 1e-6);
    }

    #[test]
    fn test_auto_min_max_with_nan() {
        let data = vec![f32::NAN, 2.0, f32::NAN, 5.0];
        let m = MatrixData::new(data, 2, 2);
        let (min, max) = m.range();
        assert!((min - 2.0).abs() < 1e-6);
        assert!((max - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_auto_min_max_all_nan() {
        let data = vec![f32::NAN, f32::NAN];
        let m = MatrixData::new(data, 1, 2);
        let (min, max) = m.range();
        assert!((min).abs() < 1e-6);
        assert!((max - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_explicit_range_override() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let mut m = MatrixData::new(data, 2, 2);
        assert!((m.range().0 - 1.0).abs() < 1e-6);

        m.set_range(0.0, 10.0);
        let (min, max) = m.range();
        assert!((min).abs() < 1e-6);
        assert!((max - 10.0).abs() < 1e-6);
    }

    #[test]
    fn test_dimensions() {
        let data = vec![0.0; 12];
        let m = MatrixData::new(data, 3, 4);
        assert_eq!(m.rows(), 3);
        assert_eq!(m.cols(), 4);
    }

    #[test]
    fn test_single_element_matrix() {
        let data = vec![42.0];
        let m = MatrixData::new(data, 1, 1);
        assert_eq!(m.get_value(0, 0), Some(42.0));
        let (min, max) = m.range();
        assert!((min - 42.0).abs() < 1e-6);
        assert!((max - 42.0).abs() < 1e-6);
    }
}
