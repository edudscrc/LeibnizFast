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
/// Contains matrix dimensions, data range, and chunk offset for normalization.
/// Must match the WGSL struct layout exactly (24 bytes, padded to 16-byte alignment).
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct MatrixParams {
    /// Number of rows in this chunk (may be less than total matrix rows in staging mode)
    pub rows: u32,
    /// Number of columns in the matrix (always the full width)
    pub cols: u32,
    /// Minimum data value (maps to first colormap color)
    pub min_val: f32,
    /// Maximum data value (maps to last colormap color)
    pub max_val: f32,
    /// Row offset in the output texture (0 for single-buffer, >0 for staging chunks)
    pub row_offset: u32,
    /// Padding to align struct to 8 bytes per field (uniform buffer requirement)
    pub _pad: u32,
}

/// CPU-side matrix data for tooltip lookups and range computation.
///
/// Keeps a copy of the matrix data in memory to avoid async GPU readback
/// when the user hovers over cells.
pub struct MatrixData {
    /// Row-major flat data (empty when `range_only` is true)
    data: Vec<f32>,
    rows: u32,
    cols: u32,
    /// Data range for colormap normalization
    min_val: f32,
    max_val: f32,
    /// When true, only min/max are tracked — no data is stored.
    /// This allows huge matrices (>2 GB) to work within WASM's 4 GB limit.
    range_only: bool,
    /// Number of rows appended so far (used instead of `data.len()` when `range_only`).
    rows_appended: u32,
}

impl MatrixData {
    /// Create new matrix data from a flat row-major array.
    ///
    /// Automatically computes min/max from the data.
    pub fn new(data: Vec<f32>, rows: u32, cols: u32) -> Self {
        let (min_val, max_val) = Self::compute_range(&data);
        let rows_appended = rows;
        Self {
            data,
            rows,
            cols,
            min_val,
            max_val,
            range_only: false,
            rows_appended,
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

    /// Create a MatrixData with pre-allocated capacity for incremental building.
    ///
    /// Use `append_rows()` to add data, then `finalize()` when done.
    pub fn with_capacity(rows: u32, cols: u32) -> Self {
        let capacity = rows as usize * cols as usize;
        Self {
            data: Vec::with_capacity(capacity),
            rows,
            cols,
            min_val: f32::INFINITY,
            max_val: f32::NEG_INFINITY,
            range_only: false,
            rows_appended: 0,
        }
    }

    /// Create a MatrixData that tracks only min/max without storing data.
    ///
    /// Use this for matrices too large to fit in WASM memory (>2 GB).
    /// Tooltip hover will be unavailable, but GPU rendering still works.
    pub fn range_only(rows: u32, cols: u32) -> Self {
        Self {
            data: Vec::new(),
            rows,
            cols,
            min_val: f32::INFINITY,
            max_val: f32::NEG_INFINITY,
            range_only: true,
            rows_appended: 0,
        }
    }

    /// Returns true if this MatrixData is in range-only mode (no data stored).
    pub fn is_range_only(&self) -> bool {
        self.range_only
    }

    /// Append row data incrementally, updating the running min/max.
    ///
    /// `chunk` must contain a whole number of rows (length divisible by `cols`).
    pub fn append_rows(&mut self, chunk: &[f32]) {
        // Update running min/max
        for &v in chunk {
            if v.is_finite() {
                if v < self.min_val {
                    self.min_val = v;
                }
                if v > self.max_val {
                    self.max_val = v;
                }
            }
        }
        // Track rows appended (for rows_loaded)
        if self.cols > 0 {
            self.rows_appended += (chunk.len() / self.cols as usize) as u32;
        }
        // Only store data if not in range-only mode
        if !self.range_only {
            self.data.extend_from_slice(chunk);
        }
    }

    /// Finalize after all rows are appended. Handles all-NaN edge case.
    pub fn finalize(&mut self) {
        if self.min_val.is_infinite() {
            self.min_val = 0.0;
            self.max_val = 1.0;
        }
    }

    /// Get the number of complete rows loaded so far.
    pub fn rows_loaded(&self) -> u32 {
        if self.range_only {
            return self.rows_appended;
        }
        if self.cols == 0 {
            return 0;
        }
        (self.data.len() / self.cols as usize) as u32
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
/// Supports two modes:
/// - **Full buffer**: one storage buffer holds the entire matrix (standard path)
/// - **Staging buffer**: a smaller buffer that fits within `max_buffer_size`,
///   used for chunked compute shader processing of large matrices
///
/// Only available when compiling for WASM target.
#[cfg(target_arch = "wasm32")]
pub struct MatrixView {
    /// Storage buffer containing the raw float data (full or staging-sized)
    pub data_buffer: wgpu::Buffer,
    /// Uniform buffer containing matrix dimensions and range
    pub params_buffer: wgpu::Buffer,
    rows: u32,
    cols: u32,
    min_val: f32,
    max_val: f32,
    /// Whether this is a staging buffer (smaller than the full matrix)
    staging: bool,
    /// Number of rows that fit in the staging buffer (only meaningful when staging=true)
    staging_rows: u32,
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
            row_offset: 0,
            _pad: 0,
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
            staging: false,
            staging_rows: 0,
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

    /// Whether this view uses a staging buffer (smaller than the full matrix).
    pub fn is_staging(&self) -> bool {
        self.staging
    }

    /// Number of rows that fit in the staging buffer.
    /// Only meaningful when `is_staging()` returns true.
    pub fn staging_capacity_rows(&self) -> u32 {
        self.staging_rows
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
            row_offset: 0,
            _pad: 0,
        };
        queue.write_buffer(&self.params_buffer, 0, bytemuck::cast_slice(&[params]));
    }

    /// Create a MatrixView with an empty (uninitialized) GPU buffer.
    ///
    /// If the full matrix fits in `max_buffer_size`, creates a full-sized buffer.
    /// Otherwise, creates a **staging buffer** that holds as many rows as possible
    /// within the device limit. Data must then be processed in chunks using
    /// `write_staging_chunk()` + `update_chunk_params()`.
    pub fn with_empty_buffer(device: &wgpu::Device, rows: u32, cols: u32) -> Result<Self, String> {
        use wgpu::util::DeviceExt;

        let data_size = (rows as u64) * (cols as u64) * 4; // f32 = 4 bytes
        let max_size = device.limits().max_buffer_size;

        let (buffer_size, staging, staging_rows) = if data_size <= max_size {
            // Full matrix fits in one buffer
            (data_size, false, 0u32)
        } else {
            // Staging mode: compute how many rows fit, aligned to 16 (workgroup size)
            if cols == 0 {
                return Err("Cannot create staging buffer for 0-column matrix".to_string());
            }
            let row_bytes = cols as u64 * 4;
            let raw_rows = max_size / row_bytes;
            // Align down to 16 rows (compute shader workgroup alignment)
            let aligned_rows = ((raw_rows as u32) / 16) * 16;
            if aligned_rows == 0 {
                return Err(format!(
                    "A single row requires {} bytes but max_buffer_size is {}. \
                     Cannot create staging buffer.",
                    row_bytes, max_size
                ));
            }
            let staging_size = aligned_rows as u64 * row_bytes;
            log::info!(
                "Staging buffer: {} rows × {} cols = {} bytes (full matrix: {} bytes, max: {} bytes)",
                aligned_rows, cols, staging_size, data_size, max_size
            );
            (staging_size, true, aligned_rows)
        };

        let data_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(if staging {
                "Matrix Data Buffer (staging)"
            } else {
                "Matrix Data Buffer (chunked)"
            }),
            size: buffer_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let params = MatrixParams {
            rows,
            cols,
            min_val: 0.0,
            max_val: 1.0,
            row_offset: 0,
            _pad: 0,
        };
        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Matrix Params Buffer"),
            contents: bytemuck::cast_slice(&[params]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        Ok(Self {
            data_buffer,
            params_buffer,
            rows,
            cols,
            min_val: 0.0,
            max_val: 1.0,
            staging,
            staging_rows,
        })
    }

    /// Write a chunk of row data to the GPU buffer at the correct byte offset.
    ///
    /// For full-buffer mode: `row_offset` positions the data within the buffer.
    /// For staging mode: use `write_staging_chunk()` instead.
    pub fn write_chunk(&self, queue: &wgpu::Queue, row_offset: u32, cols: u32, chunk: &[f32]) {
        let byte_offset = (row_offset as u64) * (cols as u64) * 4;
        queue.write_buffer(&self.data_buffer, byte_offset, bytemuck::cast_slice(chunk));
    }

    /// Write chunk data to the staging buffer at offset 0 (reusing the buffer).
    ///
    /// In staging mode, the buffer is reused for each chunk — data always starts
    /// at byte 0. The `row_offset` is tracked in the uniform params instead.
    pub fn write_staging_chunk(&self, queue: &wgpu::Queue, chunk: &[f32]) {
        queue.write_buffer(&self.data_buffer, 0, bytemuck::cast_slice(chunk));
    }

    /// Update the params uniform for a specific chunk in staging mode.
    ///
    /// Sets `rows` to the chunk's row count and `row_offset` for correct
    /// texture positioning.
    pub fn update_chunk_params(
        &self,
        queue: &wgpu::Queue,
        chunk_rows: u32,
        row_offset: u32,
        min_val: f32,
        max_val: f32,
    ) {
        let params = MatrixParams {
            rows: chunk_rows,
            cols: self.cols,
            min_val,
            max_val,
            row_offset,
            _pad: 0,
        };
        queue.write_buffer(&self.params_buffer, 0, bytemuck::cast_slice(&[params]));
    }

    /// Update the min/max params uniform after all chunks are uploaded.
    pub fn update_params(&mut self, min: f32, max: f32, queue: &wgpu::Queue) {
        self.set_range(min, max, queue);
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

    #[test]
    fn test_incremental_matches_full() {
        let data: Vec<f32> = (0..120).map(|i| i as f32 * 0.1).collect();
        let full = MatrixData::new(data.clone(), 10, 12);

        let mut inc = MatrixData::with_capacity(10, 12);
        // Append in 3 chunks: 4 rows, 3 rows, 3 rows
        inc.append_rows(&data[..48]); // 4 * 12
        inc.append_rows(&data[48..84]); // 3 * 12
        inc.append_rows(&data[84..]); // 3 * 12
        inc.finalize();

        assert_eq!(full.rows(), inc.rows());
        assert_eq!(full.cols(), inc.cols());
        assert_eq!(full.raw_data(), inc.raw_data());
        let (fmin, fmax) = full.range();
        let (imin, imax) = inc.range();
        assert!((fmin - imin).abs() < 1e-6);
        assert!((fmax - imax).abs() < 1e-6);
    }

    #[test]
    fn test_append_rows_updates_range() {
        let mut m = MatrixData::with_capacity(3, 2);
        m.append_rows(&[1.0, 2.0]); // row 0: min=1, max=2
        assert!((m.range().0 - 1.0).abs() < 1e-6);
        assert!((m.range().1 - 2.0).abs() < 1e-6);

        m.append_rows(&[0.5, 3.0]); // row 1: min should drop to 0.5, max to 3.0
        assert!((m.range().0 - 0.5).abs() < 1e-6);
        assert!((m.range().1 - 3.0).abs() < 1e-6);

        m.append_rows(&[-1.0, 10.0]); // row 2: min=-1, max=10
        assert!((m.range().0 - (-1.0)).abs() < 1e-6);
        assert!((m.range().1 - 10.0).abs() < 1e-6);
        m.finalize();
    }

    #[test]
    fn test_rows_loaded_tracking() {
        let mut m = MatrixData::with_capacity(4, 3);
        assert_eq!(m.rows_loaded(), 0);

        m.append_rows(&[1.0, 2.0, 3.0]); // 1 row
        assert_eq!(m.rows_loaded(), 1);

        m.append_rows(&[4.0, 5.0, 6.0, 7.0, 8.0, 9.0]); // 2 rows
        assert_eq!(m.rows_loaded(), 3);

        m.append_rows(&[10.0, 11.0, 12.0]); // 1 row
        assert_eq!(m.rows_loaded(), 4);
        m.finalize();
    }

    #[test]
    fn test_finalize_all_nan() {
        let mut m = MatrixData::with_capacity(2, 2);
        m.append_rows(&[f32::NAN, f32::NAN, f32::NAN, f32::NAN]);
        m.finalize();
        let (min, max) = m.range();
        assert!((min - 0.0).abs() < 1e-6);
        assert!((max - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_range_only_no_data_storage() {
        let mut m = MatrixData::range_only(100, 100);
        assert!(m.is_range_only());

        // Append some data — should track range but NOT store elements
        let chunk: Vec<f32> = (0..1000).map(|i| i as f32 * 0.01).collect();
        m.append_rows(&chunk); // 10 rows of 100

        assert!((m.range().0 - 0.0).abs() < 1e-6);
        assert!((m.range().1 - 9.99).abs() < 1e-6);

        // Data should NOT be stored
        assert!(m.raw_data().is_empty());
        assert_eq!(m.get_value(0, 0), None);
        assert_eq!(m.get_value(5, 50), None);
    }

    #[test]
    fn test_range_only_finalize() {
        let mut m = MatrixData::range_only(2, 2);
        m.append_rows(&[1.0, 2.0, 3.0, 4.0]);
        m.finalize();
        let (min, max) = m.range();
        assert!((min - 1.0).abs() < 1e-6);
        assert!((max - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_range_only_rows_loaded() {
        let mut m = MatrixData::range_only(4, 3);
        assert_eq!(m.rows_loaded(), 0);

        m.append_rows(&[1.0, 2.0, 3.0]); // 1 row
        assert_eq!(m.rows_loaded(), 1);

        m.append_rows(&[4.0, 5.0, 6.0, 7.0, 8.0, 9.0]); // 2 rows
        assert_eq!(m.rows_loaded(), 3);

        m.append_rows(&[10.0, 11.0, 12.0]); // 1 row
        assert_eq!(m.rows_loaded(), 4);
        m.finalize();
    }
}
