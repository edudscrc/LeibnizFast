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
/// Must match the WGSL struct layout exactly (32 bytes, 8 × u32/f32).
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct MatrixParams {
    /// Number of rows in this chunk (may be less than total matrix rows in staging mode)
    pub rows: u32,
    /// Number of columns in *this tile* (may be less than total_cols in tiling mode)
    pub cols: u32,
    /// Minimum data value (maps to first colormap color)
    pub min_val: f32,
    /// Maximum data value (maps to last colormap color)
    pub max_val: f32,
    /// Row offset in the *full matrix* for this chunk/tile (0 for first chunk)
    pub row_offset: u32,
    /// Column offset in the *full matrix* for this tile (0 for single-tile / non-tiled)
    pub col_offset: u32,
    /// Total number of columns in the full matrix (used for buffer indexing)
    pub total_cols: u32,
    /// Y offset when writing to tile texture (for multi-chunk-per-tile staging)
    pub texture_row_offset: u32,
}

/// Page size for paged storage: 64 MB / 4 bytes per f32 = 16M elements.
const PAGE_SIZE_ELEMENTS: usize = 16 * 1024 * 1024;

/// Paged storage for large matrices.
///
/// Stores f32 data in fixed-size pages (64 MB each) so that no single
/// allocation exceeds 64 MB. Supports appending and random access by index.
pub struct PagedStorage {
    pages: Vec<Vec<f32>>,
    total_len: usize,
}

impl Default for PagedStorage {
    fn default() -> Self {
        Self::new()
    }
}

impl PagedStorage {
    /// Create an empty paged storage.
    pub fn new() -> Self {
        Self {
            pages: Vec::new(),
            total_len: 0,
        }
    }

    /// Append data, splitting across pages as needed.
    pub fn append(&mut self, data: &[f32]) {
        let mut offset = 0;
        while offset < data.len() {
            // Get or create current page
            if self.pages.is_empty() || self.pages.last().unwrap().len() == PAGE_SIZE_ELEMENTS {
                self.pages.push(Vec::with_capacity(
                    PAGE_SIZE_ELEMENTS.min(data.len() - offset + self.current_page_len()),
                ));
            }
            let page = self.pages.last_mut().unwrap();
            let space = PAGE_SIZE_ELEMENTS - page.len();
            let to_copy = space.min(data.len() - offset);
            page.extend_from_slice(&data[offset..offset + to_copy]);
            offset += to_copy;
        }
        self.total_len += data.len();
    }

    /// Get a value by flat index.
    pub fn get(&self, index: usize) -> Option<f32> {
        if index >= self.total_len {
            return None;
        }
        let page_idx = index / PAGE_SIZE_ELEMENTS;
        let offset = index % PAGE_SIZE_ELEMENTS;
        self.pages
            .get(page_idx)
            .and_then(|p| p.get(offset))
            .copied()
    }

    /// Total number of elements stored.
    pub fn len(&self) -> usize {
        self.total_len
    }

    /// Whether the storage is empty.
    pub fn is_empty(&self) -> bool {
        self.total_len == 0
    }

    /// Number of pages allocated.
    pub fn page_count(&self) -> usize {
        self.pages.len()
    }

    /// Get a reference to a specific page.
    pub fn get_page(&self, page_idx: usize) -> Option<&[f32]> {
        self.pages.get(page_idx).map(|p| p.as_slice())
    }

    /// Read a contiguous range of elements, potentially spanning pages.
    /// Returns the data in the provided buffer. Panics if out of bounds.
    pub fn read_range(&self, start: usize, buf: &mut [f32]) {
        let mut offset = 0;
        let mut global_idx = start;
        while offset < buf.len() {
            let page_idx = global_idx / PAGE_SIZE_ELEMENTS;
            let page_offset = global_idx % PAGE_SIZE_ELEMENTS;
            let page = &self.pages[page_idx];
            let available = page.len() - page_offset;
            let to_copy = available.min(buf.len() - offset);
            buf[offset..offset + to_copy]
                .copy_from_slice(&page[page_offset..page_offset + to_copy]);
            offset += to_copy;
            global_idx += to_copy;
        }
    }

    fn current_page_len(&self) -> usize {
        self.pages.last().map_or(0, |p| p.len())
    }
}

/// CPU-side matrix data for tooltip lookups and range computation.
///
/// Stores data in 64 MB pages via `PagedStorage` so no single allocation
/// is too large. Used for native (non-WASM) tests. In WASM production,
/// `JsDataSource` is used instead to keep data in the JS heap.
pub struct MatrixData {
    /// Row-major paged data
    data: PagedStorage,
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
        let mut paged = PagedStorage::new();
        paged.append(&data);
        Self {
            data: paged,
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
            self.data.get(idx)
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

    /// Get the paged storage (for iteration-based access).
    pub fn paged_data(&self) -> &PagedStorage {
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

    /// Create a MatrixData for incremental building via `append_rows()`.
    pub fn with_capacity(rows: u32, cols: u32) -> Self {
        Self {
            data: PagedStorage::new(),
            rows,
            cols,
            min_val: f32::INFINITY,
            max_val: f32::NEG_INFINITY,
        }
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
        self.data.append(chunk);
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

/// Maximum staging buffer size: 256 MB.
///
/// This caps the largest single GPU allocation. A larger staging buffer means
/// fewer compute dispatches per tile (better performance), but must stay within
/// the device's `max_buffer_size` (typically 256 MB–1 GB on WebGPU).
#[cfg(target_arch = "wasm32")]
const MAX_STAGING_BYTES: u64 = 256 * 1024 * 1024;

/// GPU-side matrix view with a staging buffer and parameters uniform.
///
/// Always uses a **staging buffer** (≤ 64 MB) that is reused per chunk. Data is
/// written to the staging buffer at offset 0, then the compute shader processes
/// it into tile textures. This eliminates the need for a full-matrix-sized GPU
/// buffer and keeps GPU memory usage constant regardless of matrix size.
///
/// Only available when compiling for WASM target.
#[cfg(target_arch = "wasm32")]
pub struct MatrixView {
    /// Staging buffer for compute shader input (reused per chunk)
    pub data_buffer: wgpu::Buffer,
    /// Uniform buffer containing matrix dimensions and range
    pub params_buffer: wgpu::Buffer,
    rows: u32,
    cols: u32,
    /// Number of rows that fit in the staging buffer
    staging_rows: u32,
}

#[cfg(target_arch = "wasm32")]
impl MatrixView {
    /// Get the number of rows.
    pub fn rows(&self) -> u32 {
        self.rows
    }

    /// Get the number of columns.
    pub fn cols(&self) -> u32 {
        self.cols
    }

    /// Number of rows that fit in the staging buffer.
    pub fn staging_capacity_rows(&self) -> u32 {
        self.staging_rows
    }

    /// Create a MatrixView with a staging buffer (≤ 64 MB).
    ///
    /// The staging buffer holds as many complete rows as fit within
    /// `min(MAX_STAGING_BYTES, max_buffer_size)`, aligned to 16 rows.
    /// Data is always processed in chunks through this buffer.
    pub fn with_empty_buffer(
        device: &wgpu::Device,
        rows: u32,
        cols: u32,
        debug: bool,
    ) -> Result<Self, String> {
        use crate::perf::PerfTimer;
        use wgpu::util::DeviceExt;
        let _timer = PerfTimer::new("MatrixView::with_empty_buffer", debug);

        if cols == 0 {
            return Err("Cannot create staging buffer for 0-column matrix".to_string());
        }

        let data_size = (rows as u64) * (cols as u64) * 4;
        let max_size = device.limits().max_buffer_size;
        let budget = MAX_STAGING_BYTES.min(max_size);

        let row_bytes = cols as u64 * 4;
        let raw_rows = (budget / row_bytes).min(rows as u64);
        // Align down to 16 rows (compute shader workgroup alignment), but allow
        // fewer if the total matrix is < 16 rows
        let aligned_rows = if rows <= 16 {
            raw_rows as u32
        } else {
            ((raw_rows as u32) / 16) * 16
        };
        if aligned_rows == 0 {
            return Err(format!(
                "A single row requires {} bytes but staging budget is {} bytes. \
                 Cannot create staging buffer.",
                row_bytes, budget
            ));
        }
        let buffer_size = aligned_rows as u64 * row_bytes;
        log::info!(
            "Staging buffer: {} rows × {} cols = {} bytes (full matrix: {} bytes, budget: {} bytes)",
            aligned_rows, cols, buffer_size, data_size, budget
        );

        let data_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Matrix Staging Buffer"),
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
            col_offset: 0,
            total_cols: cols,
            texture_row_offset: 0,
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
            staging_rows: aligned_rows,
        })
    }

    /// Write chunk data to the staging buffer at offset 0 (reusing the buffer).
    ///
    /// The staging buffer is reused for each chunk — data always starts at byte 0.
    pub fn write_staging_chunk(&self, queue: &wgpu::Queue, chunk: &[f32]) {
        queue.write_buffer(&self.data_buffer, 0, bytemuck::cast_slice(chunk));
    }
}

/// Chunk size for scanning min/max from JS Float32Array: 4M elements = 16 MB.
#[cfg(target_arch = "wasm32")]
const SCAN_CHUNK_ELEMENTS: usize = 4 * 1024 * 1024;

/// JS-heap-backed data source for matrix data.
///
/// Keeps matrix data as a `js_sys::Float32Array` in the JavaScript heap,
/// avoiding WASM's 4 GB address space limit. Tooltips and colormap changes
/// work at any matrix size since only metadata lives in WASM memory.
#[cfg(target_arch = "wasm32")]
pub struct JsDataSource {
    /// Reference to the Float32Array in JS heap
    data: js_sys::Float32Array,
    rows: u32,
    cols: u32,
    min_val: f32,
    max_val: f32,
}

#[cfg(target_arch = "wasm32")]
impl JsDataSource {
    /// Create a JsDataSource with a pre-supplied range, skipping the min/max scan.
    ///
    /// Use this when the caller already knows the desired range (e.g. manual range
    /// mode in real-time streaming). The Float32Array stays in JS heap — no copy.
    pub fn new_with_range(
        data: js_sys::Float32Array,
        rows: u32,
        cols: u32,
        min_val: f32,
        max_val: f32,
    ) -> Self {
        Self {
            data,
            rows,
            cols,
            min_val,
            max_val,
        }
    }

    /// Create a JsDataSource by scanning the Float32Array for min/max.
    ///
    /// The scan reads in small chunks (16 MB) to avoid large WASM temporaries.
    /// The Float32Array stays in JS heap — no copy into WASM memory.
    pub fn new(data: js_sys::Float32Array, rows: u32, cols: u32, debug: bool) -> Self {
        use crate::perf::PerfTimer;
        let _timer = PerfTimer::new("JsDataSource::new", debug);
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
                        if v < min_val {
                            min_val = v;
                        }
                        if v > max_val {
                            max_val = v;
                        }
                    }
                }
                offset = end;
            }
        }
        // Handle all-NaN or empty data
        if min_val.is_infinite() {
            min_val = 0.0;
            max_val = 1.0;
        }
        Self {
            data,
            rows,
            cols,
            min_val,
            max_val,
        }
    }

    /// Create a JsDataSource with an empty JS Float32Array for streaming accumulation.
    ///
    /// Allocates `rows × cols` elements in JS heap. Use `write_range()` to fill
    /// data during streaming, and `update_min_max()` to track the running range.
    pub fn from_empty(rows: u32, cols: u32) -> Self {
        let total = (rows as u32) * (cols as u32);
        let data = js_sys::Float32Array::new_with_length(total);
        Self {
            data,
            rows,
            cols,
            min_val: f32::INFINITY,
            max_val: f32::NEG_INFINITY,
        }
    }

    /// Get the value at (row, col). Returns `None` if out of bounds.
    ///
    /// Reads a single f32 from the JS Float32Array via `get_index()`.
    /// This crosses the JS/WASM boundary once — negligible for hover events.
    pub fn get_value(&self, row: u32, col: u32) -> Option<f32> {
        if row < self.rows && col < self.cols {
            let idx = row * self.cols + col;
            Some(self.data.get_index(idx))
        } else {
            None
        }
    }

    /// Read a contiguous range of elements into a buffer.
    ///
    /// Uses `subarray().copy_to()` for efficient bulk transfer from JS heap
    /// to WASM memory. Used by colormap re-dispatch to fill staging buffers.
    pub fn read_range(&self, start: usize, buf: &mut [f32]) {
        let end = start + buf.len();
        self.data.subarray(start as u32, end as u32).copy_to(buf);
    }

    /// Write data into the JS Float32Array at a given element offset.
    ///
    /// Used during streaming (`append_chunk`) to accumulate data in JS heap.
    /// Creates a temporary JS view of the WASM slice — safe because the view
    /// is consumed immediately by `set()` before any WASM memory growth.
    pub fn write_range(&self, offset: u32, chunk: &[f32]) {
        // SAFETY: Float32Array::view creates a JS typed array backed by WASM memory.
        // This is safe as long as we don't grow WASM memory between creating
        // the view and consuming it (the `set()` call copies the data immediately).
        let src_view = unsafe { js_sys::Float32Array::view(chunk) };
        self.data.set(&src_view, offset);
    }

    /// Update running min/max from a chunk of data (already in WASM memory).
    ///
    /// Called during streaming to track the data range incrementally.
    pub fn update_min_max(&mut self, chunk: &[f32]) {
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
    }

    /// Finalize after all data has been written. Handles all-NaN edge case.
    pub fn finalize(&mut self) {
        if self.min_val.is_infinite() {
            self.min_val = 0.0;
            self.max_val = 1.0;
        }
    }

    /// Override the auto-computed range with explicit values.
    pub fn set_range(&mut self, min: f32, max: f32) {
        self.min_val = min;
        self.max_val = max;
    }

    /// Get the number of rows.
    pub fn rows(&self) -> u32 {
        self.rows
    }

    /// Get the number of columns.
    pub fn cols(&self) -> u32 {
        self.cols
    }

    /// Get the current data range (min, max).
    pub fn range(&self) -> (f32, f32) {
        (self.min_val, self.max_val)
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
        // Verify element-by-element via paged storage
        for i in 0..data.len() {
            assert_eq!(full.paged_data().get(i), inc.paged_data().get(i));
        }
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

    // --- PagedStorage tests ---

    #[test]
    fn test_paged_storage_basic() {
        let mut ps = PagedStorage::new();
        assert!(ps.is_empty());
        assert_eq!(ps.len(), 0);

        ps.append(&[1.0, 2.0, 3.0]);
        assert_eq!(ps.len(), 3);
        assert_eq!(ps.get(0), Some(1.0));
        assert_eq!(ps.get(1), Some(2.0));
        assert_eq!(ps.get(2), Some(3.0));
        assert_eq!(ps.get(3), None);
    }

    #[test]
    fn test_paged_storage_cross_page_boundary() {
        let mut ps = PagedStorage::new();
        // Fill almost one full page using vec![0.0; N] (fast even in debug)
        let target = PAGE_SIZE_ELEMENTS - 2;
        let mut filler = vec![0.0f32; target];
        // Mark a few sentinel values near the boundary
        filler[target - 1] = 42.0;
        ps.append(&filler);
        assert_eq!(ps.page_count(), 1);

        // Cross the page boundary
        ps.append(&[99.0, 100.0, 101.0, 102.0]);
        assert_eq!(ps.page_count(), 2);
        assert_eq!(ps.len(), PAGE_SIZE_ELEMENTS + 2);

        // Check values around the boundary
        assert_eq!(ps.get(PAGE_SIZE_ELEMENTS - 3), Some(42.0));
        assert_eq!(ps.get(PAGE_SIZE_ELEMENTS - 2), Some(99.0));
        assert_eq!(ps.get(PAGE_SIZE_ELEMENTS - 1), Some(100.0));
        assert_eq!(ps.get(PAGE_SIZE_ELEMENTS), Some(101.0));
        assert_eq!(ps.get(PAGE_SIZE_ELEMENTS + 1), Some(102.0));
    }

    #[test]
    fn test_paged_storage_read_range() {
        let mut ps = PagedStorage::new();
        let data: Vec<f32> = (0..100).map(|i| i as f32).collect();
        ps.append(&data);

        let mut buf = [0.0f32; 10];
        ps.read_range(5, &mut buf);
        for i in 0..10 {
            assert_eq!(buf[i], (i + 5) as f32);
        }
    }

    #[test]
    fn test_paged_storage_read_range_cross_page() {
        let mut ps = PagedStorage::new();
        // Fill one page using vec![0.0; N] (fast in debug), then set sentinel values
        let mut page1 = vec![0.0f32; PAGE_SIZE_ELEMENTS];
        // Set known values near the end of page 1
        for i in 0..5 {
            page1[PAGE_SIZE_ELEMENTS - 5 + i] = (PAGE_SIZE_ELEMENTS - 5 + i) as f32;
        }
        ps.append(&page1);
        // Add 10 more elements on page 2 with known values
        let tail: Vec<f32> = (PAGE_SIZE_ELEMENTS..PAGE_SIZE_ELEMENTS + 10)
            .map(|i| i as f32)
            .collect();
        ps.append(&tail);

        // Read across page boundary
        let mut buf = [0.0f32; 15];
        ps.read_range(PAGE_SIZE_ELEMENTS - 5, &mut buf);
        for i in 0..15 {
            assert_eq!(buf[i], (PAGE_SIZE_ELEMENTS - 5 + i) as f32);
        }
    }
}
