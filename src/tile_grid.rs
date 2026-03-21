//! # Tile Grid
//!
//! Splits a large matrix into a 2-D grid of tiles, each fitting within the
//! GPU's `maxTextureDimension2D` limit.
//!
//! This is a pure-logic module with no GPU dependencies — it can be tested on
//! native targets and is used by both the compute and render pipelines.
//!
//! ## Motivation
//! Chrome's WebGPU implementation caps `maxTextureDimension2D` at 8192 (the
//! WebGPU spec minimum), even on hardware that supports 16384.  For a matrix
//! with 16 000 rows or columns we therefore need more than one texture.
//!
//! ## Layout
//! Tiles are indexed (tx, ty) where tx is the column-tile index and ty is the
//! row-tile index:
//!
//! ```text
//! (0,0) | (1,0) | (2,0)
//! (0,1) | (1,1) | (2,1)
//! ```
//!
//! The last tile in each dimension is a *remainder* tile whose size is
//! `dim % max_dim` (or `max_dim` when the dimension divides exactly).

/// Grid of fixed-size texture tiles that together cover the full matrix.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TileGrid {
    /// Total matrix rows.
    pub total_rows: u32,
    /// Total matrix columns.
    pub total_cols: u32,
    /// Maximum tile side length (= `maxTextureDimension2D` of the device).
    pub max_dim: u32,
    /// Number of column-tiles (horizontal count).
    pub tiles_x: u32,
    /// Number of row-tiles (vertical count).
    pub tiles_y: u32,
}

impl TileGrid {
    /// Build a TileGrid for the given matrix dimensions and device limit.
    ///
    /// # Panics
    /// Panics if `max_dim` is 0.
    pub fn new(total_rows: u32, total_cols: u32, max_dim: u32) -> Self {
        assert!(max_dim > 0, "max_dim must be > 0");
        let tiles_x = total_cols.div_ceil(max_dim);
        let tiles_y = total_rows.div_ceil(max_dim);
        Self {
            total_rows,
            total_cols,
            max_dim,
            tiles_x,
            tiles_y,
        }
    }

    /// Total number of tiles in the grid.
    pub fn tile_count(&self) -> usize {
        (self.tiles_x * self.tiles_y) as usize
    }

    /// Returns `true` when the matrix fits in a single tile (common fast path).
    pub fn needs_tiling(&self) -> bool {
        self.tiles_x > 1 || self.tiles_y > 1
    }

    /// Width of tile at column-index `tx`.
    pub fn tile_width(&self, tx: u32) -> u32 {
        debug_assert!(tx < self.tiles_x, "tx out of range");
        let last = self.tiles_x - 1;
        if tx < last || self.total_cols.is_multiple_of(self.max_dim) {
            self.max_dim
        } else {
            self.total_cols % self.max_dim
        }
    }

    /// Height of tile at row-index `ty`.
    pub fn tile_height(&self, ty: u32) -> u32 {
        debug_assert!(ty < self.tiles_y, "ty out of range");
        let last = self.tiles_y - 1;
        if ty < last || self.total_rows.is_multiple_of(self.max_dim) {
            self.max_dim
        } else {
            self.total_rows % self.max_dim
        }
    }

    /// (col_start, col_end) in absolute matrix coordinates for tile column `tx`.
    pub fn tile_col_range(&self, tx: u32) -> (u32, u32) {
        debug_assert!(tx < self.tiles_x, "tx out of range");
        let start = tx * self.max_dim;
        let end = (start + self.tile_width(tx)).min(self.total_cols);
        (start, end)
    }

    /// (row_start, row_end) in absolute matrix coordinates for tile row `ty`.
    pub fn tile_row_range(&self, ty: u32) -> (u32, u32) {
        debug_assert!(ty < self.tiles_y, "ty out of range");
        let start = ty * self.max_dim;
        let end = (start + self.tile_height(ty)).min(self.total_rows);
        (start, end)
    }

    /// Iterate over all (tx, ty) pairs in row-major order (ty outer, tx inner).
    pub fn iter_tiles(&self) -> impl Iterator<Item = (u32, u32)> + '_ {
        (0..self.tiles_y).flat_map(move |ty| (0..self.tiles_x).map(move |tx| (tx, ty)))
    }

    /// Flat index for tile (tx, ty): `ty * tiles_x + tx`.
    pub fn tile_index(&self, tx: u32, ty: u32) -> usize {
        (ty * self.tiles_x + tx) as usize
    }

    /// Fraction [0.0, 1.0] of the full matrix width covered by column-tiles 0..tx_end.
    /// Used to compute per-tile UV offsets for the render shader.
    pub fn col_uv_offset(&self, tx: u32) -> f32 {
        (tx * self.max_dim) as f32 / self.total_cols as f32
    }

    /// Fraction [0.0, 1.0] of the full matrix height covered by row-tiles 0..ty_end.
    pub fn row_uv_offset(&self, ty: u32) -> f32 {
        (ty * self.max_dim) as f32 / self.total_rows as f32
    }

    /// UV width of tile `tx` in the full matrix [0.0, 1.0].
    pub fn col_uv_size(&self, tx: u32) -> f32 {
        self.tile_width(tx) as f32 / self.total_cols as f32
    }

    /// UV height of tile `ty` in the full matrix [0.0, 1.0].
    pub fn row_uv_size(&self, ty: u32) -> f32 {
        self.tile_height(ty) as f32 / self.total_rows as f32
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: check that tile_col_range / tile_row_range partition the whole matrix.
    fn check_partitions(grid: &TileGrid) {
        let mut col_end = 0u32;
        for tx in 0..grid.tiles_x {
            let (start, end) = grid.tile_col_range(tx);
            assert_eq!(start, col_end, "col gap at tx={tx}");
            col_end = end;
        }
        assert_eq!(col_end, grid.total_cols, "col coverage mismatch");

        let mut row_end = 0u32;
        for ty in 0..grid.tiles_y {
            let (start, end) = grid.tile_row_range(ty);
            assert_eq!(start, row_end, "row gap at ty={ty}");
            row_end = end;
        }
        assert_eq!(row_end, grid.total_rows, "row coverage mismatch");
    }

    #[test]
    fn test_no_tiling_small_matrix() {
        let g = TileGrid::new(100, 100, 8192);
        assert!(!g.needs_tiling());
        assert_eq!(g.tile_count(), 1);
        // Single tile is a remainder tile: 100 % 8192 = 100
        assert_eq!(g.tile_width(0), 100);
        assert_eq!(g.tile_height(0), 100);
        let (cs, ce) = g.tile_col_range(0);
        assert_eq!(cs, 0);
        assert_eq!(ce, 100);
        check_partitions(&g);
    }

    #[test]
    fn test_exact_multiple_no_tiling() {
        let g = TileGrid::new(8192, 8192, 8192);
        assert!(!g.needs_tiling());
        assert_eq!(g.tiles_x, 1);
        assert_eq!(g.tiles_y, 1);
        assert_eq!(g.tile_width(0), 8192);
        assert_eq!(g.tile_height(0), 8192);
        check_partitions(&g);
    }

    #[test]
    fn test_tiling_one_axis_cols() {
        let g = TileGrid::new(4000, 16000, 8192);
        assert!(g.needs_tiling());
        assert_eq!(g.tiles_x, 2);
        assert_eq!(g.tiles_y, 1);
        assert_eq!(g.tile_count(), 2);
        assert_eq!(g.tile_width(0), 8192);
        assert_eq!(g.tile_width(1), 16000 - 8192); // 7808
        assert_eq!(g.tile_height(0), 4000); // col range: tile_height for ty=0
        check_partitions(&g);
    }

    #[test]
    fn test_tiling_both_axes_16k() {
        let g = TileGrid::new(16000, 16000, 8192);
        assert!(g.needs_tiling());
        assert_eq!(g.tiles_x, 2);
        assert_eq!(g.tiles_y, 2);
        assert_eq!(g.tile_count(), 4);
        // First tile: max_dim × max_dim
        assert_eq!(g.tile_width(0), 8192);
        assert_eq!(g.tile_height(0), 8192);
        // Remainder tiles
        let rem = 16000 - 8192; // 7808
        assert_eq!(g.tile_width(1), rem);
        assert_eq!(g.tile_height(1), rem);
        check_partitions(&g);
    }

    #[test]
    fn test_tiling_32k() {
        let g = TileGrid::new(32000, 32000, 8192);
        assert_eq!(g.tiles_x, 4);
        assert_eq!(g.tiles_y, 4);
        assert_eq!(g.tile_count(), 16);
        // 3 full tiles + 1 remainder
        for tx in 0..3 {
            assert_eq!(g.tile_width(tx), 8192);
        }
        assert_eq!(g.tile_width(3), 32000 - 3 * 8192); // 7424
        check_partitions(&g);
    }

    #[test]
    fn test_exact_multiple_both_axes() {
        // 16384 = 2 × 8192 — no remainder tiles
        let g = TileGrid::new(16384, 16384, 8192);
        assert_eq!(g.tiles_x, 2);
        assert_eq!(g.tiles_y, 2);
        for tx in 0..2 {
            assert_eq!(g.tile_width(tx), 8192, "tx={tx}");
            assert_eq!(g.tile_height(tx), 8192, "ty={tx}");
        }
        check_partitions(&g);
    }

    #[test]
    fn test_tile_index_row_major() {
        let g = TileGrid::new(16000, 24000, 8192);
        // tiles_x = 3, tiles_y = 2
        assert_eq!(g.tile_index(0, 0), 0);
        assert_eq!(g.tile_index(1, 0), 1);
        assert_eq!(g.tile_index(2, 0), 2);
        assert_eq!(g.tile_index(0, 1), 3);
        assert_eq!(g.tile_index(2, 1), 5);
    }

    #[test]
    fn test_uv_offsets() {
        let g = TileGrid::new(16000, 16000, 8192);
        // col_uv_offset(0) = 0, col_uv_offset(1) = 8192/16000
        assert!((g.col_uv_offset(0) - 0.0).abs() < 1e-6);
        let expected = 8192.0 / 16000.0;
        assert!((g.col_uv_offset(1) - expected).abs() < 1e-6);

        // col_uv_size(0) + col_uv_size(1) should sum to 1.0
        let sum = g.col_uv_size(0) + g.col_uv_size(1);
        assert!((sum - 1.0).abs() < 1e-5, "UV sizes don't sum to 1: {sum}");
    }

    #[test]
    fn test_iter_tiles_count() {
        let g = TileGrid::new(16000, 24000, 8192);
        let count = g.iter_tiles().count();
        assert_eq!(count, g.tile_count());
    }

    #[test]
    fn test_single_row_matrix() {
        let g = TileGrid::new(1, 10000, 8192);
        assert!(g.needs_tiling());
        assert_eq!(g.tiles_x, 2);
        assert_eq!(g.tiles_y, 1);
        assert_eq!(g.tile_height(0), 1);
        check_partitions(&g);
    }

    #[test]
    fn test_single_column_matrix() {
        let g = TileGrid::new(10000, 1, 8192);
        assert!(g.needs_tiling());
        assert_eq!(g.tiles_x, 1);
        assert_eq!(g.tiles_y, 2);
        assert_eq!(g.tile_width(0), 1);
        check_partitions(&g);
    }

    #[test]
    #[should_panic(expected = "max_dim must be > 0")]
    fn test_max_dim_zero_panics() {
        TileGrid::new(100, 100, 0);
    }

    #[test]
    fn test_uv_sizes_sum_to_one() {
        let g = TileGrid::new(16000, 24000, 8192);
        let col_sum: f32 = (0..g.tiles_x).map(|tx| g.col_uv_size(tx)).sum();
        let row_sum: f32 = (0..g.tiles_y).map(|ty| g.row_uv_size(ty)).sum();
        assert!((col_sum - 1.0).abs() < 1e-5, "col UV sum = {col_sum}");
        assert!((row_sum - 1.0).abs() < 1e-5, "row UV sum = {row_sum}");
    }

    #[test]
    fn test_one_by_one_matrix() {
        let g = TileGrid::new(1, 1, 8192);
        assert!(!g.needs_tiling());
        assert_eq!(g.tile_count(), 1);
        assert_eq!(g.tile_width(0), 1);
        assert_eq!(g.tile_height(0), 1);
        check_partitions(&g);
    }
}
