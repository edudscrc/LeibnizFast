//! # Chunked Upload
//!
//! Pure-logic module that computes chunk boundaries for splitting large matrix
//! uploads into GPU-friendly pieces.

/// Configuration for chunked uploads.
pub struct ChunkConfig {
    /// Number of rows per chunk. `None` means auto-compute based on ~16MB chunks.
    pub chunk_rows: Option<u32>,
}

/// Computes chunk boundaries for row-aligned matrix uploads.
///
/// Chunks are aligned to 16 rows (matching the compute shader workgroup size)
/// and default to ~16MB per chunk.
pub struct ChunkedUploader {
    rows: u32,
    #[allow(dead_code)]
    cols: u32,
    chunk_rows: u32,
    current_row: u32,
}

/// Target chunk size in bytes (~16MB).
const TARGET_CHUNK_BYTES: usize = 16 * 1024 * 1024;

/// Workgroup alignment (must match compute shader workgroup_size.y).
pub(crate) const WORKGROUP_ALIGNMENT: u32 = 16;

impl ChunkedUploader {
    /// Create a new ChunkedUploader.
    ///
    /// If `config.chunk_rows` is `None`, auto-computes chunk size targeting
    /// ~16MB per chunk, aligned to 16 rows.
    pub fn new(rows: u32, cols: u32, config: ChunkConfig) -> Self {
        let chunk_rows = match config.chunk_rows {
            Some(cr) => cr.min(rows),
            None => {
                if cols == 0 {
                    return Self {
                        rows,
                        cols,
                        chunk_rows: rows,
                        current_row: 0,
                    };
                }
                let row_bytes = cols as usize * 4; // f32 = 4 bytes
                let raw_rows = TARGET_CHUNK_BYTES / row_bytes;
                // Align down to WORKGROUP_ALIGNMENT, minimum WORKGROUP_ALIGNMENT
                let aligned = (raw_rows as u32 / WORKGROUP_ALIGNMENT) * WORKGROUP_ALIGNMENT;
                aligned.max(WORKGROUP_ALIGNMENT).min(rows)
            }
        };

        Self {
            rows,
            cols,
            chunk_rows,
            current_row: 0,
        }
    }

    /// Get the next chunk range as `(start_row, end_row)` (exclusive end).
    ///
    /// Returns `None` if all rows have been covered.
    pub fn next_chunk_range(&self) -> Option<(u32, u32)> {
        if self.current_row >= self.rows {
            return None;
        }
        let end = (self.current_row + self.chunk_rows).min(self.rows);
        Some((self.current_row, end))
    }

    /// Advance past the current chunk.
    pub fn advance(&mut self) {
        if self.current_row < self.rows {
            self.current_row = (self.current_row + self.chunk_rows).min(self.rows);
        }
    }

    /// Check if all rows have been processed.
    pub fn is_complete(&self) -> bool {
        self.current_row >= self.rows
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chunk_ranges_cover_all_rows() {
        let mut uploader = ChunkedUploader::new(
            1000,
            1000,
            ChunkConfig {
                chunk_rows: Some(256),
            },
        );

        let mut covered_rows = 0u32;
        let mut ranges = Vec::new();
        while let Some((start, end)) = uploader.next_chunk_range() {
            assert_eq!(start, covered_rows, "Chunks must be contiguous");
            assert!(end > start, "Chunk must have positive size");
            ranges.push((start, end));
            covered_rows = end;
            uploader.advance();
        }
        assert_eq!(covered_rows, 1000, "Must cover all rows");
        assert!(uploader.is_complete());
    }

    #[test]
    fn test_chunk_alignment_to_16() {
        // Auto chunk size for a 10000x10000 matrix
        let uploader = ChunkedUploader::new(10000, 10000, ChunkConfig { chunk_rows: None });

        // The first chunk should be aligned to 16
        if let Some((start, end)) = uploader.next_chunk_range() {
            assert_eq!(start, 0);
            let chunk_size = end - start;
            // If not the last chunk, must be aligned to 16
            if end < 10000 {
                assert_eq!(
                    chunk_size % WORKGROUP_ALIGNMENT,
                    0,
                    "Chunk size {chunk_size} must be aligned to {WORKGROUP_ALIGNMENT}"
                );
            }
        }
    }

    #[test]
    fn test_small_matrix_single_chunk() {
        let mut uploader = ChunkedUploader::new(10, 10, ChunkConfig { chunk_rows: None });

        let range = uploader.next_chunk_range();
        assert_eq!(range, Some((0, 10)));
        uploader.advance();
        assert!(uploader.is_complete());
        assert_eq!(uploader.next_chunk_range(), None);
    }

    #[test]
    fn test_large_matrix_multiple_chunks() {
        // 8000x8000 = 256MB, should need multiple 16MB chunks
        let mut uploader = ChunkedUploader::new(8000, 8000, ChunkConfig { chunk_rows: None });

        let mut chunk_count = 0;
        while uploader.next_chunk_range().is_some() {
            chunk_count += 1;
            uploader.advance();
        }
        assert!(
            chunk_count > 1,
            "8000x8000 matrix should need multiple chunks, got {chunk_count}"
        );
        assert!(uploader.is_complete());
    }

    #[test]
    fn test_zero_cols_produces_single_chunk() {
        let mut uploader = ChunkedUploader::new(100, 0, ChunkConfig { chunk_rows: None });
        assert_eq!(uploader.next_chunk_range(), Some((0, 100)));
        uploader.advance();
        assert!(uploader.is_complete());
    }

    #[test]
    fn test_next_chunk_after_complete_returns_none() {
        let mut uploader = ChunkedUploader::new(10, 10, ChunkConfig { chunk_rows: None });
        uploader.advance();
        assert!(uploader.is_complete());
        assert_eq!(uploader.next_chunk_range(), None);
        // Calling advance again should not panic
        uploader.advance();
        assert_eq!(uploader.next_chunk_range(), None);
    }

    #[test]
    fn test_explicit_chunk_rows_larger_than_matrix() {
        let mut uploader = ChunkedUploader::new(
            50,
            100,
            ChunkConfig {
                chunk_rows: Some(1000),
            },
        );
        // chunk_rows clamped to rows
        assert_eq!(uploader.next_chunk_range(), Some((0, 50)));
        uploader.advance();
        assert!(uploader.is_complete());
    }

    #[test]
    fn test_single_row_matrix() {
        let mut uploader = ChunkedUploader::new(1, 1000, ChunkConfig { chunk_rows: None });
        assert_eq!(uploader.next_chunk_range(), Some((0, 1)));
        uploader.advance();
        assert!(uploader.is_complete());
    }
}
