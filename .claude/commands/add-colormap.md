Add a new colormap named `$ARGUMENTS` to LeibnizFast following TDD.

Steps:
1. Read `src/colormap_data.rs` to understand the existing colormap format (`[[u8; 3]; 256]` const arrays).
2. Read `src/colormap.rs` to see `COLORMAP_NAMES`, `get_colormap_by_name`, and existing tests.
3. Read `js/types.ts` to see the `ColormapName` union type.
4. **Write the test first** in `src/colormap.rs` — add `test_<name>_known_values` that checks at least index 0, 128, and 255.
5. Add the `[[u8; 3]; 256]` const to `src/colormap_data.rs`.
6. Add the name to `COLORMAP_NAMES` and `get_colormap_by_name` in `src/colormap.rs`.
7. Add the string literal to `ColormapName` in `js/types.ts`.
8. Run `cargo test` to confirm all tests pass.

Naming conventions: const name is `SCREAMING_SNAKE_CASE`, JS/match name is lowercase with hyphens (e.g. `"plasma"`, `"cool-warm"`).
