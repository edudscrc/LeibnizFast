# Rust Standards

## Idiomatic Code
* **Clippy is Law:** Code must pass `cargo clippy` without warnings. Treat Clippy suggestions as requirements.
* **Formatting:** All code must be formatted using standard `cargo fmt`.
* **Error Handling:** Never use `unwrap()` or `expect()` in production code. Always propagate errors using the `?` operator and return `Result<T, E>` with custom, descriptive error types.

## Architecture & Memory
* **Wasm Interop:** Use `wasm-bindgen` carefully. Minimize crossing the boundary between JavaScript and Wasm, as serializing/deserializing data across this boundary is expensive. Pass pointers to linear memory when transferring large arrays or buffers.
* **Inline Tests:** Place unit tests in the same file as the implementation within a `#[cfg(test)]` module block.