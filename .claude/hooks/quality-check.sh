#!/bin/bash

# Extract the file path that Claude just edited from the JSON payload
FILE_PATH=$(jq -r '.tool_input.file_path')

# Exit cleanly if no file was found
if [ -z "$FILE_PATH" ] || [ "$FILE_PATH" == "null" ]; then
  exit 0
fi

echo "Running checks for: $FILE_PATH"

case "$FILE_PATH" in
  # RUST WORKFLOW
  *.rs)
    echo "--> Formatting Rust..."
    cargo fmt -- "$FILE_PATH"
    
    echo "--> Running Clippy..."
    if ! cargo clippy -- -D warnings; then
        echo "Error: Clippy failed. Please fix the warnings." >&2
        exit 2
    fi
    
    echo "--> Running Rust Tests..."
    if ! cargo test; then
        echo "Error: Rust tests failed. Please review the output and fix the logic." >&2
        exit 2
    fi

    echo "--> Building WebAssembly..."
    if ! wasm-pack build --target web; then
        echo "Error: wasm-pack build failed. Ensure your Rust code is compatible with the Wasm target." >&2
        exit 2
    fi
    ;;
    
  # TYPESCRIPT WORKFLOW
  *.ts|*.tsx)
    echo "--> Formatting TypeScript..."
    npx prettier --write "$FILE_PATH"

    echo "--> Linting TypeScript..."
    if ! npx eslint "$FILE_PATH"; then
        echo "Error: ESLint failed. Please fix the linting errors." >&2
        exit 2
    fi
    
    echo "--> Type Checking..."
    if ! npx tsc --noEmit; then
        echo "Error: TypeScript type-check failed." >&2
        exit 2
    fi
    
    echo "--> Running TypeScript Tests..."
    if ! npm run test; then
        echo "Error: TypeScript tests failed." >&2
        exit 2
    fi
    ;;
    
  # WEBGPU (WGSL) WORKFLOW
  *.wgsl)
    echo "--> Formatting WebGPU Shader..."
    npx prettier --plugin=prettier-plugin-wgsl --write "$FILE_PATH"
    ;;
    
  *)
    exit 0
    ;;
esac

echo "All checks passed successfully."
exit 0