# WebGPU and WGSL Standards

## WGSL (WebGPU Shading Language)
* **Workgroup Optimization:** Explicitly document the reasoning behind your `@workgroup_size` choices in the shader comments. 
* **Buffer Alignment:** Pay strict attention to memory alignment rules (e.g., `vec3` alignment issues in arrays). Prefer explicit padding or using `vec4` to avoid cross-platform layout mismatch errors.
* **Modular Shaders:** If the API supports it, separate WGSL code into logical modules rather than massive single-file shaders.

## Resource Management
* **Pipeline Caching:** Create compute and render pipelines once during initialization and reuse them. Do not create new pipelines inside the main render/compute loop.
* **Explicit Destruction:** Ensure that WebGPU resources (buffers, textures, bind groups) are explicitly destroyed when the library user calls a cleanup or disposal method, preventing memory leaks in the browser.