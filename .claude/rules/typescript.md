# TypeScript Standards

## API Design for Library Authors
* **Strict Typing:** Enable `strict: true` in `tsconfig.json`. Do not use `any`. Use `unknown` if the type is truly dynamic, and narrow it with type guards.
* **Explicit Exports:** Only export what is strictly necessary for the public API. Keep internal utility functions private to the module.
* **JSDoc Comments:** Every public-facing class, interface, and function must have complete JSDoc comments detailing parameters, return types, and usage examples. This provides intellisense for the developers using the library.

## Formatting & Syntax
* Prefer `interface` over `type` for public API shapes to allow for declaration merging.
* Use ES Modules (`import`/`export`) rather than CommonJS.
* Keep the TypeScript layer thin; its primary job is memory management coordination between JavaScript and the WebAssembly/WebGPU layers.