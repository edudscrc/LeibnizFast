# General Coding Standards

## 1. Test-Driven Development (TDD)
* **Write Tests First:** Never write implementation logic before the corresponding test exists.
* **Test Coverage:** Ensure both the "happy path" and edge cases (e.g., null inputs, out-of-bounds indices, memory allocation failures) are covered.
* **Isolation:** Tests must not depend on the execution order of other tests.

## 2. Code Maintainability
* **No Magic Numbers:** Never hardcode numeric or string literals in the middle of logic. Extract all configuration values, grid sizes, memory limits, and offsets into clearly named constants at the top of the file or in a dedicated configuration module.
* **Descriptive Naming:** Variable and function names must describe their exact intent. Favor `compute_matrix_transform` over `calc_mtx`. 
* **Single Responsibility:** Functions and methods should do one thing. If a function contains the word "and" in its description, it should likely be split apart.
* **Fail Fast:** Validate inputs and parameters immediately at the boundaries of the API. Throw clear, descriptive errors if inputs are invalid rather than allowing silent failures deeper in the execution stack.