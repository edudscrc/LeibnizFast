Add a new interaction mode named `$ARGUMENTS` to LeibnizFast following TDD.

Steps:
1. Read `src/interaction.rs` fully — understand `InteractionState`, `mouse_down`/`mouse_move`/`mouse_up` transitions, and existing tests.
2. Read `src/lib.rs` event handlers (`on_mouse_down`, `on_mouse_move`, `on_mouse_up`, `on_wheel`) to understand how interaction state connects to rendering.
3. **Write tests first** in `src/interaction.rs` — cover all state transitions into and out of the new mode.
4. Add the variant to `InteractionState` enum.
5. Add transition logic in `mouse_down`, `mouse_move`, `mouse_up` as needed.
6. Wire up in `src/lib.rs` event handlers — handle the new variant to produce the correct camera/render effect.
7. Run `cargo test` to confirm all tests pass.
