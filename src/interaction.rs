//! # Interaction
//!
//! State machine for mouse/pointer events. Handles transitions between
//! idle, dragging (pan), and hover states without tangled boolean flags.

/// Result of processing a mouse event — tells the caller what action to take.
#[derive(Debug, PartialEq)]
pub enum InteractionResult {
    /// No action needed
    None,
    /// Pan the camera by (dx, dy) screen pixels
    Pan { dx: f32, dy: f32 },
    /// Mouse is hovering — caller should do tooltip lookup
    Hover,
}

/// Mouse interaction state machine.
///
/// Transitions:
/// - `Idle` → mouse_down → `Dragging`
/// - `Dragging` → mouse_move → `Pan` result
/// - `Dragging` → mouse_up → `Idle`
/// - `Idle` → mouse_move → `Hover` result
#[derive(Debug)]
pub enum InteractionState {
    /// No mouse button pressed
    Idle,
    /// Mouse button held, tracking last position for delta computation
    Dragging { last_x: f32, last_y: f32 },
}

impl InteractionState {
    /// Create a new interaction state (starts idle).
    pub fn new() -> Self {
        Self::Idle
    }

    /// Handle mouse button press at the given screen position.
    pub fn mouse_down(&mut self, x: f32, y: f32) {
        *self = Self::Dragging {
            last_x: x,
            last_y: y,
        };
    }

    /// Handle mouse movement. Returns the appropriate action.
    ///
    /// In `Dragging` state: returns `Pan` with the delta from last position.
    /// In `Idle` state: returns `Hover` so the caller can do tooltip lookup.
    pub fn mouse_move(&mut self, x: f32, y: f32) -> InteractionResult {
        match *self {
            Self::Dragging {
                ref mut last_x,
                ref mut last_y,
            } => {
                let dx = x - *last_x;
                let dy = y - *last_y;
                *last_x = x;
                *last_y = y;
                InteractionResult::Pan { dx, dy }
            }
            Self::Idle => InteractionResult::Hover,
        }
    }

    /// Handle mouse button release. Returns to `Idle` state.
    pub fn mouse_up(&mut self) {
        *self = Self::Idle;
    }
}

impl Default for InteractionState {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_initial_state_is_idle() {
        let state = InteractionState::new();
        assert!(matches!(state, InteractionState::Idle));
    }

    #[test]
    fn test_idle_to_dragging_on_mouse_down() {
        let mut state = InteractionState::new();
        state.mouse_down(100.0, 200.0);
        assert!(matches!(
            state,
            InteractionState::Dragging {
                last_x: 100.0,
                last_y: 200.0
            }
        ));
    }

    #[test]
    fn test_dragging_mouse_move_returns_pan_delta() {
        let mut state = InteractionState::new();
        state.mouse_down(100.0, 200.0);

        let result = state.mouse_move(130.0, 210.0);
        assert_eq!(result, InteractionResult::Pan { dx: 30.0, dy: 10.0 });
    }

    #[test]
    fn test_dragging_mouse_move_updates_last_position() {
        let mut state = InteractionState::new();
        state.mouse_down(100.0, 200.0);

        // First move
        state.mouse_move(130.0, 210.0);

        // Second move — delta should be from (130, 210), not from (100, 200)
        let result = state.mouse_move(140.0, 220.0);
        assert_eq!(result, InteractionResult::Pan { dx: 10.0, dy: 10.0 });
    }

    #[test]
    fn test_dragging_to_idle_on_mouse_up() {
        let mut state = InteractionState::new();
        state.mouse_down(100.0, 200.0);
        state.mouse_up();
        assert!(matches!(state, InteractionState::Idle));
    }

    #[test]
    fn test_idle_mouse_move_returns_hover() {
        let mut state = InteractionState::new();
        let result = state.mouse_move(50.0, 60.0);
        assert_eq!(result, InteractionResult::Hover);
    }

    #[test]
    fn test_full_drag_cycle() {
        let mut state = InteractionState::new();

        // Start idle
        assert!(matches!(state, InteractionState::Idle));

        // Mouse down → dragging
        state.mouse_down(100.0, 100.0);
        assert!(matches!(state, InteractionState::Dragging { .. }));

        // Move → pan
        let result = state.mouse_move(150.0, 120.0);
        assert_eq!(result, InteractionResult::Pan { dx: 50.0, dy: 20.0 });

        // Mouse up → idle
        state.mouse_up();
        assert!(matches!(state, InteractionState::Idle));

        // Move while idle → hover
        let result = state.mouse_move(200.0, 200.0);
        assert_eq!(result, InteractionResult::Hover);
    }

    #[test]
    fn test_double_mouse_down_resets_drag_origin() {
        let mut state = InteractionState::new();
        state.mouse_down(100.0, 200.0);
        // Second mouse_down before mouse_up → resets drag anchor
        state.mouse_down(500.0, 600.0);

        let result = state.mouse_move(510.0, 610.0);
        assert_eq!(
            result,
            InteractionResult::Pan { dx: 10.0, dy: 10.0 },
            "Delta should be relative to second mouse_down"
        );
    }

    #[test]
    fn test_mouse_up_while_idle_is_noop() {
        let mut state = InteractionState::new();
        state.mouse_up();
        assert!(matches!(state, InteractionState::Idle));
    }

    #[test]
    fn test_drag_zero_delta() {
        let mut state = InteractionState::new();
        state.mouse_down(100.0, 200.0);
        let result = state.mouse_move(100.0, 200.0);
        assert_eq!(result, InteractionResult::Pan { dx: 0.0, dy: 0.0 });
    }

    #[test]
    fn test_default_trait() {
        let state = InteractionState::default();
        assert!(matches!(state, InteractionState::Idle));
    }
}
