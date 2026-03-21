#!/bin/bash

# Extract the shell command Claude is attempting to run
COMMAND=$(jq -r '.tool_input.command // empty')

# If no command is found (e.g., using a different tool), allow it to proceed
if [ -z "$COMMAND" ]; then
  exit 0
fi

# Define an array of regex patterns for dangerous commands
DANGEROUS_PATTERNS=(
  "rm -rf"
  "rm -fr"
  "rm -r \."
  "rm -r \*"
  "mkfs"
  "> /dev/null" # Often used to hide critical errors
)

# Loop through patterns and check if the command matches
for PATTERN in "${DANGEROUS_PATTERNS[@]}"; do
  if echo "$COMMAND" | grep -qE "$PATTERN"; then
    # Output the error to stderr and exit 1 to block the tool use
    echo "SECURITY BLOCK: Attempted to run a destructive command." >&2
    echo "Blocked command: $COMMAND" >&2
    echo "Action required: If this deletion is absolutely necessary, ask the user to execute it manually." >&2
    exit 1
  fi
done

# If no dangerous patterns are matched, allow the command to proceed
exit 0