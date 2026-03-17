Run all LeibnizFast quality checks and report results.

Execute in order, stopping on first failure:

```bash
cd /home/edudscrc/repos/LeibnizFast
cargo fmt --check
cargo clippy -- -D warnings
cargo test
npx prettier --check js/
npx eslint js/
```

Report which checks passed and which failed. If any fail, show the relevant error output and suggest a fix.
