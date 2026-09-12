# gpu-cron

- Use mise for tool versions and project tasks, pnpm for every JavaScript dependency operation, and uv for every Python operation. Do not use npm, npx, or pip. Training: `uv run --locked --project training python training/train.py`. Never invoke Python or pip directly.
- Training uses MLX. The browser package uses specialized WGSL and packed learned weights with zero runtime dependencies. ONNX exports and ONNX Runtime are development-only verification tools. Keep deployment and training independent. WebGPU is mandatory: no CPU inference backends or automatic fallback.
- Cron uses summed hashed feature embeddings, bidirectional affine scans, family scores and semantic token roles. Keep learned semantic-role/family predictions distinct from exact value interpretation and calendar semantics. Never present family/token metrics as end-to-end parsing accuracy.
- Reject unconsumed syntax. Never silently discard a schedule restriction or emit an approximate cron expression.
- Preview and cron use local wall-clock fields. Preserve the documented reference, timezone, alternate-week anchor, and DST policies.
- Verify with `mise run check`, including real browser WebGPU/reference parity. If WebGPU cannot be tested, report that explicitly.

- Keep `pnpm-lock.yaml` and `training/uv.lock` committed. Setup uses frozen/locked installs. Do not regenerate trained artifacts for dependency-tooling changes.
