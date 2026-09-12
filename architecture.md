# Training and deployment

The cron model trains in MLX on Apple Metal. Browser deployment uses specialized WGSL and embedded packed weights, with no ONNX loader, runtime dependencies or CPU inference fallback.

CPU preparation mechanically tokenizes text and extracts word/shape features. The GPU runs learned embeddings, bidirectional scans, semantic-role predictions and family scores. CPU decoding interprets the predicted values and resolves calendar arithmetic.

## Cron network

The 35,783-parameter model has three embedding tables: 1,024 full-word hash rows, 256 consonant-hash rows and 32 shape rows. Shape encodes token kind, length bucket and capitalization. Each token carries these bits in an exactly representable float32 integer; zero is padding. The three learned embeddings sum with a 1/√3 scale. There is no vocabulary lookup or shared unknown-word ID. Hash collisions remain possible, but two words must collide across multiple features to become indistinguishable.

Embeddings produce 24 sigmoid gates and 24 tanh candidates. Forward/backward recurrences use `h = a*h_previous+b`. Associative composition `(a2*a1, b2+a2*b1)` permits logarithmic parallel scans. Concatenating the embedding and both contexts feeds a 72 → 32 ReLU → 23 head. Seven channels pool over non-padding tokens into family scores. Sixteen channels predict semantic roles: filler (`prefix`), recurrence marker, interval quantity, hour, minute, meridiem, clock offset, clock direction, weekday, range, month day, month, excluded weekday, excluded month, exclusion marker and unresolved (`unknown`).

The browser performs no semantic interpretation before inference. The compiler reads only the predicted roles, decodes their values and applies calendar arithmetic. For example, `quarter:clock-offset`, `to:clock-direction`, `seven:hour`, `evening:meridiem` becomes 18:45. It never searches the whole sentence to recover a missing role or override the network's family. Unknown roles, unconsumed ranges and inconsistent fields produce diagnostics. Interval schedules reject included weekdays and additional clock roles instead of dropping them. Numeric recurrence tokens without a quantity role are rejected, and meridiem/direction values are validated. Wrongly predicted filler can still discard meaningful language: this remains a measurable model error, not a guaranteed validation boundary.

`result.model.tokens` exposes all raw labels and softmax scores even when compilation fails. The demo displays network predictions separately from compiler output. No confidence threshold or CPU fallback is used.

## Training and evaluation

`training/data.py` generates meaning first and renders language independently of the runtime compiler. Positive meanings are assigned to disjoint splits before rendering; all authored supported meanings remain reserved. Data preparation renders the meanings to annotated JSONL files. Training loads those files once and resamples batches from the fixed records each epoch, without rendering new text. Variations compose spoken/digital clocks, ranges, lists, singular/plural weekdays, ordinals, alternate weeks, task prefixes, punctuation, clause order and exclusions. Unsupported generator categories stay in training, with separate authored unsupported requests providing development evaluation.

MLX uses compiled autodiff, AdamW, gradient clipping, warmup/cosine decay and token embedding dropout. Only the final 12 epochs apply deployment-matched six-bit fake quantization. Model, optimizer, RNG and split identity are checkpointed; resumed epochs regenerate their own deterministic data. No augmentation is applied during export or inference. Input JSONL is validated without regenerating text or labels; a snapshot of the loaded records is written beside the candidate. The trainer writes artifacts to `training/candidate/` (resume checkpoints remain under `training/`); the separate promotion command reruns WebGPU/MLX parity and exact-schedule/rejection regression floors on all three development collections and the complete-bundle size budget before copying artifacts. ONNX export is an optional verification step after promotion.

The existing authored collection and former “but not” holdout have informed development. They are no longer untouched benchmarks, and “but not” is now taught. Their requests are not fitted by the trainer. The authored texts and schedule targets remain unchanged; old clause annotations were refined into semantic roles without using model predictions or compiler outputs. Token scores therefore cannot be directly compared with the old five-label task. Exact schedule accuracy still compares the same authored targets. See [MODEL_CARD.md](MODEL_CARD.md) for actual WebGPU measurements.

## Runtime and export

`src/model/runtime.ts` specializes embedding/gating, scan composition, projection, token classification and family pooling in WGSL. It reuses devices, pipelines, weights, buffers and bind groups. Calls are queued with input snapshots; capacity grows geometrically and batches are chunked at 128. Inputs have 512 padded slots, while dispatch uses the next power of two covering actual tokens. Device loss is an error.

The signed six-bit weights occupy 26,838 packed bytes and expand to 143,132 float32 bytes. ONNX describes the same decoded coefficients and complete graph solely for development verification. Numerical references are not imported by deployed entrypoints. Existing checks compare MLX, ONNX and real Chromium WebGPU, including buffer lifecycle and concurrent calls.

The complete cron library has a 40,000-byte Brotli budget, raised from 30,000 for this richer network. Size measurements include preparation, decoding, metadata, weights and WGSL. Demo HTML/CSS are separate. Production checks reject ONNX/WASM payloads and runtime dependencies.

## References

Cron follows the feature-based semantic tagging and bidirectional affine scans in [gpu-time](https://github.com/arikchakma/gpu-time). Packed weights and specialized deployment follow [gpu-lexer](https://github.com/vercel-labs/gpu-lexer). This is an original MLX implementation tailored to cron, not an exact port: the role set is smaller than gpu-time's and no gpu-lexer whole-document tree is included.
