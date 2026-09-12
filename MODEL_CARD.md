# gpu-cron model card

## Architecture

A **35,783-parameter feature-sum bidirectional affine-scan tagger**, trained in MLX on Apple Metal and deployed as specialized WebGPU shaders. Mechanical tokens preserve source offsets. Each token supplies a full-word hash, consonant hash and shape (kind, length, case), addressing 1,024 + 256 + 32 learned 24-dimensional embedding rows. Their normalized sum replaces the old 182-entry word vocabulary and shared unknown-word ID. Unfamiliar words retain separate features, though hash collisions remain possible.

A 24 → 48 affine projection produces gates/candidates for parallel forward and backward scans. Embedding and both contexts feed a 72 → 32 → 23 head. Seven channels pool into family scores; 16 predict semantic token roles: prefix/filler, recurrence, quantity, hour, minute, meridiem, clock-offset, clock-direction, weekday, range, monthday, month, excluded-weekday, excluded-month, exclusion and unknown.

The compiler consumes predicted semantic values and performs number decoding, clock arithmetic and calendar interpretation. It does not match whole sentences, override the predicted family or recover missing roles from raw text. Unknown roles, incompatible fields and unconsumed predicted ranges are rejected. Incorrect filler predictions can still discard real restrictions; successful compilation is not proof of intended meaning. Raw network labels/scores stay visible independently of compiler output. No confidence gate or CPU inference fallback exists.

## Training

Data version 12 generates structured meanings first. All supported authored meanings remain reserved from training; generated positive meanings are grouped before rendering. The epoch-zero manifest contains 23,177 training records across 6,819 groups. Each of 45 epochs renders fresh phrasings of those training meanings and samples families evenly. Renderers compose digital/spoken clocks, clock offsets, bare hours, ordinal/list/range forms, singular/plural weekdays, task prose, alternate weeks and reordered exclusions. Six existing offline assistant paraphrases retain frozen training parents.

Unsupported generator categories stay in training. Their ordinary words retain semantic roles where possible; unresolved restrictions receive unknown labels. Generated development contains supported requests only, so its score says nothing about rejection quality. The separate authored collection supplies ambiguous and unsupported requests.

MLX 0.32.2 uses compiled autodiff, AdamW, clipping, warmup/cosine decay, 8% token embedding dropout and 10% context dropout. The final 12 epochs use deployment-matched six-bit fake quantization. Dropout is disabled at export/inference. The recorded M2 Max loop took 107.5 seconds, including fresh epoch data preparation. Model/optimizer/RNG checkpoints and deterministic epoch seeds support resume. Artifact, source, manifest and fixture hashes are recorded in `training/report.json`.

## Actual WebGPU evaluation

| Collection | Requests | Token accuracy | Exact schedules, supported requests |
| --- | ---: | ---: | ---: |
| Generated development | 2661 | 99.98% | 2649/2661 |
| Former but-not holdout | 489 | 99.96% | 485/489 |
| Separately authored development | 200 | 89.82% | **116/120 (96.67%)** |

The original model produced **23/120** exact authored schedules; the current model produces **116/120** against the same request text and structured targets. With annotated roles supplied, the compiler now produces **120/120**, up from 29/120. Authored family accuracy is 85.50%. Across the 52 unsupported and 28 ambiguous requests, **9/80 are incorrectly accepted**; their separate rejection rates are 86.54% and 92.86%. These remaining errors are exposed rather than repaired by runtime language rules.

These are **development measurements**, not untouched or real-user accuracy. Earlier results informed this iteration. “But not” is now taught, and the generated collection changed, so its current result is not an unseen-construction result. Authored texts were assistant-written separately from the generator. Original clause annotations remain under `clause`; deterministic semantic-role refinement uses neither network nor compiler output. Token accuracy cannot be directly compared with the old five-label task. The trainer does not fit authored requests, but a fresh independent collection is needed for a new generalization claim.

Reports in `training/evaluation/` bind results to weights, dataset, feature encoder, compiler and WGSL hashes. Full predictions and diagnostics live in ignored `test-artifacts/cron-*-predictions.json`. Evaluation separates network output, actual compiler results and compiler results with annotated roles.

## Deployment and limitations

Weights occupy **26,838 packed bytes**, expanded to 143,132 float32 bytes. The complete minified library is 60,221 bytes, or **31,595 bytes (30.9 KiB) Brotli-compressed**. This includes API, features, weights, metadata, decoding and WGSL; demo HTML/CSS are separate. The complete-bundle budget increased from 30,000 to 40,000 bytes for the richer network.

The runtime reuses the device, pipelines, weights and buffers, queues calls and chunks batches at 128. Inputs retain the 512-character limit; training uses a 64-token bucket. Long-input and multilingual generalization are unmeasured. Six-bit quantization and platform differences are checked against MLX and the optional ONNX verification graph using the existing fixtures and real Chromium WebGPU. No new regression cases were added.

ONNX remains a development-only portable export (175,307 bytes). No ONNX, WASM loader or runtime dependency ships in the browser.

## Reproduce

```sh
mise run train:cron
mise exec -- pnpm run evaluate:cron
mise exec -- pnpm run evaluate:cron:holdout
mise run check
```
