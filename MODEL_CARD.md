# gpu-cron model card

## Architecture

A **35,783-parameter feature-sum bidirectional affine-scan tagger**, trained in MLX on Apple Metal and deployed as specialized WebGPU shaders. Mechanical tokens preserve source offsets. Each token supplies a full-word hash, consonant hash and shape (kind, length, case), addressing 1,024 + 256 + 32 learned 24-dimensional embedding rows. Their sum is scaled by 1/√3; there is no vocabulary lookup or shared unknown-word ID. Unfamiliar words retain separate features, though hash collisions remain possible.

A 24 → 48 affine projection produces gates/candidates for parallel forward and backward scans. Each scan applies `h = a*h_previous + b`. Associative composition `(a2*a1, b2+a2*b1)` permits logarithmic parallel scans. Embedding and both contexts feed a 72 → 32 → 23 head. Seven channels pool into family scores; 16 predict semantic token roles: prefix/filler, recurrence, quantity, hour, minute, meridiem, clock-offset, clock-direction, weekday, range, monthday, month, excluded-weekday, excluded-month, exclusion and unknown.

The compiler consumes predicted semantic values and performs number decoding, clock arithmetic and calendar interpretation. It does not match whole sentences, override the predicted family or recover missing roles from raw text. Unknown roles, incompatible fields and unconsumed predicted ranges are rejected. Incorrect filler predictions can still discard real restrictions; successful compilation is not proof of intended meaning. Raw network labels/scores stay visible independently of compiler output. No confidence gate or CPU inference fallback exists.

## Training

The shipped model was trained with the earlier on-the-fly rendering pipeline described here. New candidate runs load fixed JSONL files from disk; the recorded measurements below are not results of that revised workflow.

Data version 12 generates structured meanings first. All supported authored meanings remain reserved from training; generated positive meanings are grouped before rendering. The epoch-zero manifest contains 23,177 training records across 6,819 groups. Each of 45 epochs renders fresh phrasings of those training meanings and samples families evenly. Renderers compose digital/spoken clocks, clock offsets, bare hours, ordinal/list/range forms, singular/plural weekdays, task prose, alternate weeks and reordered exclusions. Six offline assistant paraphrases and their original training examples are preserved in `training/data/curated.jsonl`.

Unsupported generator categories stay in training. Their ordinary words retain semantic roles where possible; unresolved restrictions receive unknown labels. Generated development contains supported requests only, so its score says nothing about rejection quality. The separate authored collection supplies ambiguous and unsupported requests.

MLX 0.32.2 uses compiled autodiff, AdamW, clipping, warmup/cosine decay, 8% token embedding dropout and 10% context dropout. The final 12 epochs use deployment-matched six-bit fake quantization. Dropout is disabled at export/inference. The recorded M2 Max loop took 107.5 seconds, including fresh epoch data preparation. Artifact, source, manifest and fixture hashes are recorded in `training/report.json`.

## Actual WebGPU evaluation

| Collection | Requests | Token accuracy | Exact schedules, supported requests |
| --- | ---: | ---: | ---: |
| Generated development | 2661 | 99.98% | 2649/2661 |
| Former but-not holdout | 489 | 99.96% | 485/489 |
| Separately authored development | 200 | 89.82% | **116/120 (96.67%)** |

The original model produced **23/120** exact authored schedules; the current model produces **116/120** against the same request text and structured targets. With annotated roles supplied, the compiler now produces **120/120**, up from 29/120. Authored family accuracy is 85.50%. Across the 52 unsupported and 28 ambiguous requests, **9/80 are incorrectly accepted**; their separate rejection rates are 86.54% and 92.86%. These remaining errors are exposed rather than repaired by runtime language rules.

These are **development measurements**, not untouched or real-user accuracy. Earlier results informed this iteration. “But not” is now taught, and the generated collection changed, so its current result is not an unseen-construction result. Authored texts were assistant-written separately from the generator. Original clause annotations remain under `clause`; deterministic semantic-role refinement uses neither network nor compiler output. Token accuracy cannot be directly compared with the old five-label task. The trainer does not fit authored requests, but a fresh independent collection is needed for a new generalization claim.

Regenerated reports in ignored `test-artifacts/evaluation/` bind results to weights, dataset, feature encoder, compiler and WGSL hashes. Full predictions and diagnostics live in ignored `test-artifacts/cron-*-predictions.json`. Evaluation separates network output, actual compiler results and compiler results with annotated roles.

## Deployment and limitations

Weights occupy **26,838 packed bytes**, expanded to 143,132 float32 bytes. The complete minified library is 63,922 bytes, or **32,032 bytes (31.3 KiB) Brotli-compressed**. This includes API, features, weights, metadata, decoding and WGSL; demo HTML/CSS are separate. The complete-bundle budget increased from 30,000 to 40,000 bytes for the richer network.

The runtime reuses the device, pipelines, weights and buffers, queues calls and chunks batches at 128. Inputs retain the 512-character limit; training uses a 64-token bucket. Long-input and multilingual generalization are unmeasured. Six-bit quantization and platform differences are checked by comparing the existing quantized MLX fixtures directly with real Chromium WebGPU. No new regression cases were added.

No model loader, WASM payload or runtime dependency ships in the browser.

## Candidate promotion

Training reads `training/data/*.jsonl` once (or a supplied `--data-dir`), uses the same records each epoch with balanced family sampling, and produces isolated candidate artifacts. Input labels are preserved and validated, and the loaded dataset is snapshotted beside the candidate. The trainer starts from the fixed seed on each run; checkpoint/resume support is omitted. Generate the files explicitly with `prepare:cron-data`; checks never regenerate them. Promotion requires fresh WebGPU/MLX parity, the complete-bundle size budget, and the mixed development regression floors in `training/quality.json`. The floors retain the recorded 116/120 authored supported schedules and at most 9/80 false acceptances; they prevent regression and do not claim to solve those remaining failures. Shipped coefficients and fixtures were not regenerated for the pipeline refactor. Original source and manifest hashes remain in the report under `sourceMaintenance`. The maintained generator is version 13: removing unused templates changes training wording, while preserving split meanings and evaluation records. The shipped measurements above still describe the version-12 model.

## Reproduce

```sh
mise exec -- pnpm run prepare:cron-data
mise run train:cron
mise exec -- pnpm run evaluate:cron:candidate
mise exec -- pnpm run promote:cron
mise run check
```
