# gpu-cron model card

## Architecture

A **35,783-parameter feature-sum bidirectional affine-scan tagger**, trained in MLX on Apple Metal and deployed as specialized WebGPU shaders. Mechanical tokens preserve source offsets. Each token supplies a full-word hash, consonant hash and shape (kind, length, case), addressing 1,024 + 256 + 32 learned 24-dimensional embedding rows. Their sum is scaled by 1/√3; there is no vocabulary lookup or shared unknown-word ID. Unfamiliar words retain separate features, though hash collisions remain possible.

A 24 → 48 affine projection produces gates/candidates for parallel forward and backward scans. Each scan applies `h = a*h_previous + b`. Associative composition `(a2*a1, b2+a2*b1)` permits logarithmic parallel scans. Embedding and both contexts feed a 72 → 32 → 23 head. Seven channels pool into family scores; 16 predict semantic token roles: prefix/filler, recurrence, quantity, hour, minute, meridiem, clock-offset, clock-direction, weekday, range, monthday, month, excluded-weekday, excluded-month, exclusion and unknown.

The compiler consumes predicted semantic values and performs number decoding, clock arithmetic and calendar interpretation. It does not match whole sentences, override the predicted family or recover missing roles from raw text. Unknown roles, incompatible fields and unconsumed predicted ranges are rejected. Incorrect filler predictions can still discard real restrictions; successful compilation is not proof of intended meaning. Raw network labels/scores stay visible independently of compiler output. No confidence gate or CPU inference fallback exists.

## Training

The shipped model was trained from the committed JSONL files. Data version 13 generates structured meanings before wording and keeps paraphrases of the same meaning in one split. All supported authored meanings remain reserved from training. The training file contains 23,177 records across 6,819 meaning groups. Each of 60 epochs samples families evenly from those fixed records. Six curated assistant paraphrases and their original training examples are preserved in `training/data/curated.jsonl`.

Unsupported generator categories stay in training. Their ordinary words retain semantic roles where possible; unresolved restrictions receive unknown labels. Generated development contains supported requests only, so its score says nothing about rejection quality. The separate authored collection supplies ambiguous and unsupported requests.

MLX 0.32.2 uses compiled autodiff, AdamW, clipping, warmup/cosine decay, 8% token embedding dropout and 10% context dropout. The final 12 epochs use deployment-matched six-bit fake quantization. Dropout is disabled at export/inference. The recorded M2 Max training loop took 14.3 seconds. Artifact, source, manifest and fixture hashes are recorded in `training/report.json`.

## Actual WebGPU evaluation

| Collection | Requests | Token accuracy | Exact schedules, supported requests |
| --- | ---: | ---: | ---: |
| Generated development | 2661 | 99.98% | 2653/2661 |
| Former but-not holdout | 489 | 99.99% | 488/489 |
| Separately authored development | 200 | 89.76% | **117/120 (97.50%)** |

The original model produced **23/120** exact authored schedules; the current model produces **117/120** against the same request text and structured targets. With annotated roles supplied, the compiler produces **120/120**. Authored family accuracy is 84.50%. Across the 52 unsupported and 28 ambiguous requests, **5/80 are incorrectly accepted**; their separate rejection rates are 92.31% and 96.43%. These remaining errors are exposed rather than repaired by runtime language rules.

These are **development measurements**, not untouched or real-user accuracy. Earlier results informed this iteration. “But not” is now taught, and the generated collection changed, so its current result is not an unseen-construction result. Authored texts were assistant-written separately from the generator. Original clause annotations remain under `clause`; deterministic semantic-role refinement uses neither network nor compiler output. Token accuracy cannot be directly compared with the old five-label task. The trainer does not fit authored requests, but a fresh independent collection is needed for a new generalization claim.

Evaluation prints network metrics, actual compiler results, and compiler results with annotated roles separately. Model/data identity and quality floors are checked directly; no evaluation reports are saved.

## Deployment and limitations

Weights occupy **26,838 packed bytes**, expanded to 143,132 float32 bytes. The complete minified library is 64,980 bytes, or **31,511 bytes (30.8 KiB) Brotli-compressed**. This includes API, features, weights, metadata, decoding and WGSL; demo HTML/CSS are separate. The complete-bundle budget is 40,000 bytes.

The runtime reuses the device, pipelines, weights and buffers, queues calls and chunks batches at 128. Inputs retain the 512-character limit; training uses a 64-token bucket. Long-input and multilingual generalization are unmeasured. Six-bit quantization and platform differences are checked by comparing the existing quantized MLX fixtures directly with real Chromium WebGPU. No new regression cases were added.

No model loader, WASM payload or runtime dependency ships in the browser.

## Candidate promotion

Training reads `training/data/*.jsonl` once (or a supplied `--data-dir`), uses the same records each epoch with balanced family sampling, and produces isolated candidate artifacts. Input labels are preserved and validated, and the loaded dataset is snapshotted beside the candidate. The trainer starts from the fixed seed on each run; checkpoint/resume support is omitted. Generate the files explicitly with `prepare:cron-data`; checks never regenerate them. Promotion requires fresh WebGPU/MLX parity, the complete-bundle size budget, and the development regression floors in `training/quality.json`. The floors preserve the measurements above while remaining development checks rather than generalization claims.

## Reproduce

```sh
mise exec -- pnpm run prepare:cron-data
mise run train:cron
mise exec -- pnpm run evaluate:cron:candidate
mise exec -- pnpm run promote:cron
mise run check
```
