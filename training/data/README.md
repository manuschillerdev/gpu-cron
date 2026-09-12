# Cron data and evaluation

`authored.jsonl` contains 200 requests composed and annotated by Codex separately from the generator: 120 supported, 52 unsupported and 28 ambiguous. They are assistant-authored examples, not collected user traffic. There is no review gate. Their text and structured schedule targets are unchanged from the previous iteration. Their supported meanings remain excluded from generated training data.

Records contain source/provenance, canonical `groupId`, family, token offsets, semantic roles, target status/schedule and a language category. `clause` preserves each original coarse annotation. `annotationVersion: 2` records the deterministic refinement from those clause annotations to hour/minute/weekday/etc. No model predictions or compiler outputs create the labels. The annotation migration lives in `semantic_labels()` and is inspectable alongside the original clauses.

The 16 roles are defined in `training/data.py` and mirrored in `src/features.ts`. The generator renders labelled clauses from structured meanings, then refines value roles. The compiler separately decodes values from predicted roles. A supported target describes the intended representable recurrence, not whether the current model succeeds. Ambiguous requests have no invented time. Multiple clocks, business calendars, embedded timezones, date bounds and event triggers are outside the current schedule contract.

## Generation

Canonical positive meaning determines the split before rendering. Family is omitted from the grouping key so equivalent recurrences cannot leak under different family names. All paraphrases of a generated meaning remain together. Authored meanings are reserved first. Unsupported generator categories stay in training; generated development scores therefore cover supported schedules only, while authored evaluation measures unsupported/ambiguous behavior separately.

The generator uses a deterministic surface seed to render the training meanings with digital and spoken clocks, relative clock arithmetic, lists/ranges, ordinals, task phrases, punctuation, alternate weeks and reordered exclusions. Evaluation rendering uses fixed seeds. Preparation writes the samples as JSONL, one annotated object per line. The trainer loads them once and reuses those samples in every epoch, resampling balanced batches without creating new text. It reads semantic roles exactly as stored; it does not refine or regenerate annotations while loading. `manifest.json` records split identities and hashes. `src/model/vocabulary.json` is an audit list of training words; it is not a runtime vocabulary or input lookup.

```sh
mise exec -- pnpm run prepare:cron-data
mise run train:cron
mise exec -- pnpm run evaluate:cron:candidate
mise exec -- pnpm run promote:cron
```

Inspect or edit the JSONL before fitting. Token offsets and labels must still align with the text, and meanings must not leak between splits. Use `--data-dir PATH` to read a separate directory with the same four JSONL filenames. Checks and evaluation do not regenerate inputs; rerunning preparation explicitly replaces generated files.

Training writes a candidate manifest, a snapshot of the loaded data, weights, and fixtures under ignored `training/candidate/`. Promotion reruns WebGPU/MLX parity, exact-schedule and rejection floors, and the complete-bundle size budget before replacing shipped artifacts. The generated JSONL files are ignored by git and reproducible. The authored collection, six offline assistant paraphrases, frozen parents, generator and manifest are versioned. Training samples families evenly, applies 8% token embedding dropout and 10% context dropout and uses six-bit fake quantization in the final 12 epochs. Augmentation is disabled for evaluation/export.

## Offline paraphrases

The six existing `paraphrases.jsonl` entries inherit their frozen training parents' exact meanings. No separate teacher was downloaded or run. The interchange tool can prepare prompts or import externally produced proposals:

```sh
mise exec -- uv run --locked --project training python training/paraphrases.py prompts --limit 20
mise exec -- uv run --locked --project training python training/paraphrases.py import --file proposals.jsonl --model MODEL_NAME
```

The interchange accepts coarse labelled pieces and refines them into semantic roles. Import verifies parent membership, unchanged target and token alignment; those checks cannot prove meaning fidelity. Text and provenance remain inspectable without an approval gate.

## Evaluation

```sh
mise exec -- pnpm run evaluate:cron
mise exec -- pnpm run evaluate:cron:holdout
```

The historical `holdout` command/file names are retained for compatibility. **These collections are now development benchmarks:** previous results informed generator, model and compiler changes. “But not” is now taught in training; its evaluation still uses separate meanings, not a withheld construction. A fresh untouched collection is required for a new generalization claim.

Actual WebGPU inference is measured separately from compilation. Reports include family/token accuracy, exact role-sequence accuracy, exact structured schedules, unsupported/ambiguous rejection rates, and compiler accuracy with annotated roles. The last measure isolates compiler capacity. Wrongly accepting unsupported language is reported separately; rejecting a supported request counts as a failure.

Exact comparison uses minutes, hours, month days, weekdays, months and week interval, fixing timezone/reference externally and excluding redundant family labels. Authored exact schedule accuracy remains comparable with the original 23/120 baseline. Token accuracy does not: the task changed from five coarse labels to 16 semantic roles. Generated datasets also changed, so do not describe their before/after totals as the same examples.

Historical summary reports live in `training/evaluation/`. New reports default to ignored `test-artifacts/evaluation/` (use the evaluator's `--write` flag to refresh committed reports); candidate reports stay in `training/candidate/evaluation/`. The regular check task enforces the floors in `training/quality.json` across all three collections; full predictions and diagnostics are in ignored `test-artifacts/cron-*-predictions.json`. The evaluator checks weights, manifest and audit-vocabulary hashes against the training report. No new regression tests were added for this iteration.
