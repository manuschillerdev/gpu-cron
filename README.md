# gpu-cron

Turn English recurring schedules into structured calendar rules, five-field cron, and timezone-aware previews. A 35,783-parameter MLX-trained sequence model reads context in both directions and predicts semantic roles for clock values, weekdays and exclusions on WebGPU. TypeScript interprets those predicted values and resolves the exact calendar. There is no CPU inference fallback.

The browser package embeds packed learned weights and a purpose-built WGSL runtime with zero runtime dependencies. MLX trains the model; real browser checks compare the WGSL outputs directly with quantized MLX fixtures.

## Run locally

**mise** pins Node.js 26.1.0, Python 3.12.14, pnpm 10.32.1, and uv 0.12.5. **pnpm** manages JavaScript dependencies; **uv** manages Python dependencies and the training environment. Both lockfiles are committed.

```sh
mise trust
mise install
mise run setup
mise exec -- pnpm run prepare:cron-data
mise run dev
```

Open the localhost URL printed by Vite. The playground performs all parsing locally; it does not call an API or load fonts from a third party.

```sh
pnpm run build          # publishable ESM and declarations in dist/
pnpm test               # parsing, calendar, quantized-model parity, provenance
pnpm run test:browser   # real Chromium WebGPU + desktop/mobile playground checks
pnpm run build:demo     # static site in site/
pnpm run size           # bundled/minified/Brotli bytes; 40,000-byte budget
pnpm run check          # all library and browser checks
```

The build uses the [Go-based TypeScript 7 compiler](https://devblogs.microsoft.com/typescript/announcing-typescript-7-0-rc/) (`tsgo`), pinned in the pnpm lockfile. Python checks use Ruff, ty, and pytest; currently there are no Python tests to collect. The TypeScript tests and browser checks are included in strict type-checking. Node runs the test files using its built-in type stripping.

Use `mise run check` for the complete suite or `mise run build` for the library and demo. With mise activated in your shell, the pnpm/uv commands below use the pinned tools; otherwise prefix them with `mise exec --`. `mise run setup` installs from both lockfiles without updating them. Python selection follows the mise `UV_PYTHON` setting.

Setup downloads the Chromium revision pinned by Playwright through `pnpm exec playwright install chromium`; tests use that managed browser with WebGPU. They fail if GPU inference is unavailable; they never count a CPU fallback as a GPU pass.

## API

After building, import `./dist/index.js` locally. The package export is `gpu-cron` when installed; this repository has not been published to npm.

```ts
import { parse, parseMany, defineParser } from 'gpu-cron';

const result = await parse('Every weekday at 9:30am', {
  reference: '2026-09-11T12:00:00Z',
  timeZone: 'Europe/Berlin',
  count: 5,
});

result.cron;        // '30 9 * * 1-5'
result.schedule;    // typed recurrence, timezone, and alternate-week anchor
result.occurrences; // future ISO instants, strictly after reference
result.description;
result.diagnostics;
result.model;       // predicted family, raw token spans/labels/scores, actual inference backend

const parser = defineParser();
try {
  const batch = await parser.parseMany(['every 15 minutes', 'daily at noon'], {
    timeZone: 'UTC', count: 0,
  });
} finally {
  parser.dispose();
}
```

Options default to the current instant, **UTC**, and five preview occurrences. `reference` must include `Z` or a UTC offset; timezone-less strings are rejected. Preview references currently support years 2000–2099, count is 0–100, expressions are at most 512 characters, and batches contain at most 4,096 expressions. Invalid API options throw; unsupported language returns diagnostics and a null schedule.

Model inference requires WebGPU. Unsupported browsers and device errors produce an error; there is no CPU or automatic fallback. Results expose the WebGPU execution provider and output location. Mechanical feature preparation, predicted-value decoding, cron compilation and calendar resolution remain CPU work.

Cron weights are embedded as packed signed six-bit values and expanded to float32 once during initialization. The runtime creates specialized WGSL pipelines for the fixed network, uploads weights once, and reuses input/intermediate/readback buffers, growing them when a larger batch arrives. Each instance queues calls to keep buffer reuse safe. No general model loader, operator library, WASM payload, or runtime dependency is shipped. See [MODEL_CARD.md](MODEL_CARD.md).

`pnpm run size` measures the complete standalone library, including packed weights, feature processing, decoding, and WGSL runtime. The complete library is **31.6 KiB** with Brotli compression. Demo HTML/CSS are separate. These are compressed download sizes, not GPU-memory sizes.

## Supported language

| Input | Cron / result |
| --- | --- |
| `every 15 minutes` | `*/15 * * * *` |
| `every 2 hours` | `0 */2 * * *` |
| `daily at midnight` | `0 0 * * *` |
| `every weekday at 9:30am` | `30 9 * * 1-5` |
| `on monday and wednesday at 09:15` | `15 9 * * 1,3` |
| `every friday at 4pm except in december` | `0 16 * 1-11 5` |
| `on the 1st and 15th of every month at noon` | `0 12 1,15 * *` |
| `every other friday at 4pm` | structured schedule + preview; cron is null |
| `on the last day of every month at noon` | structured schedule + preview; cron is null |

Variations include spoken clocks ("quarter to seven in the evening"), digital clocks, weekday/month names and ranges, ordinal month days, time-first phrases, task prefixes and month/weekday exclusions. The compiler consumes predicted semantic roles and validates their values. Multi-time schedules, holidays, business calendars, start/end clauses and embedded timezones remain outside the contract. The model can still reject valid requests or mislabel unsupported language; measured errors are listed in [MODEL_CARD.md](MODEL_CARD.md).

## Exactness and calendar policy

- Cron is **five fields**: minute, hour, day of month, month, weekday. It contains neither a timezone nor a command. Configure your scheduler's timezone separately to match `result.schedule.timeZone`.
- Minute/hour intervals align to the wall-clock field boundary. Intervals with included weekday restrictions are currently rejected rather than dropped. Only divisors of 60 or 24 are accepted; `every 7 minutes` is rejected because `*/7` resets each hour and would not preserve seven-minute spacing.
- Bare `at 4` means **04:00** and produces an explicit assumption diagnostic. Use `4pm` for 16:00.
- Alternate weeks use Monday-based local weeks anchored to the reference's week. Preserve `schedule.anchorWeek` to retain that phase; changing the reference on a new parse can change the phase.
- Month exclusions do not reset the alternate-week phase.
- Nonexistent local times in a DST gap are skipped. Repeated local times during a fold produce both instants. This matches simple wall-clock matching, but schedulers may have their own DST policy.
- Month days that do not exist (e.g. the 31st in February) are skipped. Previews search at most five years and report if the requested count cannot be found.
- Standard cron cannot represent alternate weeks, a last-day rule, or the intersection of restricted month-days and weekdays. Those schedules return `cron: null` with a `not-cron` diagnostic and keep their structured recurrence and preview. No approximate replacement is emitted.
- Successful parsing is not proof of the user's intent. Expose the description and preview before using a generated schedule.

## Data and evaluation

The 200 annotated requests in [`training/data/authored.jsonl`](training/data/authored.jsonl) were assistant-authored separately from the generator. They are development examples, not collected user traffic. Ambiguous and unsupported requests have explicit labels, with no guessed clock times.

```sh
mise exec -- pnpm run prepare:cron-data       # deterministic, meaning-disjoint splits
mise exec -- pnpm run evaluate:cron          # development only, actual WebGPU
mise exec -- pnpm run evaluate:cron:holdout  # evaluate the two historical holdout collections
```

Reports separate token/family accuracy, exact structured schedule accuracy, ambiguous/unsupported rejection, and compiler performance given correct annotated labels. Category breakdowns distinguish model failures from compiler limitations. The browser likewise displays **What the network predicted** separately from **Compiler result**. The named holdout collections now serve as development benchmarks because prior results informed this iteration; their requests are not fitted by the trainer. The regular check task evaluates all three collections against explicit regression floors in `training/quality.json`.

The six curated assistant paraphrases and their original training examples are stored together in `training/data/curated.jsonl`. No separate teacher was run.

## Train the tiny model

```sh
mise exec -- pnpm run prepare:cron-data    # generate inspectable JSONL files
mise run train:cron                       # train a candidate from those files
mise exec -- pnpm run evaluate:cron:candidate # WebGPU parity + semantic quality gate
mise exec -- pnpm run promote:cron        # recheck, then promote candidate artifacts
mise run check
```

Training uses locked uv dependencies, MLX and Apple Metal. Structured meanings are split before rendering, with authored meanings reserved. The generator writes annotated samples to `training/data/*.jsonl`. The trainer loads those files once, then samples families evenly from the same records each epoch; it does not generate sentences during fitting. Semantic roles identify quantities, clock values, offsets, weekdays and exclusions. Token/context dropout improves robustness, and the final 12 epochs apply six-bit fake quantization. Curated examples preserve their original meanings and provenance.

The candidate gate checks exact schedules and unsupported/ambiguous rejection against `training/quality.json`. Dataset hashes bind each floor to its collection. These gates prevent measured regressions; they do not establish generalization to new language.

The network sums learned 24-dimensional word-hash, consonant-hash and shape embeddings, runs bidirectional affine scans, and predicts 16 semantic roles plus seven family scores. It has no word-vocabulary lookup or shared unknown-word ID. All inference operators are specialized WGSL, independent of the training framework.

The recorded M2 Max run took 107.5 seconds for 45 epochs using the earlier pipeline. New runs use the fixed JSONL corpus. Each run starts from the same seed; the trainer has no checkpoint/resume protocol. Training writes a candidate that must pass the WebGPU, semantic-quality, and size checks before promotion.

**Known limits:** the shipped model produces 116/120 exact authored schedules and incorrectly accepts 9/80 unsupported or ambiguous requests. See [MODEL_CARD.md](MODEL_CARD.md) for the complete measurements and architecture.

Each JSONL line is one annotated object: `text`, `family`, canonical `groupId`, token offsets and semantic roles, structured `target`, and source/provenance. `clause` retains the original coarse annotation; `annotationVersion: 2` identifies semantic-role refinement. The generator assigns meanings to disjoint splits before rendering, so paraphrases stay together. `authored.jsonl` and `curated.jsonl` are versioned source data; the larger generated splits are reproducible and ignored.

Inspect or edit `training/data/train.jsonl` before training. Labels are read as stored, with token-offset, role, family, and split-leakage validation. `--data-dir PATH` selects another directory containing `train.jsonl`, `development.jsonl`, `patternHoldout.jsonl`, and `authored.jsonl`. Training snapshots the loaded records under `training/candidate/data/` so evaluation uses the same data. Checks and evaluation never regenerate these inputs. Rerunning `prepare:cron-data` intentionally replaces generated data, so keep any curated edits elsewhere first.

The trainer fits and exports; `evaluate.mjs` computes quality metrics using actual WebGPU inference. Checks print their summaries to the console without saving reports or screenshots. Historical training source hashes are preserved in the training report under `sourceMaintenance`; current source and manifest hashes identify the maintained pipeline without implying that unchanged weights were retrained. Generator version 13 removes unused templates; its training wording differs from the shipped model’s historical corpus, while evaluation examples and split meanings are unchanged.

## Source map

```text
src/                    Browser library
  index.ts              Public parser interface
  features.ts           Mechanical tokenization and feature hashing
  model/                Packed weights, decoding, and WebGPU inference
  compile.ts            Predicted values → calendar rules → cron
  calendar.ts           Timezone and DST-aware previews
  types.ts              Public result and option types
demo/                   Playground TypeScript and CSS
training/               Complete offline model pipeline
  data.py               Prepare and load annotated JSONL
  data/                 Source examples and split manifest
  train.py              Fit with MLX and export a candidate
  evaluate.mjs          Measure candidate or shipped model in WebGPU
  promote.mjs           Verify and copy a passing candidate
  quality.json          Semantic regression floors
  report.json           Shipped model provenance
test/                   Verification only; nothing ships in the library
  *.test.ts            Node parsing, calendar, and model checks
  browser.ts           Real WebGPU and demo checks
  reference.ts         Numerical reference for verification
  model-fixtures.json   Quantized MLX reference outputs
  size.ts              Complete browser-library size budget
```

`tsconfig.json` checks `src/`, `demo/`, and `test/`. `tsconfig.build.json` emits only `src/` into `dist/`; tests import that built output and its declarations. The test and type-check commands build it first, including on a fresh checkout.

The root `index.html` is Vite's conventional demo entry. Tool configuration and lockfiles stay beside their package manifests. `dist/`, `site/`, and `training/candidate/` are generated and ignored.

The cron contract follows the [crontab manual](https://man7.org/linux/man-pages/man5/crontab.5.html), particularly field-step resets and the OR relationship between restricted day fields. Browser inference uses WebGPU directly, verified against MLX fixtures.

MIT licensed. Inspired by the small-model approach in [gpu-time](https://github.com/arikchakma/gpu-time) and [gpu-lexer](https://github.com/vercel-labs/gpu-lexer); implementation and training data here are original.

## Dependency maintenance

Use `mise exec -- pnpm add <package>` for JavaScript dependencies and `mise exec -- uv add --project training <package>` for Python dependencies. Commit the corresponding lockfile with manifest changes. Tool versions belong in `mise.toml`; keep Node requirements, `packageManager` and `UV_PYTHON` consistent with those pins.

`mise run train:cron` writes packed weights, MLX fixtures, data identity, and a report to ignored `training/candidate/`. It never replaces shipped artifacts. `promote:cron` reruns actual WebGPU/MLX parity and exact-schedule/rejection evaluation and the complete-bundle size budget before copying the candidate into the project.
