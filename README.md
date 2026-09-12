# gpu-cron

Turn English recurring schedules into structured calendar rules, five-field cron, and timezone-aware previews. A 35,783-parameter MLX-trained sequence model reads context in both directions and predicts semantic roles for clock values, weekdays and exclusions on WebGPU. TypeScript interprets those predicted values and resolves the exact calendar. There is no CPU inference fallback.

The browser package embeds packed learned weights and a purpose-built WGSL runtime with zero runtime dependencies. MLX trains the model; ONNX exports are development-only verification artifacts.

## Run locally

**mise** pins Node.js 26.1.0, Python 3.12.14, pnpm 10.32.1, and uv 0.12.5. **pnpm** manages JavaScript dependencies; **uv** manages Python dependencies and the training environment. Both lockfiles are committed.

```sh
mise trust
mise install
mise run setup
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

Use `mise run check` for the complete suite or `mise run build` for the library and demo. With mise activated in your shell, the pnpm/uv commands below use the pinned tools; otherwise prefix them with `mise exec --`. `mise run setup` installs from both lockfiles without updating them. Python selection is pinned in `.python-version` and the mise `UV_PYTHON` environment setting.

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

const parser = defineParser({ backend: 'webgpu' });
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

Cron weights are embedded as packed signed six-bit values and expanded to float32 once during initialization. The runtime creates specialized WGSL pipelines for the fixed network, uploads weights once, and reuses input/intermediate/readback buffers, growing them when a larger batch arrives. Each instance queues calls to keep buffer reuse safe. No general model loader, operator library, WASM payload, or runtime dependency is shipped. See [architecture.md](architecture.md).

`pnpm run size` measures the complete standalone library, including packed weights, feature processing, decoding, and WGSL runtime. The complete library is **30.9 KiB** with Brotli compression. Demo HTML/CSS are separate. These are compressed download sizes, not GPU-memory sizes.

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
- Minute/hour intervals align to the wall-clock field boundary. Only divisors of 60 or 24 are accepted; `every 7 minutes` is rejected because `*/7` resets each hour and would not preserve seven-minute spacing.
- Bare `at 4` means **04:00** and produces an explicit assumption diagnostic. Use `4pm` for 16:00.
- Alternate weeks use Monday-based local weeks anchored to the reference's week. Preserve `schedule.anchorWeek` to retain that phase; changing the reference on a new parse can change the phase.
- Month exclusions do not reset the alternate-week phase.
- Nonexistent local times in a DST gap are skipped. Repeated local times during a fold produce both instants. This matches simple wall-clock matching, but schedulers may have their own DST policy.
- Month days that do not exist (e.g. the 31st in February) are skipped. Previews search at most five years and report if the requested count cannot be found.
- Standard cron cannot represent alternate weeks, a last-day rule, or the intersection of restricted month-days and weekdays. Those schedules return `cron: null` with a `not-cron` diagnostic and keep their structured recurrence and preview. No approximate replacement is emitted.
- Successful parsing is not proof of the user's intent. Expose the description and preview before using a generated schedule.

## Data and evaluation

The 200 annotated requests live in [`training/data/authored.jsonl`](training/data/authored.jsonl). They were composed separately from the renderer in this session; no review gate is required. They are assistant-authored, not collected user traffic. Ambiguous and unsupported requests have explicit labels, with no guessed clock times.

```sh
mise exec -- pnpm run prepare:cron-data       # deterministic, meaning-disjoint splits
mise exec -- pnpm run evaluate:cron          # development only, actual WebGPU
mise exec -- pnpm run evaluate:cron:holdout  # evaluate the two historical holdout collections
```

Reports separate token/family accuracy, exact structured schedule accuracy, ambiguous/unsupported rejection, and compiler performance given correct annotated labels. Category breakdowns distinguish model failures from compiler limitations. The browser likewise displays **What the network predicted** separately from **Compiler result**. The named holdout collections now serve as development benchmarks because prior results informed this iteration; their requests are not fitted by the trainer or run in the regular check task.

Offline paraphrase prompts and annotated imports use `training/paraphrases.py`; see [the protocol](training/data/README.md). No teacher model is shipped in the browser. The current six paraphrases were authored in this session, without running a separate local teacher.

## Train the tiny model

```sh
mise run train:cron          # train with MLX, then export ONNX
mise run benchmark:training  # synchronized compiled MLX GPU epochs
mise run check
```

Training uses locked uv dependencies, MLX and Apple Metal. Structured meanings are split before rendering, with authored meanings reserved. Every epoch generates fresh composable phrasings; families are sampled evenly. Semantic roles identify quantities, clock values, offsets, weekdays and exclusions. Token/context dropout improves robustness, and the final 12 epochs apply six-bit fake quantization. Six offline assistant paraphrases retain frozen training parents. No teacher is shipped in the browser.

The network sums learned 24-dimensional word-hash, consonant-hash and shape embeddings, runs bidirectional affine scans, and predicts 16 semantic roles plus seven family scores. It has no word-vocabulary lookup or shared unknown-word ID. All inference operators are specialized WGSL, independent of the training framework. `src/model/vocabulary.json` is only a training-word audit artifact.

The recorded M2 Max run took 107.5 seconds for 45 epochs, regenerating 23,177 training records per epoch. Checkpoints preserve model, optimizer, RNG and epoch identity. Resume using `mise exec -- uv run --locked --project training python training/train.py --resume`. The subsequent ONNX export verifies the exact shipped coefficients; it is not imported by the browser. `mise exec -- pnpm run export:onnx` exports existing weights without retraining.

Actual WebGPU evaluation now produces **116/120 exact authored schedules**, versus the original 23/120. The compiler produces 120/120 with annotated roles, versus 29/120. **9/80 unsupported or ambiguous authored requests are still incorrectly accepted.** These are development results on assistant-authored requests, not broad real-user accuracy. Old token scores are not directly comparable because the label vocabulary changed. See [MODEL_CARD.md](MODEL_CARD.md) and [the data protocol](training/data/README.md).

## Source map

- `src/features.ts`: mechanical tokenization and packed word-hash, consonant-hash and shape features.
- `src/model/`: packed model, specialized GPU runtime, and weight decoding.
- `models/`: optional verification ONNX graph and export provenance.
- `src/compile.ts`: semantic-value decoding, typed recurrence, descriptions and cron export.
- `src/calendar.ts`: local calendar matching and DST-aware future instants.
- `training/`: generator, trainer, uv lockfile, and measured provenance.
- `test/`: semantic examples, rejection cases, DST policy, and MLX parity.
- `demo/`: browser playground.

The cron contract follows the [crontab manual](https://man7.org/linux/man-pages/man5/crontab.5.html), particularly field-step resets and the OR relationship between restricted day fields. Browser inference uses WebGPU directly; ONNX Runtime is a development-only numerical reference.

MIT licensed. Inspired by the small-model approach in [gpu-time](https://github.com/arikchakma/gpu-time) and [gpu-lexer](https://github.com/vercel-labs/gpu-lexer); implementation and training data here are original.

## Dependency maintenance

Use `mise exec -- pnpm add <package>` for JavaScript dependencies and `mise exec -- uv add --project training <package>` for Python dependencies. Commit the corresponding lockfile with manifest changes. Tool versions belong in `mise.toml`; keep Node requirements, `packageManager`, `.python-version` and `UV_PYTHON` consistent with those pins.

`mise run train:cron` trains with MLX on Apple Metal and exports packed weights plus the ONNX verification graph.
