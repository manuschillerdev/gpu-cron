/** Dataset evaluation, separate from regression tests. Inference is actual WebGPU. */
import { readFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { resolve } from 'node:path';
import { createServer } from 'vite';
import { chromium } from 'playwright';

const candidate = process.argv.includes('--candidate');
const check = process.argv.includes('--check');
const names = check
  ? ['development', 'patternHoldout', 'authoredHoldout']
  : process.argv.includes('--holdout')
    ? ['patternHoldout', 'authoredHoldout']
    : ['development'];
const directory = candidate ? 'training/candidate' : 'training';
const dataDirectory = directory + '/data';
const weightsPath = candidate ? directory + '/weights.json' : 'src/model/weights.json';
const quality = JSON.parse(await readFile('training/quality.json', 'utf8'));
const failures = [];
const server = await createServer({
  plugins: candidate
    ? [
        {
          name: 'candidate-weights',
          enforce: 'pre',
          resolveId(source, importer) {
            if (source === './weights.json' && importer?.endsWith('/src/model/parameters.ts'))
              return resolve(weightsPath);
          },
        },
      ]
    : [],
  server: { host: '127.0.0.1', port: 4190, strictPort: false, hmr: false, watch: null },
  logLevel: 'error',
});
let browser;
const sha = (bytes) => createHash('sha256').update(bytes).digest('hex');
try {
  const training = JSON.parse(await readFile(directory + '/report.json', 'utf8'));
  if (
    sha(await readFile(dataDirectory + '/manifest.json')) !== training.manifestSha256 ||
    sha(await readFile(weightsPath)) !== training.weightsSha256
  )
    throw new Error(
      'Dataset/model identity differs from the recorded training run. Retrain or restore the matching artifacts before evaluation.',
    );
  if (candidate) {
    for (const [path, expected] of [
      ['training/train.py', training.trainingSourceSha256],
      ['training/data.py', training.dataSourceSha256],
      [directory + '/model-fixtures.json', training.fixturesSha256],
    ]) {
      if (sha(await readFile(path)) !== expected)
        throw new Error('Candidate provenance differs: ' + path);
    }
  }
  await server.listen();
  browser = await chromium.launch({
    channel: 'chromium',
    headless: true,
    args: ['--enable-unsafe-webgpu'],
  });
  const page = await browser.newPage();
  await page.goto(server.resolvedUrls.local[0]);
  await page.waitForFunction(() =>
    document.querySelector('#status').textContent.startsWith('WEBGPU'),
  );
  // Candidate fixtures were computed in MLX after quantization. Verify the actual
  // candidate GPU graph before considering any semantic quality measurements.
  if (candidate) {
    const fixtures = JSON.parse(await readFile(directory + '/model-fixtures.json', 'utf8'));
    if (!fixtures.length) throw new Error('Candidate has no MLX verification fixtures');
    const error = await page.evaluate(async (fixtures) => {
      const { CronModel } = await import('/src/model/runtime.ts');
      const { features, FEATURE_COUNT } = await import('/src/features.ts');
      const runtime = await CronModel.create();
      try {
        const input = new Float32Array(fixtures.length * FEATURE_COUNT);
        fixtures.forEach((f, i) => {
          const encoded = features(f.text);
          if (JSON.stringify(Array.from(encoded)) !== JSON.stringify(f.features))
            throw new Error('MLX feature mismatch');
          input.set(encoded, i * FEATURE_COUNT);
        });
        const output = await runtime.logits(input);
        const expected = fixtures.flatMap((f) => f.logits);
        if (output.length !== expected.length) throw new Error('MLX output shape mismatch');
        return output.reduce(
          (error, value, i) => Math.max(error, Math.abs(value - expected[i])),
          0,
        );
      } finally {
        runtime.dispose();
      }
    }, fixtures);
    if (!Number.isFinite(error) || error >= 0.0001)
      throw new Error('Candidate WebGPU/MLX parity failed: ' + error);
  }
  const reports = {};
  for (const name of names) {
    const file = `${dataDirectory}/${name === 'authoredHoldout' ? 'authored' : name}.jsonl`;
    const bytes = await readFile(file);
    const records = bytes.toString().trim().split('\n').filter(Boolean).map(JSON.parse);
    const outcomes = await page.evaluate(async (records) => {
      const { defineParser } = await import('/src/index.ts');
      const { compile } = await import('/src/compile.ts');
      const { tokenize } = await import('/src/features.ts');
      const parser = defineParser();
      const results = [];
      const canonical = (s) =>
        s
          ? Object.fromEntries(
              ['minutes', 'hours', 'daysOfMonth', 'weekdays', 'months', 'weekInterval'].map((k) => [
                k,
                Array.isArray(s[k]) ? [...s[k]].sort((a, b) => a - b) : s[k],
              ]),
            )
          : null;
      const equal = (a, b) => JSON.stringify(canonical(a)) === JSON.stringify(canonical(b));
      try {
        for (let start = 0; start < records.length; start += 128) {
          const batch = records.slice(start, start + 128);
          const predictions = await parser.parseMany(
            batch.map((r) => r.text),
            { timeZone: 'UTC', reference: '2026-09-14T00:00:00Z', count: 0 },
          );
          predictions.forEach((pred, i) => {
            const gold = batch[i];
            const actualTokens = tokenize(gold.text);
            if (
              actualTokens.length !== gold.tokens.length ||
              actualTokens.some(
                (t, j) => t.start !== gold.tokens[j].start || t.end !== gold.tokens[j].end,
              )
            )
              throw new Error(`Annotation/token alignment mismatch: ${gold.id}`);
            const goldTokens = gold.tokens.map((t) => ({
              ...t,
              text: gold.text.slice(t.start, t.end),
              confidence: 1,
            }));
            const oracle = compile(gold.text, gold.family, 'UTC', '2026-09-14', goldTokens);
            const tokenCorrect = gold.tokens.filter(
              (t, j) => pred.model.tokens[j]?.role === t.role,
            ).length;
            const familyCorrect = pred.model.family === gold.family;
            const spansCorrect =
              tokenCorrect === gold.tokens.length &&
              pred.model.tokens.length === gold.tokens.length;
            const exact = equal(pred.schedule, gold.target.schedule);
            const oracleExact = equal(oracle.schedule, gold.target.schedule);
            const failure = exact
              ? null
              : gold.target.status !== 'supported'
                ? 'accepted-ambiguous-or-unsupported'
                : oracleExact
                  ? 'model'
                  : familyCorrect && spansCorrect
                    ? 'compiler'
                    : 'model-and-compiler';
            results.push({
              groupId: gold.groupId,
              status: gold.target.status,
              network: {
                familyCorrect,
                spansCorrect,
                tokenCorrect,
                tokenTotal: gold.tokens.length,
              },
              compiler: {
                exact,
                accepted: pred.schedule !== null,
              },
              oracleCompiler: {
                exact: oracleExact,
              },
              failure,
            });
          });
        }
      } finally {
        parser.dispose();
      }
      return results;
    }, records);
    const summarize = (rows) => {
      const count = rows.length;
      const ratio = (n) => (count ? n / count : null);
      const supported = rows.filter((r) => r.status === 'supported');
      const ambiguous = rows.filter((r) => r.status === 'ambiguous');
      const unsupported = rows.filter((r) => r.status === 'unsupported');
      const tokenTotal = rows.reduce((sum, r) => sum + r.network.tokenTotal, 0);
      return {
        examples: count,
        meaningGroups: new Set(rows.map((r) => r.groupId)).size,
        network: {
          familyAccuracy: ratio(rows.filter((r) => r.network.familyCorrect).length),
          exactSpanAccuracy: ratio(rows.filter((r) => r.network.spansCorrect).length),
          tokenAccuracy: tokenTotal
            ? rows.reduce((s, r) => s + r.network.tokenCorrect, 0) / tokenTotal
            : null,
        },
        compiler: {
          exactStructuredScheduleAccuracy: supported.length
            ? supported.filter((r) => r.compiler.exact).length / supported.length
            : null,
          supportedExamples: supported.length,
          correctOutcomeRate: ratio(rows.filter((r) => r.compiler.exact).length),
          unsupportedExamples: unsupported.length,
          unsupportedRejectionRate: unsupported.length
            ? unsupported.filter((r) => !r.compiler.accepted).length / unsupported.length
            : null,
          ambiguousExamples: ambiguous.length,
          ambiguousRejectionRate: ambiguous.length
            ? ambiguous.filter((r) => !r.compiler.accepted).length / ambiguous.length
            : null,
        },
        oracleCompiler: {
          exactStructuredScheduleAccuracy: supported.length
            ? supported.filter((r) => r.oracleCompiler.exact).length / supported.length
            : null,
        },
        errors: Object.fromEntries(
          ['model', 'compiler', 'model-and-compiler', 'accepted-ambiguous-or-unsupported'].map(
            (k) => [k, rows.filter((r) => r.failure === k).length],
          ),
        ),
      };
    };
    const summary = summarize(outcomes);
    if (check) {
      const floor = quality.splits[name];
      const exact = outcomes.filter((r) => r.status === 'supported' && r.compiler.exact).length;
      const accepted = (status) =>
        outcomes.filter((r) => r.status === status && r.compiler.accepted).length;
      if (sha(bytes) !== floor.sourceSha256 || outcomes.length !== floor.examples)
        failures.push(
          name + ': evaluation collection changed; review its quality floor explicitly',
        );
      if (exact < floor.minExactSchedules)
        failures.push(name + ': exact schedules ' + exact + ' < ' + floor.minExactSchedules);
      if (accepted('unsupported') > floor.maxUnsupportedAccepted)
        failures.push(name + ': unsupported acceptance regressed');
      if (accepted('ambiguous') > floor.maxAmbiguousAccepted)
        failures.push(name + ': ambiguous acceptance regressed');
    }
    reports[name] = summary;
  }
  console.log(JSON.stringify(reports, null, 2));
  if (failures.length) throw new Error('Semantic quality gate failed:\n' + failures.join('\n'));
  if (check)
    console.log(
      'Semantic quality gate passed. These are development regression floors, not generalization guarantees.',
    );
} finally {
  await browser?.close();
  await server.close();
}
