import { build } from 'vite';
import { brotliCompressSync } from 'node:zlib';
import { mkdir, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';
const candidate = process.argv.includes('--candidate');
const budget = 40000;
const bundle = await build({
  configFile: false,
  plugins: candidate
    ? [
        {
          name: 'candidate-weights',
          enforce: 'pre',
          resolveId(source, importer) {
            if (source === './weights.json' && importer?.endsWith('/src/model/parameters.ts'))
              return resolve('training/candidate/weights.json');
          },
        },
      ]
    : [],
  logLevel: 'silent',
  build: {
    write: false,
    minify: true,
    lib: { entry: 'src/index.ts', formats: ['es'] },
    rollupOptions: {
      onwarn(warning, warn) {
        if (warning.code === 'UNRESOLVED_IMPORT') throw new Error(warning.message);
        warn(warning);
      },
    },
  },
});
const output = (Array.isArray(bundle) ? bundle : [bundle]).flatMap((r) => r.output);
const chunks = output.filter((f) => f.type === 'chunk');
if (
  output.some((f) => f.type === 'asset') ||
  chunks.some((c) => c.imports.length || c.dynamicImports.length)
)
  throw new Error(`cron must bundle standalone, including weights and runtime`);
const code = chunks.map((c) => c.code).join('\n');
if (/\.wasm\b/.test(code)) throw new Error(`cron unexpectedly includes a general runtime`);
const report = {
  minifiedBytes: Buffer.byteLength(code),
  brotliBytes: brotliCompressSync(code).length,
  budgetBytes: budget,
  runtimeDependencies: 0,
};
if (report.brotliBytes > budget)
  throw new Error(`cron exceeds its complete-bundle budget: ${JSON.stringify(report)}`);
const reports = { cron: report };
await mkdir('test-artifacts', { recursive: true });
await writeFile(
  candidate ? 'test-artifacts/candidate-bundle-sizes.json' : 'test-artifacts/bundle-sizes.json',
  JSON.stringify(reports, null, 2) + '\n',
);
console.log(
  JSON.stringify(
    {
      bundles: reports,
      note: 'Entire standalone library: API, feature preparation, WGSL runtime, packed model and model metadata. Minified then Brotli-compressed; no runtime or weights excluded. Demo HTML/CSS are separate.',
    },
    null,
    2,
  ),
);
