/** Promote only a candidate that passes current WebGPU parity and semantic floors. */
import { spawnSync } from 'node:child_process';
import { copyFile } from 'node:fs/promises';

for (const args of [
  ['scripts/evaluate-cron.mjs', '--candidate', '--check'],
  ['scripts/size.mjs', '--candidate'],
]) {
  const result = spawnSync(process.execPath, args, { stdio: 'inherit' });
  if (result.error) throw result.error;
  if (result.status !== 0) process.exit(result.status ?? 1);
}

for (const [source, destination] of [
  ['weights.json', 'src/model/weights.json'],
  ['vocabulary.json', 'src/model/vocabulary.json'],
  ['model-fixtures.json', 'test/model-fixtures.json'],
  ['data/manifest.json', 'training/data/manifest.json'],
  ['report.json', 'training/report.json'],
  ...['train', 'development', 'patternHoldout', 'authored'].map((name) => [
    `data/${name}.jsonl`,
    `training/data/${name}.jsonl`,
  ]),
  ...['development', 'patternHoldout', 'authoredHoldout'].map((name) => [
    `evaluation/${name}.json`,
    `training/evaluation/${name}.json`,
  ]),
]) {
  await copyFile(`training/candidate/${source}`, destination);
}
console.log(
  'Promoted candidate artifacts. Run pnpm run export:onnx to refresh the optional verification graph, then mise run check before committing.',
);
