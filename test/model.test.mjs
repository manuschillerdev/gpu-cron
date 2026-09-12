import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { features } from '../dist/features.js';
import { MODEL_INFO } from '../dist/model/parameters.js';
import { cpuLogits } from './reference.mjs';

test('JavaScript features and quantized inference match the MLX export', async () => {
  const fixtures = JSON.parse(await readFile(new URL('./model-fixtures.json', import.meta.url)));
  let maximumError = 0;
  for (const fixture of fixtures) {
    const input = features(fixture.text);
    assert.deepEqual(Array.from(input), fixture.features, fixture.text);
    const logits = cpuLogits(input);
    for (let i = 0; i < logits.length; i++) maximumError = Math.max(maximumError, Math.abs(logits[i] - fixture.logits[i]));
  }
  assert.ok(maximumError < 0.0001, `maximum logit error: ${maximumError}`);
});

test('the shipped model matches its provenance report', async () => {
  const report = JSON.parse(await readFile(new URL('../training/report.json', import.meta.url)));
  const weights = await readFile(new URL('../src/model/weights.json', import.meta.url));
  const training = await readFile(new URL('../training/train.py', import.meta.url));
  assert.equal(createHash('sha256').update(weights).digest('hex'), report.weightsSha256);
  assert.equal(createHash('sha256').update(training).digest('hex'), report.trainingSourceSha256);
  assert.equal(MODEL_INFO.parameters, report.parameters);
  assert.equal(report.framework, 'mlx');
  assert.equal(report.device, 'gpu');
  for (const [path, hash] of [['../training/data.py',report.dataSourceSha256],['../training/data/manifest.json',report.manifestSha256]]) {
    assert.equal(createHash('sha256').update(await readFile(new URL(path,import.meta.url))).digest('hex'),hash);
  }
  assert.equal(createHash('sha256').update(await readFile(new URL('./model-fixtures.json', import.meta.url))).digest('hex'), report.fixturesSha256);
});
