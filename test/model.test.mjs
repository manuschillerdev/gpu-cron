import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { features, FEATURE_COUNT } from '../dist/features.js';
import { MODEL_INFO } from '../dist/model/parameters.js';
import { cpuLogits } from '../scripts/cron-reference.mjs';

test('packed cron scores match the optional ONNX verification graph', async () => {
  const ort = await import('onnxruntime-web');
  ort.env.wasm.numThreads = 1;
  const session = await ort.InferenceSession.create(await readFile(new URL('../models/cron.onnx', import.meta.url)), {executionProviders:['wasm']});
  try {
    const fixtures = JSON.parse(await readFile(new URL('./model-fixtures.json', import.meta.url)));
    const input = new Float32Array(fixtures.length * FEATURE_COUNT);
    fixtures.forEach((f,i) => input.set(features(f.text),i*FEATURE_COUNT));
    const tensor = new ort.Tensor('float32',input,[fixtures.length,FEATURE_COUNT]);
    const outputs = await session.run({features:tensor});
    try {
      const reference = cpuLogits(input);
      assert.ok(outputs.logits.data.reduce((max,v,i)=>Math.max(max,Math.abs(v-reference[i])),0) < .0001);
    } finally { tensor.dispose(); Object.values(outputs).forEach(t=>t.dispose()); }
  } finally { await session.release(); }
});

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
  for (const [path, hash] of [['../training/data.py',report.dataSourceSha256],['../src/model/vocabulary.json',report.vocabularySha256],['../training/data/manifest.json',report.manifestSha256]]) {
    assert.equal(createHash('sha256').update(await readFile(new URL(path,import.meta.url))).digest('hex'),hash);
  }
  assert.equal(createHash('sha256').update(await readFile(new URL('./model-fixtures.json', import.meta.url))).digest('hex'), report.fixturesSha256);
});

test('verification ONNX artifact and source weights match export provenance', async () => {
  const report = JSON.parse(await readFile(new URL('../models/cron-export.json', import.meta.url)));
  for (const [path, expected] of [
    ['../models/cron.onnx', report.modelSha256],
    ['../src/model/weights.json', report.sourceWeightsSha256],
    ['../training/export_onnx.py', report.exportSourceSha256],
  ]) assert.equal(createHash('sha256').update(await readFile(new URL(path, import.meta.url))).digest('hex'), expected);
  assert.equal(report.onnxCheckerPassed, true);
});
