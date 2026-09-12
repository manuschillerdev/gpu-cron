import {checkRuntimeLifecycle} from './gpu-runtime-checks.mjs';
import assert from 'node:assert/strict';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { createServer } from 'vite';
import { chromium } from 'playwright';

const server = await createServer({ server: { host: '127.0.0.1', port: 4173, strictPort: false }, logLevel: 'error' });
let browser;
try {
  await server.listen();
  const url = server.resolvedUrls.local[0];
  browser = await chromium.launch({ channel: 'chromium', headless: true, args: ['--enable-unsafe-webgpu'] });
  const page = await browser.newPage({ viewport: { width: 1440, height: 1100 } });
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.goto(url);
  await page.waitForFunction(() => !document.querySelector('#copy').disabled);
  assert.deepEqual(await checkRuntimeLifecycle(page), {reused:true,grew:true,rejected:true,snapshot:true,queued:true});
  const fixtures = JSON.parse(await readFile('test/model-fixtures.json', 'utf8'));
  const parity = await page.evaluate(async fixtures => {
    const { CronModel } = await import('/src/model/runtime.ts');
    const { cpuLogits } = await import('/scripts/cron-reference.mjs');
    const { features, FEATURE_COUNT } = await import('/dist/features.js');
    const { defineParser } = await import('/src/index.ts');
    const runtime = await CronModel.create();
    const originalSubmit = GPUQueue.prototype.submit;
    let submissions = 0;
    GPUQueue.prototype.submit = function(...args) { submissions++; return originalSubmit.apply(this,args); };
    const input = new Float32Array(fixtures.length * FEATURE_COUNT);
    fixtures.forEach((fixture, i) => input.set(features(fixture.text), i * FEATURE_COUNT));
    const gpu = await runtime.logits(input);
    const cpu = cpuLogits(input);
    const cpuScores = cpu;
    const mlxLogits = fixtures.flatMap(fixture => fixture.logits);
    const mlxError = {
      webgpu: gpu.reduce((max,v,i) => Math.max(max,Math.abs(v - mlxLogits[i])),0),
      cpu: cpuScores.reduce((max,v,i) => Math.max(max,Math.abs(v - mlxLogits[i])),0),
    };
    const outputLocations = [...runtime.outputLocations];
    let maximumError = 0;
    for (let i = 0; i < gpu.length; i++) maximumError = Math.max(maximumError, Math.abs(gpu[i] - cpu[i]));
    // Repeated and concurrent dispatches must have separate buffers.
    const repeated = await Promise.all([runtime.logits(input), runtime.logits(input)]);
    const stable = repeated.every(output => output.every((value, i) => value === gpu[i]));
    runtime.dispose();
    GPUQueue.prototype.submit = originalSubmit;
    const cp = defineParser(); const gp = defineParser({ backend: 'webgpu' });
    const options = { reference: '2026-09-11T12:00:00Z', timeZone: 'Europe/Berlin', count: 3 };
    const texts = ['every 15 minutes', 'every weekday at 9:30am', 'every other friday at 4pm except december', 'on the last day of every month at noon', 'daily at 02:30', 'on the 1st of every month at noon except weekends', 'every day at noon until friday'];
    const cpuResults = await cp.parseMany(texts, options);
    const gpuResults = await gp.parseMany(texts, options);
    const usedGPU = gpuResults.every(result => result.model.backend === 'webgpu' && result.model.executionProvider === 'webgpu' && result.model.outputLocations[0] === 'gpu-buffer');
    for (const result of [...cpuResults, ...gpuResults]) delete result.model;
    cp.dispose(); gp.dispose();
    return { maximumError, mlxError, cpuError: cpuScores.reduce((max,v,i)=>Math.max(max,Math.abs(v-cpu[i])),0), outputLocations, submissions, stable, usedGPU, sameResults: JSON.stringify(cpuResults) === JSON.stringify(gpuResults), fixtures: fixtures.length };
  }, fixtures);
  assert.ok(parity.maximumError < 0.0001 && parity.cpuError < 0.0001, JSON.stringify(parity));
  assert.ok(parity.mlxError.webgpu < 0.0001 && parity.mlxError.cpu < 0.0001, JSON.stringify(parity));
  assert.deepEqual(parity.outputLocations, ['gpu-buffer']);
  assert.ok(parity.submissions > 0);
  assert.ok(parity.stable && parity.usedGPU && parity.sameResults, JSON.stringify(parity));
  await page.waitForFunction(() => document.querySelector('#status').textContent.startsWith('WEBGPU'));
  await page.click('[data-example^="Every other"]');
  await page.waitForFunction(() => document.querySelector('#diagnostics').textContent.includes('Five-field cron'));
  assert.equal(await page.locator('#occurrences li').count(), 5);
  assert.ok(await page.locator('#copy').isDisabled());
  await page.fill('#expression', 'every day at noon until friday');
  await page.waitForFunction(() => document.querySelector('.diagnostic.error'));
  assert.equal(await page.locator('#occurrences li').count(), 0);
  await page.click('[data-example="Every weekday at 9:30am"]');
  await page.waitForFunction(() => !document.querySelector('#copy').disabled);
  await mkdir('test-artifacts', { recursive: true });
  await page.screenshot({ path: 'test-artifacts/demo-desktop.png', fullPage: true });
  await page.setViewportSize({ width: 390, height: 844 });
  assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
  await page.screenshot({ path: 'test-artifacts/demo-mobile.png', fullPage: true });
  assert.deepEqual(errors, []);
  await writeFile('test-artifacts/browser-report.json', JSON.stringify(parity, null, 2) + '\n');
  console.log('Real Chromium WebGPU parity and desktop/mobile demo checks passed:', parity);
} finally { await browser?.close(); await server.close(); }
