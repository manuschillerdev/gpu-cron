import { test, mock } from 'node:test';
import assert from 'node:assert/strict';
import { parse, parseMany, defineParser } from '../dist/index.js';

// Existing grammar/calendar checks use a test-only numerical stand-in; browser checks execute WebGPU.
import {CronModel} from '../dist/model/runtime.js';
import {cpuLogits} from '../scripts/cron-reference.mjs';
mock.method(CronModel,'create',async()=>({executionProvider:'webgpu',outputLocations:['gpu-buffer'],logits:async input=>cpuLogits(input),dispose(){}}));
const context = { reference: '2026-09-11T12:00:00Z', timeZone: 'UTC', count: 3 };
const cases = [
  ['every 15 minutes', '*/15 * * * *', '2026-09-11T12:15:00.000Z'],
  ['every minute', '* * * * *', '2026-09-11T12:01:00.000Z'],
  ['every 2 hours', '0 */2 * * *', '2026-09-11T14:00:00.000Z'],
  ['hourly', '0 * * * *', '2026-09-11T13:00:00.000Z'],
  ['every day at 9am', '0 9 * * *', '2026-09-12T09:00:00.000Z'],
  ['daily at midnight', '0 0 * * *', '2026-09-12T00:00:00.000Z'],
  ['every night at 23:30', '30 23 * * *', '2026-09-11T23:30:00.000Z'],
  ['every weekday at 9:30am', '30 9 * * 1-5', '2026-09-14T09:30:00.000Z'],
  ['every weekend at noon', '0 12 * * 0,6', '2026-09-12T12:00:00.000Z'],
  ['every monday through friday at 4pm', '0 16 * * 1-5', '2026-09-11T16:00:00.000Z'],
  ['on monday and wednesday at 09:15', '15 9 * * 1,3', '2026-09-14T09:15:00.000Z'],
  ['every fri-sun at 5pm', '0 17 * * 0,5,6', '2026-09-11T17:00:00.000Z'],
  ['on the 1st and 15th of every month at noon', '0 12 1,15 * *', '2026-09-15T12:00:00.000Z'],
  ['every month on the 1st at 09:00', '0 9 1 * *', '2026-10-01T09:00:00.000Z'],
  ['on the first of every month at noon', '0 12 1 * *', '2026-10-01T12:00:00.000Z'],
  ['every friday at 4pm except in december', '0 16 * 1-11 5', '2026-09-11T16:00:00.000Z'],
  ['every day at noon except weekends', '0 12 * * 1-5', '2026-09-14T12:00:00.000Z'],
  ['please run every five minutes', '*/5 * * * *', '2026-09-11T12:05:00.000Z'],
  ['at 4pm every friday', '0 16 * * 5', '2026-09-11T16:00:00.000Z'],
];
for (const [input, cron, first] of cases) test(input, async () => {
  const result = await parse(input, context);
  assert.equal(result.cron, cron, JSON.stringify(result));
  assert.equal(result.occurrences[0], first);
  assert.equal(result.occurrences.length, 3);
  assert.equal(result.diagnostics.some(d => d.severity === 'error'), false);
});

test('alternate weeks preserve phase across a month exclusion', async () => {
  const result = await parse('every other Friday at 4pm except in December', { ...context, reference: '2026-11-20T12:00:00Z', count: 5 });
  assert.equal(result.cron, null);
  assert.equal(result.schedule.weekInterval, 2);
  assert.equal(result.schedule.anchorWeek, '2026-11-16');
  assert.deepEqual(result.occurrences, ['2026-11-20T16:00:00.000Z', '2027-01-01T16:00:00.000Z', '2027-01-15T16:00:00.000Z', '2027-01-29T16:00:00.000Z', '2027-02-12T16:00:00.000Z']);
  assert.ok(result.diagnostics.some(d => d.code === 'not-cron'));
});

test('last day is calendar-aware and not exported as approximate cron', async () => {
  const result = await parse('on the last day of every month at noon', { ...context, reference: '2028-01-01T00:00:00Z' });
  assert.equal(result.cron, null);
  assert.deepEqual(result.occurrences, ['2028-01-31T12:00:00.000Z', '2028-02-29T12:00:00.000Z', '2028-03-31T12:00:00.000Z']);
});

test('day-of-month plus weekday exclusions do not become cron OR semantics', async () => {
  const result = await parse('on the 1st of every month at noon except weekends', context);
  assert.ok(result.schedule);
  assert.equal(result.cron, null);
  assert.deepEqual(result.occurrences, ['2026-10-01T12:00:00.000Z', '2026-12-01T12:00:00.000Z', '2027-01-01T12:00:00.000Z']);
});

test('DST spring gap is skipped', async () => {
  const result = await parse('daily at 02:30', { reference: '2026-03-28T02:00:00Z', timeZone: 'Europe/Berlin', count: 2 });
  assert.deepEqual(result.occurrences, ['2026-03-30T00:30:00.000Z', '2026-03-31T00:30:00.000Z']);
});

test('DST autumn fold returns both instants in chronological order', async () => {
  const result = await parse('daily at 02:30', { reference: '2026-10-24T23:00:00Z', timeZone: 'Europe/Berlin', count: 3 });
  assert.deepEqual(result.occurrences, ['2026-10-25T00:30:00.000Z', '2026-10-25T01:30:00.000Z', '2026-10-26T01:30:00.000Z']);
});

test('non-whole-hour timezone is resolved exactly', async () => {
  const result = await parse('daily at 09:00', { ...context, timeZone: 'Asia/Kathmandu' });
  assert.equal(result.occurrences[0], '2026-09-12T03:15:00.000Z');
});

test('bare clocks are explicitly interpreted as 24-hour time', async () => {
  const result = await parse('every friday at 4', context);
  assert.equal(result.cron, '0 4 * * 5');
  assert.ok(result.diagnostics.some(d => d.code === 'assumption'));
});

for (const input of [
  '', 'hello world', 'every 0 minutes', 'every 7 minutes', 'every 25 hours', 'every day at 25:00',
  'every friday at 13pm', 'daily at 9:60', 'every day at 9am and 5pm',
  'every day at noon until friday', 'every day at noon except holidays',
  'every monday at noon except monday', 'every 15 minutes between 9am and 5pm',
  'on the 32nd of every month at noon', 'tomorrow at noon', 'every first monday of the month at noon',
  'every day at noon except january-december', 'every day at noon UTC',
]) test(`rejects without guessing: ${input}`, async () => {
  const result = await parse(input, context);
  assert.equal(result.schedule, null, JSON.stringify(result));
  assert.equal(result.cron, null);
  assert.deepEqual(result.occurrences, []);
  assert.ok(result.diagnostics.some(d => d.severity === 'error'));
});

test('options are validated, empty batches are supported, and preview can be disabled', async () => {
  await assert.rejects(parse('daily at noon', { timeZone: 'Not/AZone' }), RangeError);
  await assert.rejects(parse('daily at noon', { reference: '2026-09-11' }), RangeError);
  await assert.rejects(parse('daily at noon', { count: -1 }), RangeError);
  await assert.rejects(parse('x'.repeat(513)), RangeError);
  await assert.rejects(parseMany([42]), TypeError);
  assert.deepEqual(await parseMany([]), []);
  assert.deepEqual((await parse('daily at noon', { ...context, count: 0 })).occurrences, []);
});

test('batch and individual parsing agree; disposed instances reject calls', async () => {
  const parser = defineParser({ backend: 'webgpu' });
  const inputs = cases.map(([text]) => text);
  assert.deepEqual(await parser.parseMany(inputs, context), await Promise.all(inputs.map(text => parser.parse(text, context))));
  parser.dispose();
  await assert.rejects(parser.parse('daily at noon'), /disposed/);
});

test('explicit WebGPU does not silently fall back in Node', async () => {
  mock.restoreAll();
  const parser = defineParser({ backend: 'webgpu' });
  await assert.rejects(parser.parse('daily at noon'), /WebGPU/);
  parser.dispose();
});
