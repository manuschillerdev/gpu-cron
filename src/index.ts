import { FEATURE_COUNT, features } from './features.js';
import { prediction, OUTPUT_COUNT } from './model/parameters.js';
import { CronModel } from './model/runtime.js';
import { context, preview } from './calendar.js';
import { compile, cron, describe } from './compile.js';
import type { Backend, ParseOptions, ParseResult, Parser } from './types.js';

export type { Backend, Diagnostic, Family, ParseOptions, ParseResult, Parser, Schedule } from './types.js';
export { MODEL_INFO } from './model/parameters.js';

export function defineParser(options: { backend?: Backend } = {}): Parser {
  const requested = options.backend ?? 'webgpu';
  if (requested !== 'webgpu') throw new RangeError('WebGPU is the only supported backend.');
  let disposed = false;
  let gpu: Promise<CronModel> | undefined;
  async function parseMany(texts: readonly string[], options: ParseOptions = {}): Promise<ParseResult[]> {
    if (disposed) throw new Error('Parser is disposed.');
    if (!Array.isArray(texts) || texts.length > 4096) throw new RangeError('Supply an array of at most 4096 expressions.');
    const ctx = context(options);
    if (!texts.length) return [];
    for (const text of texts) {
      if (typeof text !== 'string') throw new TypeError('Every input must be a string.');
      if (text.length > 512) throw new RangeError('Each expression must be at most 512 characters.');
    }
    const input = new Float32Array(texts.length * FEATURE_COUNT);
    texts.forEach((text, i) => input.set(features(text), i * FEATURE_COUNT));
    const backend = 'webgpu' as const;
    gpu ??= CronModel.create();
    const runtime = await gpu;
    if (disposed) { runtime.dispose(); throw new Error('Parser is disposed.'); }
    const logits = await runtime.logits(input);
    if (disposed) throw new Error('Parser is disposed.');
    return texts.map((text, i) => {
      const model = { ...prediction(logits.subarray(i * OUTPUT_COUNT, (i + 1) * OUTPUT_COUNT), text), backend, executionProvider: runtime.executionProvider, outputLocations: [...runtime.outputLocations] };
      const empty: ParseResult = { input: text, schedule: null, cron: null, description: null, occurrences: [], diagnostics: [], model };
      if (!text.trim()) return { ...empty, diagnostics: [{ code: 'invalid-input', severity: 'error', message: 'Enter a recurring schedule.' }] };
      const compiled = compile(text, model.family, ctx.timeZone, ctx.anchorWeek, model.tokens);
      if (!compiled.schedule) return { ...empty, diagnostics: compiled.diagnostics };
      const schedule = compiled.schedule;
      const exported = cron(schedule);
      const occurrences = preview(schedule, ctx.reference, ctx.count);
      const diagnostics = [...compiled.diagnostics];
      if (exported.reason) diagnostics.push({ code: 'not-cron', severity: 'warning', message: exported.reason });
      if (occurrences.length < ctx.count) diagnostics.push({ code: 'preview-limit', severity: 'warning', message: 'Fewer occurrences were found within the five-year preview horizon.' });
      return { ...empty, schedule, cron: exported.expression, description: describe(schedule), occurrences, diagnostics };
    });
  }
  return {
    parse: async (text, options) => (await parseMany([text], options))[0]!,
    parseMany,
    dispose() { disposed = true; void gpu?.then(runtime => runtime.dispose(), () => {}); },
  };
}

const defaultParser = defineParser();
export const parse: Parser['parse'] = (text, options) => defaultParser.parse(text, options);
export const parseMany: Parser['parseMany'] = (texts, options) => defaultParser.parseMany(texts, options);
