import { FEATURE_COUNT, features } from './features.js';
import { prediction, OUTPUT_COUNT } from './model/parameters.js';
import { CronModel } from './model/runtime.js';
import { context, preview } from './calendar.js';
import { compile, cron, describe } from './compile.js';
import type { ParseOptions, ParseResult, Parser } from './types.js';

export type { Diagnostic, Family, ParseOptions, ParseResult, Parser, Schedule } from './types.js';
export { MODEL_INFO } from './model/parameters.js';

function emptyResult(text: string, model: ParseResult['model']): ParseResult {
  return {
    input: text,
    schedule: null,
    cron: null,
    description: null,
    occurrences: [],
    diagnostics: [],
    model,
  };
}

function resultFromPrediction(
  text: string,
  logits: Float32Array,
  runtime: CronModel,
  options: ReturnType<typeof context>,
): ParseResult {
  const model: ParseResult['model'] = {
    ...prediction(logits, text),
    backend: 'webgpu',
    executionProvider: runtime.executionProvider,
    outputLocations: [...runtime.outputLocations],
  };
  const result = emptyResult(text, model);

  if (!text.trim()) {
    result.diagnostics.push({
      code: 'invalid-input',
      severity: 'error',
      message: 'Enter a recurring schedule.',
    });
    return result;
  }

  const compiled = compile(text, model.family, options.timeZone, options.anchorWeek, model.tokens);
  if (!compiled.schedule) {
    result.diagnostics = compiled.diagnostics;
    return result;
  }

  result.schedule = compiled.schedule;
  result.description = describe(compiled.schedule);
  result.occurrences = preview(compiled.schedule, options.reference, options.count);
  result.diagnostics = compiled.diagnostics;

  const exported = cron(compiled.schedule);
  result.cron = exported.expression;
  if (exported.reason)
    result.diagnostics.push({ code: 'not-cron', severity: 'warning', message: exported.reason });
  if (result.occurrences.length < options.count)
    result.diagnostics.push({
      code: 'preview-limit',
      severity: 'warning',
      message: 'Fewer occurrences were found within the five-year preview horizon.',
    });
  return result;
}

export function defineParser(): Parser {
  let disposed = false;
  let gpu: Promise<CronModel> | undefined;
  async function parseMany(
    texts: readonly string[],
    options: ParseOptions = {},
  ): Promise<ParseResult[]> {
    if (disposed) throw new Error('Parser is disposed.');
    if (!Array.isArray(texts) || texts.length > 4096)
      throw new RangeError('Supply an array of at most 4096 expressions.');
    const ctx = context(options);
    if (!texts.length) return [];
    for (const text of texts) {
      if (typeof text !== 'string') throw new TypeError('Every input must be a string.');
      if (text.length > 512)
        throw new RangeError('Each expression must be at most 512 characters.');
    }
    const input = new Float32Array(texts.length * FEATURE_COUNT);
    texts.forEach((text, i) => input.set(features(text), i * FEATURE_COUNT));
    gpu ??= CronModel.create();
    const runtime = await gpu;
    if (disposed) {
      runtime.dispose();
      throw new Error('Parser is disposed.');
    }
    const logits = await runtime.logits(input);
    if (disposed) throw new Error('Parser is disposed.');
    return texts.map((text, index) => {
      const start = index * OUTPUT_COUNT;
      return resultFromPrediction(text, logits.subarray(start, start + OUTPUT_COUNT), runtime, ctx);
    });
  }
  return {
    parse: async (text, options) => (await parseMany([text], options))[0]!,
    parseMany,
    dispose() {
      disposed = true;
      void gpu?.then(
        (runtime) => runtime.dispose(),
        () => {},
      );
    },
  };
}

const defaultParser = defineParser();
export const parse: Parser['parse'] = (text, options) => defaultParser.parse(text, options);
export const parseMany: Parser['parseMany'] = (texts, options) =>
  defaultParser.parseMany(texts, options);
