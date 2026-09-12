export const FEATURE_COUNT = 512;
export const HIDDEN_COUNT = 24;
export const ROLES = ['prefix', 'recurrence', 'quantity', 'hour', 'minute', 'meridiem', 'clock-offset', 'clock-direction', 'weekday', 'range', 'monthday', 'month', 'excluded-weekday', 'excluded-month', 'exclusion', 'unknown'] as const;
export interface Token { text: string; start: number; end: number }
export function tokenize(text: string): Token[] {
  return Array.from(text.matchAll(/[A-Za-z]+|[0-9]+|[^\s]/gu), m => ({text:m[0], start:m.index, end:m.index+m[0].length}));
}
export const FAMILIES = ['minutes', 'hours', 'daily', 'weekly', 'biweekly', 'monthly', 'invalid'] as const;

export const DAY_NAMES = ['sunday', 'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday'];
export const MONTH_NAMES = ['january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october', 'november', 'december'];
/** Shared with training/train.py; cross-language parity is checked in tests. */
export function features(text: string): Float32Array {
  const tokens = tokenize(text);
  if (tokens.length > FEATURE_COUNT) throw new RangeError('Sequence exceeds token capacity.');
  const result = new Float32Array(FEATURE_COUNT);
  for (let i = 0; i < tokens.length; i++) {
    const text = tokens[i]!.text, word = text.toLowerCase();
    const kind = /^\d+$/.test(word) ? 1 : /^[a-z]+$/.test(word) ? 0 : 2;
    const shape = kind | (Math.min(Math.floor(word.length / 3), 3) << 2) | (Number(text !== word) << 4);
    result[i] = 1 + (hash(word) & 1023) + ((hash(word.replace(/[aeiou]/g, '')) & 255) << 10) + (shape << 18);
  }
  return result;
}

function hash(text: string): number {
  let value = 2166136261;
  for (const char of text) value = Math.imul(value ^ char.codePointAt(0)!, 16777619);
  return value >>> 0;
}
/** Three independent feature tables; all packed inputs are exactly representable f32 integers. */
export function featureRows(packed: number): number[] {
  const bits = packed - 1;
  return [bits & 1023, 1024 + ((bits >>> 10) & 255), 1280 + ((bits >>> 18) & 31)];
}
