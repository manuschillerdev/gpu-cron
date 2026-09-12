export const FEATURE_COUNT = 512;
export const HIDDEN_COUNT = 24;
const WORD_ROWS = 1024;
const CONSONANT_ROWS = 256;
export const ROLES = [
  'prefix',
  'recurrence',
  'quantity',
  'hour',
  'minute',
  'meridiem',
  'clock-offset',
  'clock-direction',
  'weekday',
  'range',
  'monthday',
  'month',
  'excluded-weekday',
  'excluded-month',
  'exclusion',
  'unknown',
] as const;
export interface Token {
  text: string;
  start: number;
  end: number;
}
export function tokenize(text: string): Token[] {
  return Array.from(text.matchAll(/[A-Za-z]+|[0-9]+|[^\s]/gu), (m) => ({
    text: m[0],
    start: m.index,
    end: m.index + m[0].length,
  }));
}
export const FAMILIES = [
  'minutes',
  'hours',
  'daily',
  'weekly',
  'biweekly',
  'monthly',
  'invalid',
] as const;

export const DAY_NAMES = [
  'sunday',
  'monday',
  'tuesday',
  'wednesday',
  'thursday',
  'friday',
  'saturday',
];
export const MONTH_NAMES = [
  'january',
  'february',
  'march',
  'april',
  'may',
  'june',
  'july',
  'august',
  'september',
  'october',
  'november',
  'december',
];
/** Shared with training/train.py; cross-language parity is checked in tests. */
export function features(text: string): Float32Array {
  const tokens = tokenize(text);
  if (tokens.length > FEATURE_COUNT) throw new RangeError('Sequence exceeds token capacity.');
  const result = new Float32Array(FEATURE_COUNT);
  for (let index = 0; index < tokens.length; index++)
    result[index] = encodeToken(tokens[index]!.text);
  return result;
}

/** Pack three lookup-table indices into one exactly representable float32 integer. */
function encodeToken(text: string): number {
  const word = text.toLowerCase();
  const kind = /^\d+$/.test(word) ? 1 : /^[a-z]+$/.test(word) ? 0 : 2;
  const lengthBucket = Math.min(Math.floor(word.length / 3), 3);
  const hasUppercase = Number(text !== word);
  const shape = kind | (lengthBucket << 2) | (hasUppercase << 4);
  const consonants = word.replace(/[aeiou]/g, '');

  const wordRow = hash(word) & (WORD_ROWS - 1);
  const consonantRow = hash(consonants) & (CONSONANT_ROWS - 1);
  return 1 + wordRow + (consonantRow << 10) + (shape << 18);
}

function hash(text: string): number {
  let value = 2166136261;
  for (const char of text) value = Math.imul(value ^ char.codePointAt(0)!, 16777619);
  return value >>> 0;
}
/** Three independent feature tables; all packed inputs are exactly representable f32 integers. */
export function featureRows(packed: number): number[] {
  const bits = packed - 1;
  const wordRow = bits & (WORD_ROWS - 1);
  const consonantRow = WORD_ROWS + ((bits >>> 10) & (CONSONANT_ROWS - 1));
  const shapeRow = WORD_ROWS + CONSONANT_ROWS + ((bits >>> 18) & 31);
  return [wordRow, consonantRow, shapeRow];
}
