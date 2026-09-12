import { DAY_NAMES, MONTH_NAMES } from './features.js';
import type { Diagnostic, Family, Schedule } from './types.js';
import type { TokenPrediction } from './model/parameters.js';

const ALL_MONTHS = Array.from({ length: 12 }, (_, i) => i + 1);
const ALL_DAYS = Array.from({ length: 7 }, (_, i) => i);
const range = (count: number, start = 0) => Array.from({ length: count }, (_, i) => start + i);
const unique = (values: number[]) => [...new Set(values)].sort((a, b) => a - b);
class CompileError extends Error {
  constructor(
    public code: 'unsupported-syntax' | 'invalid-value',
    message: string,
  ) {
    super(message);
  }
}
function unsupported(message: string): never {
  throw new CompileError('unsupported-syntax', message);
}
function invalid(message: string): never {
  throw new CompileError('invalid-value', message);
}
const cardinalWords =
  'zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty'.split(
    ' ',
  );
const ordinalWords =
  'zeroth first second third fourth fifth sixth seventh eighth ninth tenth eleventh twelfth thirteenth fourteenth fifteenth sixteenth seventeenth eighteenth nineteenth twentieth'.split(
    ' ',
  );
const values: Record<string, number> = Object.fromEntries([
  ...cardinalWords.map((word, value) => [word, value]),
  ...ordinalWords.map((word, value) => [word, value]),
  ['thirty', 30],
  ['thirtieth', 30],
  ['forty', 40],
  ['fifty', 50],
  ['quarter', 15],
  ['half', 30],
  ['noon', 12],
  ['midnight', 0],
  ['last', -1],
  ['end', -1],
]);
const word = (t: TokenPrediction) => t.text.toLowerCase();

/** Decode only values identified by the network; never recover missing roles from text. */
function numericValue(tokens: TokenPrediction[]): number {
  if (!tokens.length) unsupported('The model did not identify a required numeric value.');
  let total = 0;
  for (let index = 0; index < tokens.length; index++) {
    const text = word(tokens[index]!);
    const value = /^\d+$/.test(text) ? Number(text) : values[text];
    if (value === undefined) unsupported(`Cannot interpret the predicted numeric value "${text}".`);
    const completesCompoundNumber =
      total >= 20 && total % 10 === 0 && value >= 1 && value <= 9 && !/^\d+$/.test(text);
    if (index > 0 && !completesCompoundNumber)
      unsupported('The model predicted more than one numeric value for a field.');
    total += value;
  }
  return total;
}

function namedValue(token: TokenPrediction, months = false): number[] {
  const text = word(token);
  if (!months && /^weekdays?$/.test(text)) return [1, 2, 3, 4, 5];
  if (!months && /^weekends?$/.test(text)) return [0, 6];
  const names = months ? MONTH_NAMES : DAY_NAMES;
  const index = names.findIndex(
    (name) => text === name || text === name.slice(0, 3) || text === name + 's',
  );
  if (index < 0)
    unsupported(`Cannot interpret the predicted ${months ? 'month' : 'weekday'} "${text}".`);
  return [index + (months ? 1 : 0)];
}

function namedValues(
  tokens: TokenPrediction[],
  role: TokenPrediction['role'],
  consumed: Set<TokenPrediction>,
): number[] {
  const months = role === 'month' || role === 'excluded-month';
  const result: number[] = [];
  const selected = tokens
    .map((token, index) => ({ token, index }))
    .filter(({ token }) => token.role === role);

  for (let selectedIndex = 0; selectedIndex < selected.length; selectedIndex++) {
    const current = selected[selectedIndex]!;
    const firstValues = namedValue(current.token, months);
    result.push(...firstValues);

    const next = selected[selectedIndex + 1];
    if (!next) continue;
    const between = tokens.slice(current.index + 1, next.index);
    const rangeTokens = between.filter((token) => token.role === 'range');
    if (!rangeTokens.length) continue;

    const lastValues = namedValue(next.token, months);
    if (firstValues.length !== 1 || lastValues.length !== 1)
      unsupported('Ranges require individual day or month endpoints.');
    for (const token of rangeTokens) consumed.add(token);

    let value = firstValues[0]!;
    for (let count = 0; value !== lastValues[0]! && count < 12; count++) {
      value = months ? (value % 12) + 1 : (value + 1) % 7;
      result.push(value);
    }
    selectedIndex++;
  }
  return unique(result);
}

type Role = TokenPrediction['role'];
type TokensByRole = Record<Role, TokenPrediction[]>;

function indexTokensByRole(tokens: TokenPrediction[]): TokensByRole {
  const result: TokensByRole = {
    prefix: [],
    recurrence: [],
    quantity: [],
    hour: [],
    minute: [],
    meridiem: [],
    'clock-offset': [],
    'clock-direction': [],
    weekday: [],
    range: [],
    monthday: [],
    month: [],
    'excluded-weekday': [],
    'excluded-month': [],
    exclusion: [],
    unknown: [],
  };
  for (const token of tokens) result[token.role].push(token);
  return result;
}

function validatePrediction(
  family: Family,
  roles: TokensByRole,
): asserts family is Exclude<Family, 'invalid'> {
  if (family === 'invalid' || Object.values(roles).every((tokens) => tokens.length === 0))
    unsupported('The model identified unsupported or unresolved schedule language.');
  if (roles.unknown.length)
    unsupported('The model identified unsupported or unresolved schedule language.');
  if (family !== 'monthly' && roles.monthday.length)
    unsupported('Predicted month days conflict with the recurrence family.');
  if (family !== 'minutes' && family !== 'hours' && roles.quantity.length)
    unsupported('An interval quantity conflicts with the clock-based recurrence.');
}

function emptySchedule(
  family: Exclude<Family, 'invalid'>,
  timeZone: string,
  anchorWeek: string,
): Schedule {
  return {
    family,
    minutes: [0],
    hours: [0],
    daysOfMonth: null,
    weekdays: null,
    months: [...ALL_MONTHS],
    weekInterval: family === 'biweekly' ? 2 : 1,
    anchorWeek,
    timeZone,
  };
}

function compileInterval(schedule: Schedule, roles: TokensByRole): void {
  const hasClock = [
    ...roles.hour,
    ...roles.minute,
    ...roles.meridiem,
    ...roles['clock-offset'],
    ...roles['clock-direction'],
  ].length;
  if (hasClock) unsupported('An interval with an additional clock restriction is not supported.');
  if (roles.weekday.length) unsupported('An interval with included weekdays is not supported.');
  if (roles.recurrence.some((token) => /^\d+$/.test(word(token))))
    unsupported('A numeric recurrence token has no predicted quantity role.');

  const interval = roles.quantity.length ? numericValue(roles.quantity) : 1;
  const fieldSize = schedule.family === 'minutes' ? 60 : 24;
  if (interval < 1 || interval > fieldSize || fieldSize % interval !== 0)
    invalid(
      `The interval must divide ${fieldSize} evenly to preserve spacing across cron field boundaries.`,
    );

  if (schedule.family === 'minutes') {
    schedule.minutes = range(60).filter((value) => value % interval === 0);
    schedule.hours = range(24);
  } else {
    schedule.minutes = [0];
    schedule.hours = range(24).filter((value) => value % interval === 0);
  }
}

interface Clock {
  hour: number;
  minute: number;
  assumed24Hour: boolean;
}

function compileClock(roles: TokensByRole): Clock {
  let hour = numericValue(roles.hour);
  let minute = roles.minute.length ? numericValue(roles.minute) : 0;
  const meridiems = roles.meridiem.map(word).filter((value) => value !== 'm');

  if (meridiems.length > 1) unsupported('More than one meridiem was predicted.');
  if (hour < 0 || hour > 23 || minute < 0 || minute > 59) invalid('Clock time is out of range.');

  if (meridiems.length === 1) {
    const meridiem = meridiems[0]!;
    const afternoon = ['pm', 'p', 'afternoon', 'evening', 'night'];
    const valid = ['am', 'a', 'morning', ...afternoon];
    if (!valid.includes(meridiem)) unsupported('Cannot interpret the predicted meridiem.');
    if (hour < 1 || hour > 12) invalid('A clock with AM/PM must have an hour between 1 and 12.');
    hour = (hour % 12) + (afternoon.includes(meridiem) ? 12 : 0);
  }

  if (roles['clock-offset'].length) {
    if (roles.minute.length || roles['clock-direction'].length !== 1)
      unsupported('Clock offsets need one direction and no separate minute field.');
    const offset = numericValue(roles['clock-offset']);
    if (offset < 1 || offset > 59) invalid('Clock offset is out of range.');

    const direction = word(roles['clock-direction'][0]!);
    if (!['to', 'before', 'past', 'after'].includes(direction))
      unsupported('Cannot interpret the predicted clock direction.');
    const signedOffset = ['to', 'before'].includes(direction) ? -offset : offset;
    const minutesSinceMidnight = (hour * 60 + signedOffset + 24 * 60) % (24 * 60);
    hour = Math.floor(minutesSinceMidnight / 60);
    minute = minutesSinceMidnight % 60;
  } else if (roles['clock-direction'].length) {
    unsupported('Clock direction has no offset.');
  }

  const namedClock = roles.hour.some((token) => ['noon', 'midnight'].includes(word(token)));
  return {
    hour,
    minute,
    assumed24Hour:
      meridiems.length === 0 &&
      roles.minute.length === 0 &&
      roles['clock-offset'].length === 0 &&
      !namedClock,
  };
}

function monthDays(text: string, tokens: TokenPrediction[]): number[] {
  if (!tokens.length) unsupported('The model did not identify a day of the month.');

  // Adjacent words such as "twenty first" form one value. Punctuation starts a new value.
  const groups: TokenPrediction[][] = [];
  for (const token of tokens) {
    const previous = groups.at(-1)?.at(-1);
    const separatedOnlyBySpace = previous && /^\s+$/.test(text.slice(previous.end, token.start));
    if (separatedOnlyBySpace && !['last', 'end'].includes(word(token))) groups.at(-1)!.push(token);
    else groups.push([token]);
  }

  const days = unique(groups.map(numericValue));
  if (days.some((day) => day !== -1 && (day < 1 || day > 31)))
    invalid('Day of month must be between 1 and 31, or last.');
  return days;
}

function compileCalendarFamily(
  text: string,
  schedule: Schedule,
  tokens: TokenPrediction[],
  roles: TokensByRole,
  consumedRanges: Set<TokenPrediction>,
  diagnostics: Diagnostic[],
): void {
  const clock = compileClock(roles);
  schedule.hours = [clock.hour];
  schedule.minutes = [clock.minute];

  if (clock.assumed24Hour)
    diagnostics.push({
      code: 'assumption',
      severity: 'info',
      message: `Interpreted the predicted hour as ${String(clock.hour).padStart(2, '0')}:00 using a 24-hour clock.`,
    });

  if (schedule.family === 'weekly' || schedule.family === 'biweekly') {
    schedule.weekdays = namedValues(tokens, 'weekday', consumedRanges);
    if (!schedule.weekdays.length) unsupported('The model did not identify scheduled weekdays.');
  } else if (schedule.family === 'daily' && roles.weekday.length) {
    unsupported('Daily family conflicts with predicted weekday restrictions.');
  }

  if (schedule.family === 'monthly') {
    if (roles.weekday.length)
      unsupported(
        'A monthly ordinal combined with a weekday requires an unsupported nth-weekday rule.',
      );
    schedule.daysOfMonth = monthDays(text, roles.monthday);
  }

  if (schedule.family === 'biweekly')
    diagnostics.push({
      code: 'assumption',
      severity: 'info',
      message: `Alternate weeks are anchored to the local week starting ${schedule.anchorWeek} (Monday). Preserve anchorWeek when reusing this schedule.`,
    });
}

function applyRestrictions(
  schedule: Schedule,
  tokens: TokenPrediction[],
  roles: TokensByRole,
  consumedRanges: Set<TokenPrediction>,
): void {
  const includedMonths = namedValues(tokens, 'month', consumedRanges);
  if (includedMonths.length) schedule.months = includedMonths;

  const excludedMonths = namedValues(tokens, 'excluded-month', consumedRanges);
  const excludedWeekdays = namedValues(tokens, 'excluded-weekday', consumedRanges);
  const hasExcludedValue = excludedMonths.length || excludedWeekdays.length;
  if (hasExcludedValue && !roles.exclusion.length)
    unsupported('An excluded value has no predicted exclusion marker.');
  if (roles.exclusion.length && !hasExcludedValue)
    unsupported('The model identified an exclusion without a supported target.');

  schedule.months = schedule.months.filter((month) => !excludedMonths.includes(month));
  if (!schedule.months.length) invalid('The exclusions remove every month.');

  if (excludedWeekdays.length) {
    const includedWeekdays = schedule.weekdays ?? ALL_DAYS;
    schedule.weekdays = includedWeekdays.filter((day) => !excludedWeekdays.includes(day));
    if (!schedule.weekdays.length) invalid('The exclusions remove every scheduled weekday.');
  }

  if (roles.range.some((token) => !consumedRanges.has(token)))
    unsupported('A predicted range has no matching day or month endpoints.');
}

export function compile(
  text: string,
  family: Family,
  timeZone: string,
  anchorWeek: string,
  tokens: TokenPrediction[],
): { schedule: Schedule | null; diagnostics: Diagnostic[] } {
  const diagnostics: Diagnostic[] = [];
  try {
    const roles = indexTokensByRole(tokens);
    validatePrediction(family, roles);

    const schedule = emptySchedule(family, timeZone, anchorWeek);
    const consumedRanges = new Set<TokenPrediction>();

    if (family === 'minutes' || family === 'hours') {
      compileInterval(schedule, roles);
    } else {
      compileCalendarFamily(text, schedule, tokens, roles, consumedRanges, diagnostics);
    }

    applyRestrictions(schedule, tokens, roles, consumedRanges);
    return { schedule, diagnostics };
  } catch (error) {
    if (!(error instanceof CompileError)) throw error;
    return {
      schedule: null,
      diagnostics: [
        ...diagnostics,
        { code: error.code, severity: 'error', message: error.message },
      ],
    };
  }
}

function field(values: number[], count: number, start = 0): string {
  if (values.length === count) return '*';
  // Only emit steps for complete, zero-based field partitions.
  if (start === 0 && values.length > 1 && values[0] === 0) {
    const step = values[1]!;
    if (
      count % step === 0 &&
      values.length === count / step &&
      values.every((value, i) => value === i * step)
    )
      return `*/${step}`;
  }
  if (values.length > 2 && values.every((value, i) => value === values[0]! + i))
    return `${values[0]}-${values.at(-1)}`;
  return values.join(',');
}

export function cron(schedule: Schedule): {
  expression: string | null;
  reason?: string;
} {
  if (schedule.weekInterval === 2)
    return {
      expression: null,
      reason:
        'Five-field cron cannot preserve an every-other-week interval. Use the structured schedule and its anchorWeek.',
    };
  if (schedule.daysOfMonth?.includes(-1))
    return {
      expression: null,
      reason:
        'The last day of a month requires calendar logic; standard five-field cron has no L operator.',
    };
  if (schedule.daysOfMonth && schedule.weekdays)
    return {
      expression: null,
      reason:
        'Cron combines restricted day-of-month and weekday fields with OR. This schedule requires both restrictions (AND).',
    };
  return {
    expression: [
      field(schedule.minutes, 60),
      field(schedule.hours, 24),
      schedule.daysOfMonth ? field(schedule.daysOfMonth, 31, 1) : '*',
      field(schedule.months, 12, 1),
      schedule.weekdays ? field(schedule.weekdays, 7) : '*',
    ].join(' '),
  };
}

export function describe(schedule: Schedule): string {
  const time = `${String(schedule.hours[0]).padStart(2, '0')}:${String(schedule.minutes[0]).padStart(2, '0')}`;
  let text: string;
  if (schedule.family === 'minutes')
    text = `Every ${60 / schedule.minutes.length} minute${schedule.minutes.length === 60 ? '' : 's'}`;
  else if (schedule.family === 'hours')
    text = `Every ${24 / schedule.hours.length} hour${schedule.hours.length === 24 ? '' : 's'}, on the hour`;
  else if (schedule.daysOfMonth)
    text = `${schedule.daysOfMonth.map((day) => (day === -1 ? 'The last day' : `Day ${day}`)).join(' and ')} of each month at ${time}`;
  else if (schedule.weekdays)
    text = `${schedule.weekInterval === 2 ? 'Every other week on' : 'Every'} ${schedule.weekdays.map((day) => DAY_NAMES[day]).join(', ')} at ${time}`;
  else text = `Every day at ${time}`;
  if (
    (schedule.family === 'minutes' || schedule.family === 'hours' || schedule.daysOfMonth) &&
    schedule.weekdays
  )
    text += `, on ${schedule.weekdays.map((day) => DAY_NAMES[day]).join(', ')}`;
  if (schedule.months.length !== 12)
    text += `, except ${ALL_MONTHS.filter((month) => !schedule.months.includes(month))
      .map((month) => MONTH_NAMES[month - 1])
      .join(', ')}`;
  return `${text} · ${schedule.timeZone}`;
}
