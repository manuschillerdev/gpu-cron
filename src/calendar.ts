import type { ParseOptions, Schedule } from './types.js';

const DAY = 86_400_000;
const MINUTE = 60_000;
interface Parts {
  year: number;
  month: number;
  day: number;
  hour: number;
  minute: number;
}

function formatter(timeZone: string): Intl.DateTimeFormat {
  return new Intl.DateTimeFormat('en-US', {
    timeZone,
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
    hourCycle: 'h23',
  });
}

function parts(time: number, format: Intl.DateTimeFormat): Parts {
  const result: Record<string, number> = {};
  for (const part of format.formatToParts(time))
    if (part.type !== 'literal') result[part.type] = Number(part.value);
  return result as unknown as Parts;
}

function epoch(p: Parts): number {
  return Date.UTC(p.year, p.month - 1, p.day, p.hour, p.minute);
}

function mondayOfWeek(localDay: number): number {
  const weekday = new Date(localDay).getUTCDay();
  return localDay - ((weekday + 6) % 7) * DAY;
}

export function context(options: ParseOptions): {
  reference: number;
  timeZone: string;
  count: number;
  anchorWeek: string;
} {
  const timeZone = options.timeZone ?? 'UTC';
  const format = formatter(timeZone); // Validates IANA zone; no silent host-zone fallback.
  const value = options.reference;
  if (typeof value === 'string' && !/T.*(?:Z|[+-]\d{2}:\d{2})$/i.test(value))
    throw new RangeError('reference must include a time and Z or an explicit UTC offset.');
  const reference = value === undefined ? Date.now() : new Date(value).getTime();
  if (
    !Number.isFinite(reference) ||
    reference < Date.UTC(2000, 0, 1) ||
    reference > Date.UTC(2099, 11, 31)
  )
    throw new RangeError('reference must be a valid instant between 2000 and 2099.');
  const count = options.count ?? 5;
  if (!Number.isInteger(count) || count < 0 || count > 100)
    throw new RangeError('count must be an integer from 0 to 100.');
  const local = parts(reference, format);
  const date = Date.UTC(local.year, local.month - 1, local.day);
  return {
    reference,
    timeZone,
    count,
    anchorWeek: new Date(mondayOfWeek(date)).toISOString().slice(0, 10),
  };
}

function dateMatches(schedule: Schedule, localDay: number, anchor: number): boolean {
  const date = new Date(localDay);
  const weekday = date.getUTCDay();
  const month = date.getUTCMonth() + 1;
  const day = date.getUTCDate();

  if (!schedule.months.includes(month)) return false;
  if (schedule.weekdays && !schedule.weekdays.includes(weekday)) return false;

  if (schedule.daysOfMonth) {
    const lastDay = new Date(Date.UTC(date.getUTCFullYear(), month, 0)).getUTCDate();
    const matchesMonthDay = schedule.daysOfMonth.some(
      (value) => value === day || (value === -1 && day === lastDay),
    );
    if (!matchesMonthDay) return false;
  }

  const weeksFromAnchor = Math.round((mondayOfWeek(localDay) - anchor) / (7 * DAY));
  return schedule.weekInterval === 1 || weeksFromAnchor % 2 === 0;
}

function offsetsNear(localDay: number, format: Intl.DateTimeFormat): Set<number> {
  // Sampling both sides of a transition finds every offset that can map this local day.
  const offsets = new Set<number>();
  for (const hours of [-36, -12, 12, 36]) {
    const sample = localDay + hours * 3_600_000;
    offsets.add(epoch(parts(sample, format)) - sample);
  }
  return offsets;
}

function instantsOnDay(
  schedule: Schedule,
  localDay: number,
  reference: number,
  format: Intl.DateTimeFormat,
): number[] {
  const result: number[] = [];
  const offsets = offsetsNear(localDay, format);

  for (const hour of schedule.hours) {
    for (const minute of schedule.minutes) {
      const wallClock = localDay + hour * 3_600_000 + minute * MINUTE;
      for (const offset of offsets) {
        const instant = wallClock - offset;
        const existsInTimeZone = epoch(parts(instant, format)) === wallClock;
        if (instant > reference && existsInTimeZone) result.push(instant);
      }
    }
  }
  return result.sort((a, b) => a - b);
}

/** Exclusive of reference. Gaps are skipped; both instants in a repeated hour are returned. */
export function preview(schedule: Schedule, reference: number, count: number): string[] {
  if (count === 0) return [];
  const format = formatter(schedule.timeZone);
  const initial = parts(reference, format);
  const start = Date.UTC(initial.year, initial.month - 1, initial.day);
  const anchor = Date.parse(`${schedule.anchorWeek}T00:00:00Z`);
  const output: number[] = [];
  for (let delta = 0; delta < 366 * 5; delta++) {
    const localDay = start + delta * DAY;
    if (!dateMatches(schedule, localDay, anchor)) continue;
    output.push(...instantsOnDay(schedule, localDay, reference, format));
    if (output.length >= count) break;
  }
  return output.slice(0, count).map((value) => new Date(value).toISOString());
}
