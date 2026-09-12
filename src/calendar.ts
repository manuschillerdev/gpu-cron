import type { ParseOptions, Schedule } from './types.js';

const DAY = 86_400_000;
const MINUTE = 60_000;
interface Parts { year: number; month: number; day: number; hour: number; minute: number }

export function formatter(timeZone: string): Intl.DateTimeFormat {
  return new Intl.DateTimeFormat('en-US', { timeZone, year: 'numeric', month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit', hourCycle: 'h23' });
}

function parts(time: number, format: Intl.DateTimeFormat): Parts {
  const result: Record<string, number> = {};
  for (const part of format.formatToParts(time)) if (part.type !== 'literal') result[part.type] = Number(part.value);
  return result as unknown as Parts;
}

function epoch(p: Parts): number { return Date.UTC(p.year, p.month - 1, p.day, p.hour, p.minute); }

export function context(options: ParseOptions): { reference: number; timeZone: string; count: number; anchorWeek: string } {
  const timeZone = options.timeZone ?? 'UTC';
  const format = formatter(timeZone); // Validates IANA zone; no silent host-zone fallback.
  const value = options.reference;
  if (typeof value === 'string' && !/T.*(?:Z|[+-]\d{2}:\d{2})$/i.test(value)) throw new RangeError('reference must include a time and Z or an explicit UTC offset.');
  const reference = value === undefined ? Date.now() : new Date(value).getTime();
  if (!Number.isFinite(reference) || reference < Date.UTC(2000, 0, 1) || reference > Date.UTC(2099, 11, 31)) throw new RangeError('reference must be a valid instant between 2000 and 2099.');
  const count = options.count ?? 5;
  if (!Number.isInteger(count) || count < 0 || count > 100) throw new RangeError('count must be an integer from 0 to 100.');
  const local = parts(reference, format);
  const date = Date.UTC(local.year, local.month - 1, local.day);
  const monday = date - ((new Date(date).getUTCDay() + 6) % 7) * DAY;
  return { reference, timeZone, count, anchorWeek: new Date(monday).toISOString().slice(0, 10) };
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
    const date = new Date(localDay);
    const weekday = date.getUTCDay();
    const month = date.getUTCMonth() + 1;
    const day = date.getUTCDate();
    if (!schedule.months.includes(month)) continue;
    if (schedule.weekdays && !schedule.weekdays.includes(weekday)) continue;
    if (schedule.daysOfMonth) {
      const last = new Date(Date.UTC(date.getUTCFullYear(), month, 0)).getUTCDate();
      if (!schedule.daysOfMonth.some(value => value === day || (value === -1 && day === last))) continue;
    }
    const monday = localDay - ((weekday + 6) % 7) * DAY;
    if (schedule.weekInterval === 2 && Math.round((monday - anchor) / (7 * DAY)) % 2 !== 0) continue;
    // Sampling surrounding instants captures offsets on both sides of a DST transition.
    const offsets = new Set<number>();
    for (const shift of [-36, -12, 12, 36]) {
      const sample = localDay + shift * 3_600_000;
      offsets.add(epoch(parts(sample, format)) - sample);
    }
    const today: number[] = [];
    for (const hour of schedule.hours) for (const minute of schedule.minutes) {
      const wall = localDay + hour * 3_600_000 + minute * MINUTE;
      for (const offset of offsets) {
        const instant = wall - offset;
        if (instant > reference && epoch(parts(instant, format)) === wall) today.push(instant);
      }
    }
    output.push(...today.sort((a, b) => a - b));
    if (output.length >= count) break;
  }
  return output.slice(0, count).map(value => new Date(value).toISOString());
}
