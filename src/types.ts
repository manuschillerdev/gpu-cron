export type Family = 'minutes' | 'hours' | 'daily' | 'weekly' | 'biweekly' | 'monthly' | 'invalid';

export interface Diagnostic {
  code:
    | 'invalid-input'
    | 'unsupported-syntax'
    | 'low-confidence'
    | 'invalid-value'
    | 'not-cron'
    | 'assumption'
    | 'preview-limit';
  severity: 'error' | 'warning' | 'info';
  message: string;
}

/** Calendar recurrence. All fields use local wall-clock time in timeZone. */
export interface Schedule {
  family: Exclude<Family, 'invalid'>;
  minutes: number[];
  hours: number[];
  /** null means every day; -1 means the last day of the month. */
  daysOfMonth: number[] | null;
  /** Sunday = 0. null means every weekday. */
  weekdays: number[] | null;
  months: number[];
  weekInterval: 1 | 2;
  /** Local Monday YYYY-MM-DD, preserving the phase of alternate weeks. */
  anchorWeek: string;
  timeZone: string;
}

export interface ParseOptions {
  /** An instant with Z or an explicit offset. Defaults to now. */
  reference?: string | Date;
  /** Defaults to UTC, never implicitly uses the host timezone. */
  timeZone?: string;
  /** Number of future instants, exclusive of reference. Default 5; 0 disables preview. */
  count?: number;
}

export interface ParseResult {
  input: string;
  schedule: Schedule | null;
  /** Standard five-field cron, or null when the schedule is not representable. */
  cron: string | null;
  description: string | null;
  occurrences: string[];
  diagnostics: Diagnostic[];
  model: {
    family: Family;
    confidence: number;
    tokens: import('./model/parameters.js').TokenPrediction[];
    backend: 'webgpu';
    executionProvider: 'webgpu';
    outputLocations: string[];
  };
}

export interface Parser {
  parse(text: string, options?: ParseOptions): Promise<ParseResult>;
  parseMany(texts: readonly string[], options?: ParseOptions): Promise<ParseResult[]>;
  dispose(): void;
}
