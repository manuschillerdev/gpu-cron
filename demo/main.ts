import { defineParser } from '../src/index.js';
import type { ParseResult } from '../src/index.js';
import './style.css';

function get<T extends HTMLElement>(id: string): T {
  return document.getElementById(id) as T;
}
const expression = get<HTMLTextAreaElement>('expression');
const timezone = get<HTMLSelectElement>('timezone');
const reference = get<HTMLInputElement>('reference');
const copy = get<HTMLButtonElement>('copy');
reference.value = new Date().toISOString();
const parser = defineParser();
let revision = 0;
let result: ParseResult | null = null;
let timer: ReturnType<typeof setTimeout>;

function clearOutput(): void {
  result = null;
  copy.disabled = true;
  get('occurrences').replaceChildren();
  get('diagnostics').replaceChildren();
  get('json').textContent = '';
  get('network-tokens').replaceChildren();
  get('network-status').textContent = 'Waiting for inference…';
  get('compiler-status').textContent = 'Waiting for model predictions…';
  document.querySelectorAll('#cron-fields strong').forEach((node) => {
    node.textContent = '—';
  });
}

function renderNetwork(result: ParseResult, elapsedMilliseconds: number): void {
  get('status').textContent =
    `${result.model.backend.toUpperCase()} · ${elapsedMilliseconds.toFixed(0)} ms`;

  const unknownTokens = result.model.tokens.filter((token) => token.role === 'unknown').length;
  const unknownSummary = unknownTokens
    ? ` · ${unknownTokens} unknown token${unknownTokens === 1 ? '' : 's'}`
    : '';
  get('network-status').textContent =
    `${result.model.family} · ${(result.model.confidence * 100).toFixed(1)}% model score${unknownSummary}`;

  const container = get('network-tokens');
  for (const token of result.model.tokens) {
    const item = document.createElement('span');
    item.className = 'network-token';
    item.dataset.role = token.role;
    item.title = `${token.role}: ${(token.confidence * 100).toFixed(1)}% model score`;

    const text = document.createElement('span');
    text.textContent = token.text;
    const label = document.createElement('small');
    label.textContent = token.role;
    item.append(text, label);
    container.append(item);
  }
}

function renderCompiler(result: ParseResult): void {
  get('description').textContent =
    result.description ?? 'The compiler could not produce a schedule.';
  get('compiler-status').textContent = result.schedule
    ? 'Structured schedule produced.'
    : 'No schedule produced. Compiler diagnostics are shown below.';

  copy.disabled = !result.cron;
  const fields = result.cron?.split(' ') ?? ['—', '—', '—', '—', '—'];
  document.querySelectorAll('#cron-fields strong').forEach((node, index) => {
    node.textContent = fields[index]!;
  });

  const diagnostics = get('diagnostics');
  for (const diagnostic of result.diagnostics) {
    const message = document.createElement('p');
    message.className = `diagnostic ${diagnostic.severity}`;
    message.textContent = diagnostic.message;
    diagnostics.append(message);
  }
}

function renderOccurrences(result: ParseResult): void {
  const date = new Intl.DateTimeFormat('en-GB', {
    timeZone: timezone.value,
    weekday: 'short',
    day: '2-digit',
    month: 'short',
    year: 'numeric',
  });
  const time = new Intl.DateTimeFormat('en-GB', {
    timeZone: timezone.value,
    hour: '2-digit',
    minute: '2-digit',
    timeZoneName: 'short',
  });

  get('preview-zone').textContent = timezone.value;
  const occurrences = get('occurrences');
  result.occurrences.forEach((instant, index) => {
    const item = document.createElement('li');
    const sequence = document.createElement('span');
    sequence.className = 'run-index';
    sequence.textContent = String(index + 1).padStart(2, '0');
    const day = document.createElement('span');
    day.textContent = date.format(new Date(instant));
    const clock = document.createElement('time');
    clock.dateTime = instant;
    clock.textContent = time.format(new Date(instant));
    item.append(sequence, day, clock);
    occurrences.append(item);
  });
}

function renderResult(next: ParseResult, elapsedMilliseconds: number): void {
  result = next;
  renderNetwork(next, elapsedMilliseconds);
  renderCompiler(next);
  renderOccurrences(next);
  get('json').textContent = JSON.stringify(next, null, 2);
}

async function update(): Promise<void> {
  const current = ++revision;
  const panel = document.querySelector('.result')!;
  panel.setAttribute('aria-busy', 'true');
  clearOutput();
  get('description').textContent = 'Reading your schedule…';
  get('status').textContent = 'Reading…';
  try {
    const start = performance.now();
    const next = await parser.parse(expression.value, {
      timeZone: timezone.value,
      reference: reference.value,
      count: 5,
    });
    if (current !== revision) return;
    renderResult(next, performance.now() - start);
  } catch (error) {
    if (current !== revision) return;
    clearOutput();
    get('description').textContent = error instanceof Error ? error.message : String(error);
    get('status').textContent = 'Needs attention';
  } finally {
    if (current === revision) panel.setAttribute('aria-busy', 'false');
  }
}

expression.addEventListener('input', () => {
  ++revision;
  clearOutput();
  get('description').textContent = 'Reading your schedule…';
  clearTimeout(timer);
  timer = setTimeout(() => void update(), 180);
});
timezone.addEventListener('change', () => void update());
reference.addEventListener('change', () => void update());
document.querySelectorAll<HTMLButtonElement>('[data-example]').forEach((button) =>
  button.addEventListener('click', () => {
    clearTimeout(timer);
    expression.value = button.dataset.example!;
    void update();
  }),
);
copy.addEventListener('click', async () => {
  if (!result?.cron) return;
  try {
    await navigator.clipboard.writeText(result.cron);
    copy.textContent = 'Copied ✓';
  } catch {
    copy.textContent = 'Copy unavailable';
  }
  setTimeout(() => {
    copy.textContent = 'Copy cron ↗';
  }, 1500);
});
window.addEventListener('pagehide', () => parser.dispose());
void update();
