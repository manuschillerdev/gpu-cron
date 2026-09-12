import { defineParser } from '../src/index.js';
import type { ParseResult } from '../src/index.js';
import './style.css';

function get<T extends HTMLElement>(id: string): T { return document.getElementById(id) as T; }
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
  document.querySelectorAll('#cron-fields strong').forEach(node => { node.textContent = '—'; });
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
    const next = await parser.parse(expression.value, { timeZone: timezone.value, reference: reference.value, count: 5 });
    if (current !== revision) return;
    result = next;
    get('status').textContent = `${next.model.backend.toUpperCase()} · ${(performance.now() - start).toFixed(0)} ms`;
    get('description').textContent = next.description ?? 'The compiler could not produce a schedule.';
    const unknown = next.model.tokens.filter(token => token.role === 'unknown').length;
    get('network-status').textContent = `${next.model.family} · ${(next.model.confidence * 100).toFixed(1)}% model score${unknown ? ` · ${unknown} unknown token${unknown === 1 ? '' : 's'}` : ''}`;
    for (const token of next.model.tokens) {
      const item = document.createElement('span');
      item.className = 'network-token'; item.dataset.role = token.role;
      const text = document.createElement('span'); text.textContent = token.text;
      const label = document.createElement('small'); label.textContent = token.role;
      item.title = `${token.role}: ${(token.confidence * 100).toFixed(1)}% model score`;
      item.append(text, label); get('network-tokens').append(item);
    }
    get('compiler-status').textContent = next.schedule ? 'Structured schedule produced.' : 'No schedule produced. Compiler diagnostics are shown below.';
    copy.disabled = !next.cron;
    const fields = next.cron?.split(' ') ?? ['—', '—', '—', '—', '—'];
    document.querySelectorAll('#cron-fields strong').forEach((node, i) => { node.textContent = fields[i]!; });
    for (const diagnostic of next.diagnostics) {
      const node = document.createElement('p');
      node.className = `diagnostic ${diagnostic.severity}`;
      node.textContent = diagnostic.message;
      get('diagnostics').append(node);
    }
    const date = new Intl.DateTimeFormat('en-GB', { timeZone: timezone.value, weekday: 'short', day: '2-digit', month: 'short', year: 'numeric' });
    const time = new Intl.DateTimeFormat('en-GB', { timeZone: timezone.value, hour: '2-digit', minute: '2-digit', timeZoneName: 'short' });
    get('preview-zone').textContent = timezone.value;
    next.occurrences.forEach((instant, i) => {
      const li = document.createElement('li');
      const index = document.createElement('span'); index.className = 'run-index'; index.textContent = String(i + 1).padStart(2, '0');
      const day = document.createElement('span'); day.textContent = date.format(new Date(instant));
      const clock = document.createElement('time'); clock.dateTime = instant; clock.textContent = time.format(new Date(instant));
      li.append(index, day, clock); get('occurrences').append(li);
    });
    get('json').textContent = JSON.stringify(next, null, 2);
  } catch (error) {
    if (current !== revision) return;
    clearOutput();
    get('description').textContent = error instanceof Error ? error.message : String(error);
    get('status').textContent = 'Needs attention';
  } finally { if (current === revision) panel.setAttribute('aria-busy', 'false'); }
}

expression.addEventListener('input', () => {
  ++revision; clearOutput(); get('description').textContent = 'Reading your schedule…';
  clearTimeout(timer); timer = setTimeout(() => void update(), 180);
});
timezone.addEventListener('change', () => void update());
reference.addEventListener('change', () => void update());
document.querySelectorAll<HTMLButtonElement>('[data-example]').forEach(button => button.addEventListener('click', () => {
  clearTimeout(timer); expression.value = button.dataset.example!; void update();
}));
copy.addEventListener('click', async () => {
  if (!result?.cron) return;
  try { await navigator.clipboard.writeText(result.cron); copy.textContent = 'Copied ✓'; }
  catch { copy.textContent = 'Copy unavailable'; }
  setTimeout(() => { copy.textContent = 'Copy cron ↗'; }, 1500);
});
window.addEventListener('pagehide', () => parser.dispose());
void update();
