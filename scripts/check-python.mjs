import { spawnSync } from 'node:child_process';

// The project currently has JavaScript tests only. Pytest still runs so an
// existing Python suite, if present, cannot be silently omitted from checks.
for (const args of [
  ['ruff', 'check', '.'],
  ['ruff', 'format', '--check', '.'],
  ['--group', 'verification', 'ty', 'check'],
  ['pytest'],
]) {
  const result = spawnSync('uv', ['run', '--locked', ...args], {
    cwd: new URL('../training/', import.meta.url),
    stdio: 'inherit',
  });
  if (result.error) throw result.error;
  if (args[0] === 'pytest' && result.status === 5) {
    console.log('No Python tests collected; the existing regression suite is JavaScript.');
  } else if (result.status !== 0) {
    process.exit(result.status ?? 1);
  }
}
