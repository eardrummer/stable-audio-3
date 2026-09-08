#!/usr/bin/env node
/**
 * Vendor Strudel's sample banks into public/strudel-samples/ so Live Code works
 * with no network at all — which is what matters on stage.
 *
 * The audio is deliberately gitignored: it is tens of megabytes of third-party
 * samples that do not belong in this repo's history. Run this once per machine.
 *
 *   node scripts/fetch-strudel-samples.mjs            # core banks
 *   node scripts/fetch-strudel-samples.mjs --all      # + tidal-drum-machines
 */
import { mkdir, writeFile, readFile, stat } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import { dirname, join } from 'node:path';

const DOUGH = 'https://raw.githubusercontent.com/felixroos/dough-samples/main';
const OUT = new URL('../public/strudel-samples/', import.meta.url).pathname;

// Dirt-Samples is the classic `bd sd hh cp` vocabulary and is only ~74 files.
// tidal-drum-machines is 2348 files, so it is opt-in behind --all.
const CORE = ['Dirt-Samples', 'piano', 'EmuSP12'];
const EXTRA = ['tidal-drum-machines'];

const banks = process.argv.includes('--all') ? [...CORE, ...EXTRA] : CORE;
const CONCURRENCY = 12;

const human = (b) => b > 1e6 ? `${(b / 1e6).toFixed(1)} MB` : `${(b / 1e3).toFixed(0)} kB`;

/** Walk a sample map and yield every relative audio path it references. */
function collect(map) {
  const out = [];
  for (const [key, val] of Object.entries(map)) {
    if (key.startsWith('_')) continue;
    if (Array.isArray(val)) out.push(...val);
    else if (val && typeof val === 'object') for (const v of Object.values(val)) out.push(...[v].flat());
    else if (typeof val === 'string') out.push(val);
  }
  return out.filter((p) => typeof p === 'string');
}

async function download(url, dest) {
  if (existsSync(dest)) return (await stat(dest)).size;   // resumable across runs
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${res.status} ${url}`);
  const buf = Buffer.from(await res.arrayBuffer());
  await mkdir(dirname(dest), { recursive: true });
  await writeFile(dest, buf);
  return buf.length;
}

async function pool(items, worker) {
  let i = 0, bytes = 0, failed = 0;
  await Promise.all(Array.from({ length: CONCURRENCY }, async () => {
    while (i < items.length) {
      const idx = i++;
      try { bytes += await worker(items[idx]); }
      catch { failed++; }
      if (idx % 50 === 0) process.stdout.write(`\r    ${idx}/${items.length} …`);
    }
  }));
  return { bytes, failed };
}

let total = 0;
console.log(`\nFetching Strudel sample banks into public/strudel-samples/\n`);
for (const bank of banks) {
  const mapUrl = `${DOUGH}/${bank}.json`;
  const map = await (await fetch(mapUrl)).json();
  const base = (map._base || '').replace(/\/?$/, '/');
  const files = collect(map);
  console.log(`  ${bank} — ${files.length} files`);

  const { bytes, failed } = await pool(files, (rel) =>
    download(base + rel, join(OUT, bank, rel)));

  // Rewrite _base so Strudel loads from our local folder instead of the CDN.
  map._base = `/strudel-samples/${bank}/`;
  await mkdir(OUT, { recursive: true });
  await writeFile(join(OUT, `${bank}.json`), JSON.stringify(map));
  total += bytes;
  process.stdout.write(`\r    ${files.length} files, ${human(bytes)}` +
    (failed ? `, ${failed} failed` : '') + '\n');
}

// A manifest the app reads to know which banks are actually present.
await writeFile(join(OUT, 'banks.json'), JSON.stringify(banks));
console.log(`\n  total ${human(total)} in public/strudel-samples/`);
console.log(`  banks: ${banks.join(', ')}`);
if (!process.argv.includes('--all')) {
  console.log(`  (re-run with --all to add tidal-drum-machines, ~2348 more files)\n`);
}
