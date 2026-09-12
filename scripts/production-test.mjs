import {preview} from 'vite';
import {chromium} from 'playwright';
import assert from 'node:assert/strict';
import {readdir} from 'node:fs/promises';

const assets = await readdir('site/assets');
assert.ok(assets.every(name => !/\.wasm$/.test(name)), 'Production output must contain no WASM payloads');
const server = await preview({preview:{host:'127.0.0.1',port:4180,strictPort:false},logLevel:'error'});
let browser;
try {
  browser = await chromium.launch({channel:'chromium',headless:true,args:['--enable-unsafe-webgpu']});
  const page = await browser.newPage();
  const forbidden = [], errors = [];
  page.on('request',r=>{if(/\.wasm(?:\?|$)/.test(r.url()))forbidden.push(r.url());});
  page.on('pageerror',e=>errors.push(e.message));
  const base = server.resolvedUrls.local[0];
  await page.goto(base);
  await page.waitForFunction(()=>document.querySelector('#status').textContent.startsWith('WEBGPU'));
  assert.equal(await page.locator('#copy').isDisabled(),false);
  assert.deepEqual(forbidden,[]);
  assert.deepEqual(errors,[]);
  console.log('The production demo passes on WebGPU with zero WASM assets or requests.');
} finally { await browser?.close(); await new Promise(resolve=>server.httpServer.close(resolve)); }
