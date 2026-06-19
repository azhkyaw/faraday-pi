// Record the Faraday Pi web UI answering a question (SSE token streaming + sources panel)
// into docs/assets/demo-web.gif. Runs on the dev machine, driving a headless browser against
// the Pi over the LAN. No system ffmpeg needed — ffmpeg-static ships the binary.
//
//   cd scripts/web-demo && npm run setup && npm run record
//
// Env knobs: PI_URL (default Pi LAN UI), DEMO_Q (the question), OUT (gif path).
import { chromium } from 'playwright';
import ffmpegPath from 'ffmpeg-static';
import { execFileSync } from 'node:child_process';
import { mkdirSync, rmSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dir = path.dirname(fileURLToPath(import.meta.url));
const PI  = process.env.PI_URL || 'http://192.168.100.59:8000/';
const Q   = process.env.DEMO_Q || 'What CPU does the Raspberry Pi 4 use?';
const OUT = path.resolve(__dir, process.env.OUT || '../../docs/assets/demo-web.gif');
const W = 820, H = 560, FPS = 15;

const vdir = path.join(__dir, 'videos');
rmSync(vdir, { recursive: true, force: true });
mkdirSync(vdir, { recursive: true });

const browser = await chromium.launch();
const ctx = await browser.newContext({
  viewport: { width: W, height: H },
  deviceScaleFactor: 2,
  recordVideo: { dir: vdir, size: { width: W, height: H } },
});
const page = await ctx.newPage();
await page.goto(PI, { waitUntil: 'networkidle' });
await page.waitForTimeout(800);
await page.fill('#q', Q);
await page.waitForTimeout(500);
await page.click('#f button');

// Server emits sources first (Sources -> Token -> Done), so wait for the panel to populate…
await page.waitForFunction(
  () => (document.querySelector('#sources')?.textContent || '').length > 0,
  null, { timeout: 90000 });
// …then wait until the answer text stops growing (the SSE stream has closed).
await page.waitForFunction(() => {
  const a = document.querySelector('#answer'); if (!a) return false;
  const n = a.textContent.length;
  window.__s = (window.__n === n && n > 0) ? (window.__s || 0) + 1 : 0;
  window.__n = n;
  return window.__s >= 4;            // ~1.6 s stable at 400 ms polling
}, null, { timeout: 90000, polling: 400 });
await page.waitForTimeout(1800);     // let the finished answer rest on screen

const video = page.video();
await ctx.close();                   // flushes the .webm
await browser.close();
const webm = await video.path();

// webm -> high-quality gif via a two-pass palette (palettegen / paletteuse).
// SPEED compresses the Pi's RAG "thinking" gap; CROP_H trims the empty page below the answer.
const SPEED = Number(process.env.SPEED || 2.5);
const CROP_H = Number(process.env.CROP_H || 240);
const pal = path.join(vdir, 'palette.png');
const vf = `setpts=PTS/${SPEED},crop=${W}:${CROP_H}:0:0,fps=${FPS}`;
execFileSync(ffmpegPath, ['-y', '-i', webm, '-vf', `${vf},palettegen=stats_mode=diff`, pal]);
execFileSync(ffmpegPath, ['-y', '-i', webm, '-i', pal, '-lavfi', `${vf}[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=3`, OUT]);
console.log('wrote', OUT);
