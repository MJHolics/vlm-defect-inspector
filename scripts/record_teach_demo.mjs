// 검사기 시연 재생을 휴대폰 폭으로 캡처한다. 프레임마다 시각을 적어 ffmpeg concat으로 잇는다.
// 준비: npm i playwright-core (설치된 Chrome을 쓴다) · cd web && python -m http.server 8792
// 실행: node scripts/record_teach_demo.mjs http://127.0.0.1:8792/ <프레임 폴더>
// 묶기: ffmpeg -f concat -safe 0 -i <프레임 폴더>/list.txt -vf "fps=20,scale=780:-2,format=yuv420p" -c:v libx264 -crf 26 web/docs/demo.mp4
//       GIF는 fps=8,scale=390:-1 뒤 palettegen(128색)·paletteuse
import { chromium } from 'playwright-core';
import fs from 'node:fs';
import path from 'node:path';

const [url, outDir] = process.argv.slice(2);
fs.rmSync(outDir, { recursive: true, force: true });
fs.mkdirSync(outDir, { recursive: true });

const browser = await chromium.launch({ channel: 'chrome', headless: true });
const ctx = await browser.newContext({ viewport: { width: 390, height: 780 }, deviceScaleFactor: 2, isMobile: true, hasTouch: true, locale: 'ko-KR' });
const page = await ctx.newPage();
page.on('pageerror', (e) => console.log('pageerror:', e.message));
page.on('console', (m) => { if (m.type() === 'error') console.log('console.error:', m.text()); });
await page.goto(url);
await page.waitForFunction(() => window.__teach && document.getElementById('verdict').textContent !== '준비 중', null, { timeout: 60000 });
// 돌고 있는 재생은 건드리지 않고, 한 바퀴의 처음(정상 1/5)이 돌아올 때까지 기다린다
await page.waitForFunction(() => document.getElementById('stagebox').dataset.kind !== 'teach', null, { timeout: 60000, polling: 50 });
await page.waitForFunction(() => document.getElementById('verdict').textContent.startsWith('정상 1/'), null, { timeout: 60000, polling: 30 });

const frames = [];
const t0 = Date.now();
let n = 0;
async function shoot() {
  const f = path.join(outDir, `f${String(n++).padStart(4, '0')}.jpg`);
  const t = Date.now() - t0;
  await page.screenshot({ path: f, type: 'jpeg', quality: 92 });
  frames.push({ f, t });
}
async function shootFor(ms) { const end = Date.now() + ms; while (Date.now() < end) await shoot(); }

// 가르치기 5장 → 검사 6장 중 "다른 곳" 결과가 떠 있을 때까지
const seen = [];
let ngAt = null;
const limit = Date.now() + 40000;
while (Date.now() < limit) {
  await shoot();
  const s = await page.evaluate(() => ({ v: document.getElementById('verdict').textContent, c: document.getElementById('caption').textContent, k: document.getElementById('stagebox').dataset.kind }));
  const key = `${s.k}|${s.v}|${s.c}`;
  if (seen.at(-1) !== key) { seen.push(key); console.log(((Date.now() - t0) / 1000).toFixed(1), key); }
  const results = seen.filter((k) => !k.startsWith('teach'));
  if (results.length >= 4 && s.k === 'ng' && ngAt === null) ngAt = Date.now();
  if (ngAt && Date.now() - ngAt > 1800) break;
}
// 그 결과로 성적서
await page.evaluate(() => window.__teach.openCert());
await shootFor(2200);
const h = await page.evaluate(() => { const c = document.getElementById('cert'); return c.scrollHeight - c.clientHeight; });
for (let i = 1; i <= 24; i++) { await page.evaluate((y) => { document.getElementById('cert').scrollTop = y; }, Math.round(h * i / 24)); await shoot(); }
await shootFor(2200);
const end = Date.now() - t0;

let list = '';
frames.forEach((fr, i) => { const d = ((frames[i + 1]?.t ?? end) - fr.t) / 1000; list += `file '${fr.f.replaceAll('\\', '/')}'\nduration ${d.toFixed(3)}\n`; });
list += `file '${frames.at(-1).f.replaceAll('\\', '/')}'\n`;
fs.writeFileSync(path.join(outDir, 'list.txt'), list);
console.log(`frames ${frames.length}, ${(end / 1000).toFixed(1)}s, cert scroll ${h}px`);
await browser.close();
