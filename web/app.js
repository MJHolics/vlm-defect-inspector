// 찍어서 가르치는 검사기 — 화면·카메라·모델 호출. 판정 계산은 core.js(순수 함수, node 테스트 대상).
import { SIZE, GRID, toCHW, smooth3, peak, thresholdFrom, verdict, concat, heatRGBA, certificate } from './core.js';

const RULE = { stat: 'min', margin: 1.5 };   // reports/fewshot_teach_rule.json: dev 5종에서 고른 값
const AUG = 3;                                // 가르친 사진 한 장을 돌리고 밀어 3장 더 만든다(±15°, ±5%)
const MIN_SHOTS = 3, GOAL_SHOTS = 5, MAX_SHOTS = 12;
const VIEW = 512;
const params = new URLSearchParams(location.search);
const $ = (id) => document.getElementById(id);

// ───────────────────────── 모델 ─────────────────────────
class Engine {
  async init() {
    ort.env.wasm.numThreads = 1;              // GitHub Pages는 교차 출처 격리가 아니라 스레드를 못 쓴다
    const want = params.get('model');         // small | large (없으면 자동)
    const tries = [];
    if (want !== 'small' && navigator.gpu) tries.push(['resnet18', 'webgpu']);
    if (want === 'large') tries.push(['resnet18', 'wasm']);
    tries.push(['mobilenet_v3_small', 'wasm']);
    for (const [name, ep] of tries) {
      try {
        const opt = { executionProviders: [ep] };
        this.backbone = await ort.InferenceSession.create(`models/teach_${name}.onnx`, opt);
        this.knn = await ort.InferenceSession.create('models/knn.onnx', opt);
        this.name = name; this.ep = ep;
        const f = await this.features(blankCanvas());      // 한 번 돌려 보고 실패하면 다음 후보로
        this.dim = f.length / (GRID * GRID);
        await this.distances(f, f);
        return;
      } catch (e) { console.warn('모델 후보 실패', name, ep, e); }
    }
    throw new Error('모델을 불러오지 못했습니다');
  }

  /** 256×256 캔버스 → 패치 특징 (1024 × dim). */
  async features(canvas) {
    const rgba = canvas.getContext('2d', { willReadFrequently: true }).getImageData(0, 0, SIZE, SIZE).data;
    const t0 = performance.now();
    const out = (await this.backbone.run({ image: new ort.Tensor('float32', toCHW(rgba), [1, 3, SIZE, SIZE]) })).patches;
    const data = out.getData ? await out.getData() : out.data;
    this.featMs = performance.now() - t0;
    return Float32Array.from(data);
  }

  /** 패치별로 뱅크에서 가장 가까운 것까지의 제곱거리 (1024). */
  async distances(q, bank) {
    const d = this.dim ?? q.length / (GRID * GRID);
    const t0 = performance.now();
    const out = (await this.knn.run({
      q: new ort.Tensor('float32', q, [q.length / d, d]),
      bank: new ort.Tensor('float32', bank, [bank.length / d, d]),
    })).d;
    const data = out.getData ? await out.getData() : out.data;
    this.knnMs = performance.now() - t0;
    return Float32Array.from(data);
  }
}

function blankCanvas() {
  const c = document.createElement('canvas');
  c.width = c.height = SIZE;
  const g = c.getContext('2d');
  g.fillStyle = '#777'; g.fillRect(0, 0, SIZE, SIZE);
  return c;
}

/** 아무 그림 원본(비디오·비트맵·캔버스)의 가운데 정사각형을 256 캔버스로. */
function squareCanvas(src, sw, sh) {
  const c = document.createElement('canvas');
  c.width = c.height = SIZE;
  const s = Math.min(sw, sh);
  const g = c.getContext('2d');
  g.imageSmoothingQuality = 'high';
  g.drawImage(src, (sw - s) / 2, (sh - s) / 2, s, s, 0, 0, SIZE, SIZE);
  return c;
}

/** 돌리고 민 사본. 빈 자리는 모서리 색으로 채운다(검은 모서리가 "다른 곳"으로 잡히지 않게). */
function jittered(canvas, rnd) {
  const c = document.createElement('canvas');
  c.width = c.height = SIZE;
  const g = c.getContext('2d');
  const p = canvas.getContext('2d', { willReadFrequently: true }).getImageData(0, 0, 1, 1).data;
  g.fillStyle = `rgb(${p[0]},${p[1]},${p[2]})`;
  g.fillRect(0, 0, SIZE, SIZE);
  g.translate(SIZE / 2 + (rnd() * 2 - 1) * 0.05 * SIZE, SIZE / 2 + (rnd() * 2 - 1) * 0.05 * SIZE);
  g.rotate((rnd() * 2 - 1) * 15 * Math.PI / 180);
  g.drawImage(canvas, -SIZE / 2, -SIZE / 2);
  return c;
}

function seeded(seed) {           // mulberry32 — 같은 사진이면 같은 사본이 나오게
  let a = seed >>> 0;
  return () => {
    a |= 0; a = a + 0x6D2B79F5 | 0;
    let t = Math.imul(a ^ a >>> 15, 1 | a);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}

// ───────────────────────── 가르친 것(뱅크)과 판정 ─────────────────────────
class Teacher {
  constructor(engine, aug = AUG) { this.engine = engine; this.aug = aug; this.reset(); }
  reset() { this.shots = []; this.loo = []; this.threshold = null; }
  get ready() { return this.shots.length >= MIN_SHOTS; }

  async teach(canvas) {
    if (this.shots.length >= MAX_SHOTS) throw new Error(`가르칠 수 있는 사진은 ${MAX_SHOTS}장까지입니다`);
    const rnd = seeded(1000 + this.shots.length);
    const feats = [await this.engine.features(canvas)];
    for (let i = 0; i < this.aug; i++) feats.push(await this.engine.features(jittered(canvas, rnd)));
    this.shots.push({ canvas, own: feats[0], all: concat(feats) });
    await this.calibrate();
  }

  /** 한 장씩 빼고 나머지에 대 본 점수 → 문턱. */
  async calibrate() {
    this.loo = []; this.threshold = null;
    if (!this.ready) return;
    const all = this.shots.map((s) => s.all);
    for (let i = 0; i < this.shots.length; i++) {
      const d = await this.engine.distances(this.shots[i].own, concat(all, i));
      this.loo.push(peak(smooth3(d)).value);
    }
    this.threshold = thresholdFrom(this.loo, RULE);
  }

  async inspect(canvas) {
    if (!this.ready) throw new Error(`정상 사진을 ${MIN_SHOTS}장 이상 가르쳐야 합니다`);
    const q = await this.engine.features(canvas);
    const map = smooth3(await this.engine.distances(q, concat(this.shots.map((s) => s.all))));
    const pk = peak(map);
    return { canvas, map, peak: pk, score: pk.value, threshold: this.threshold, ...verdict(pk.value, this.threshold) };
  }
}

// ───────────────────────── 화면 ─────────────────────────
const stage = $('stage'), ctx = stage.getContext('2d');
const video = $('video');
const engine = new Engine();
let teacher;
let mode = 'demo';            // demo | live
let still = null;             // live에서 카메라 대신 보고 있는 사진(256 캔버스)
let lastResult = null;
let demoRun = 0;              // 시연 재생을 끊을 때 올린다
let busy = false;

function drawBase(canvas, g = ctx) {
  g.imageSmoothingQuality = 'high';
  g.drawImage(canvas, 0, 0, VIEW, VIEW);
}

function drawGuide() {
  const m = VIEW * 0.1, r = 28;
  ctx.save();
  ctx.strokeStyle = 'rgba(255,255,255,.75)'; ctx.lineWidth = 3; ctx.setLineDash([14, 10]);
  ctx.beginPath(); ctx.roundRect(m, m, VIEW - 2 * m, VIEW - 2 * m, r); ctx.stroke();
  ctx.restore();
}

function drawResult(res, g = ctx) {
  drawBase(res.canvas, g);
  const heat = new ImageData(heatRGBA(res.map, VIEW, res.threshold), VIEW, VIEW);
  const layer = document.createElement('canvas');
  layer.width = layer.height = VIEW;
  layer.getContext('2d').putImageData(heat, 0, 0);
  g.drawImage(layer, 0, 0);
  if (res.differs) {
    const cx = (res.peak.x + 0.5) * VIEW / GRID, cy = (res.peak.y + 0.5) * VIEW / GRID;
    g.save();
    g.lineWidth = 5; g.strokeStyle = '#fff';
    g.beginPath(); g.arc(cx, cy, 34, 0, Math.PI * 2); g.stroke();
    g.lineWidth = 3; g.strokeStyle = '#ff3b4e';
    g.beginPath(); g.arc(cx, cy, 34, 0, Math.PI * 2); g.stroke();
    g.restore();
  }
}

function setBanner(kind, title, why) {
  $('stagebox').dataset.kind = kind;
  $('verdict').textContent = title;
  $('why').textContent = why;
}

function renderShots() {
  const box = $('shots');
  box.innerHTML = '';
  for (let i = 0; i < Math.max(GOAL_SHOTS, teacher.shots.length); i++) {
    const cell = document.createElement('div');
    cell.className = 'shot';
    if (teacher.shots[i]) {
      const c = document.createElement('canvas');
      c.width = c.height = 64;
      c.getContext('2d').drawImage(teacher.shots[i].canvas, 0, 0, 64, 64);
      cell.append(c);
    } else cell.textContent = i + 1;
    box.append(cell);
  }
  $('count').textContent = `${teacher.shots.length}장`;
}

function renderPanel() {
  renderShots();
  const n = teacher.shots.length;
  $('btnTeach').textContent = n < GOAL_SHOTS ? `이게 정상 (${n}/${GOAL_SHOTS})` : `정상 더 가르치기 (${n}장)`;
  $('btnInspect').disabled = !teacher.ready || busy;
  $('btnTeach').disabled = busy || n >= MAX_SHOTS || (!!lastResult && !still);   // 결과를 보는 중에는 「다음 물건」부터
  $('btnAlso').hidden = !(mode === 'live' && lastResult && lastResult.differs);
  $('btnAgain').hidden = !(mode === 'live' && (still || lastResult));
  $('btnCert').hidden = !lastResult;
  const r = lastResult;
  const ratio = r ? r.ratio : 0;
  $('gaugeFill').style.width = `${Math.min(100, ratio / 3 * 100)}%`;
  $('gaugeFill').dataset.over = r && r.differs ? '1' : '0';
  $('gaugeVal').textContent = r ? `기준의 ${ratio.toFixed(2)}배` : '—';
  $('engine').textContent = engine.name
    ? `${engine.name === 'resnet18' ? 'ResNet-18(11MB)' : 'MobileNetV3-small(0.8MB)'} · ${engine.ep === 'webgpu' ? 'WebGPU' : 'WASM'}`
      + (engine.featMs ? ` · 특징 ${engine.featMs.toFixed(0)}ms · 비교 ${(engine.knnMs ?? 0).toFixed(0)}ms` : '')
    : '모델을 불러오는 중';
  $('steps').dataset.step = !teacher.ready ? '1' : r ? '3' : '2';
}

function explain(res) {
  if (res.differs) return ['다른 곳이 있습니다', `가르친 사진들과 가장 다른 곳을 표시했습니다. 다른 정도는 기준의 ${res.ratio.toFixed(1)}배입니다.`];
  return ['가르친 것과 같아 보입니다', `가장 다른 곳도 기준의 ${res.ratio.toFixed(2)}배로, 기준 안입니다.`];
}

async function showResult(canvas) {
  const res = await teacher.inspect(canvas);
  res.shots = teacher.shots.map((s) => s.canvas);      // 성적서용: 이 판정에 쓴 정상 사진과 기준
  res.loo = [...teacher.loo];
  lastResult = res;
  if (mode === 'live') log.push({ at: new Date(), differs: res.differs, ratio: res.ratio });
  drawResult(res);
  const [t, w] = explain(res);
  setBanner(res.differs ? 'ng' : 'ok', t, w);
  renderPanel();
  return res;
}

// ───────────────────────── 검사 성적서 ─────────────────────────
// 서버 없이 이 화면에서 만든다. PDF는 브라우저의 인쇄(「PDF로 저장」)로 낸다.
const log = [];               // 직접 해 보기에서 한 검사(이번 방문 동안만)
const stamp = (d) => new Intl.DateTimeFormat('sv-SE', { dateStyle: 'short', timeStyle: 'medium' }).format(d);

function certId(d) {
  const r = crypto.getRandomValues(new Uint8Array(3));
  return `TI-${stamp(d).slice(0, 10).replaceAll('-', '')}-${[...r].map((v) => v.toString(16).padStart(2, '0')).join('').toUpperCase()}`;
}

function openCert(res = lastResult) {
  if (!res) return null;
  const now = new Date();
  const c = certificate(res, { id: certId(now), issued: stamp(now), shots: res.shots.length, aug: teacher.aug,
    loo: res.loo, rule: RULE, model: engine.name, ep: engine.ep });
  drawResult(res, $('certImg').getContext('2d'));
  $('certId').textContent = c.id;
  $('certIssued').textContent = c.issued;
  $('certModel').textContent = c.model;
  $('certSource').textContent = mode === 'demo' ? '시연 재생(예시 병 사진)' : '직접 해 보기';
  $('certVerdict').textContent = c.verdict;
  $('certSheet').dataset.kind = c.differs ? 'ng' : 'ok';
  $('certRatio').textContent = `기준의 ${c.ratio.toFixed(2)}배` + (c.where ? ` · ${c.where.word}` : '');
  const fill = (id, items) => { $(id).replaceChildren(...items.map((t) => Object.assign(document.createElement('li'), { textContent: t }))); };
  fill('certReasons', c.reasons);
  fill('certLimits', c.limits);
  $('certShots').replaceChildren(...res.shots.map((s) => {
    const t = document.createElement('canvas');
    t.width = t.height = 96;
    t.getContext('2d').drawImage(s, 0, 0, 96, 96);
    return t;
  }));
  const rows = mode === 'live' ? log.slice(-10) : [];
  $('certLogBox').hidden = rows.length < 2;
  $('certLogSum').textContent = `이번 방문에서 검사 ${log.length}건, 그중 다름 ${log.filter((r) => r.differs).length}건`;
  $('certLog').replaceChildren(...rows.map((r) => {
    const tr = document.createElement('tr');
    for (const t of [stamp(r.at).slice(11), r.differs ? '다름' : '같음', `${r.ratio.toFixed(2)}배`]) tr.append(Object.assign(document.createElement('td'), { textContent: t }));
    return tr;
  }));
  $('cert').hidden = false;
  $('cert').scrollTop = 0;
  return c;
}

// ───────────────────────── 시연 재생(들어오면 저절로) ─────────────────────────
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function loadCanvas(url) {
  const bmp = await createImageBitmap(await (await fetch(url)).blob());   // Image.decode()는 가려진 탭에서 멈춘다
  return squareCanvas(bmp, bmp.width, bmp.height);
}

async function runDemo() {
  const my = ++demoRun;
  mode = 'demo'; still = null; lastResult = null;
  $('tag').textContent = '시연 재생 — 예시 병 사진';
  const ex = await (await fetch('examples/expected.json')).json();
  while (my === demoRun) {
    teacher.reset(); lastResult = null; renderPanel();
    for (let i = 0; i < ex.teach.length && my === demoRun; i++) {
      const c = await loadCanvas(`examples/${ex.teach[i]}`);
      if (my !== demoRun) return;
      drawBase(c); drawGuide();
      setBanner('teach', `정상 ${i + 1}/${ex.teach.length}`, '멀쩡한 병을 한 장씩 보여 주며 가르칩니다.');
      await teacher.teach(c);
      if (my !== demoRun) return;
      renderPanel();
      await sleep(650);
    }
    for (const t of ex.test) {
      if (my !== demoRun) return;
      const c = await loadCanvas(`examples/${t.file}`);
      if (my !== demoRun) return;
      await showResult(c);
      $('caption').textContent = `이 사진의 실제: ${t.truth}`;
      await sleep(2600);
    }
    $('caption').textContent = '';
    await sleep(600);
  }
}

// ───────────────────────── 직접 해 보기 ─────────────────────────
let stream = null, liveTimer = null;

function liveLoop() {
  clearInterval(liveTimer);
  liveTimer = setInterval(() => {
    if (mode !== 'live' || still || lastResult || !stream || video.readyState < 2) return;
    drawBase(squareCanvas(video, video.videoWidth, video.videoHeight));
    drawGuide();
  }, 66);
}

async function startLive(withCamera) {
  demoRun++;
  mode = 'live'; still = null; lastResult = null;
  teacher.reset();
  $('caption').textContent = '';
  $('tag').textContent = '직접 해 보기';
  $('live').hidden = false; $('btnDemo').hidden = false; $('btnCamera').hidden = true; $('btnPhotos').hidden = true;
  log.length = 0;
  setBanner('teach', '멀쩡한 것을 먼저 보여 주세요', `점선 안에 물건을 놓고 「이게 정상」을 ${GOAL_SHOTS}번 누릅니다. 매번 같은 자리, 같은 방향으로.`);
  ctx.fillStyle = '#05080b'; ctx.fillRect(0, 0, VIEW, VIEW); drawGuide();
  renderPanel();
  if (!withCamera) return;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: { ideal: 'environment' }, width: { ideal: 1280 }, height: { ideal: 1280 } }, audio: false });
    video.srcObject = stream;
    await video.play();
    liveLoop();
  } catch (e) {
    $('note').textContent = '카메라를 열지 못했습니다. 아래 「사진 고르기」나 예시 사진으로 해 볼 수 있습니다.';
    $('note').classList.add('error');
  }
}

function stopCamera() {
  clearInterval(liveTimer);
  if (stream) stream.getTracks().forEach((t) => t.stop());
  stream = null;
}

function currentCanvas() {
  if (still) return still;
  if (stream && video.readyState >= 2) return squareCanvas(video, video.videoWidth, video.videoHeight);
  return null;
}

async function guarded(fn) {
  if (busy) return;
  busy = true; renderPanel();
  try { await fn(); } catch (e) { $('note').textContent = e.message; $('note').classList.add('error'); }
  busy = false; renderPanel();
}

/** fromResult: 방금 검사한 사진을 정상으로 더 가르친다(「이것도 정상입니다」). 큰 버튼은 지금 보이는 장면만 가르친다. */
async function doTeach(fromResult = false) {
  if (mode !== 'live') await startLive(false);
  const c = fromResult === true && lastResult ? lastResult.canvas : currentCanvas();
  if (!c) { $('note').textContent = '카메라를 켜거나 사진을 골라 주세요.'; return; }
  await guarded(async () => {
    await teacher.teach(c);
    still = null; lastResult = null;
    const n = teacher.shots.length;
    setBanner('teach', `정상 ${n}장을 배웠습니다`, n < MIN_SHOTS ? `${MIN_SHOTS - n}장만 더 가르치면 검사할 수 있습니다.`
      : n < GOAL_SHOTS ? '검사할 수 있습니다. 5장까지 가르치면 더 안정적입니다.' : '이제 다른 물건을 놓고 「검사」를 누르세요.');
    if (!stream) { drawBase(c); drawGuide(); }
  });
}

async function doInspect() {
  const c = currentCanvas();
  if (!c) { $('note').textContent = '카메라를 켜거나 사진을 골라 주세요.'; return; }
  await guarded(async () => { await showResult(c); still = null; });
}

async function useStill(canvas) {
  if (mode !== 'live') await startLive(false);
  still = canvas; lastResult = null;
  drawBase(canvas); drawGuide();
  setBanner('teach', '이 사진으로', teacher.ready ? '「검사」 또는 「이게 정상」을 누르세요.' : '「이게 정상」을 눌러 가르치세요.');
  renderPanel();
}

function again() {
  still = null; lastResult = null;
  setBanner('teach', teacher.ready ? '검사할 물건을 놓으세요' : '멀쩡한 것을 먼저 보여 주세요', '점선 안에, 가르칠 때와 같은 자리·같은 방향으로.');
  if (!stream) { ctx.fillStyle = '#05080b'; ctx.fillRect(0, 0, VIEW, VIEW); drawGuide(); }
  renderPanel();
}

// ───────────────────────── 말로 하기 ─────────────────────────
let recog = null;
function toggleVoice() {
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SR) { $('heard').textContent = '이 브라우저는 음성 인식을 지원하지 않습니다. 버튼으로 해 주세요.'; return; }
  if (recog) { recog.onend = null; recog.stop(); recog = null; $('btnVoice').textContent = '말로 하기'; $('heard').textContent = ''; return; }
  recog = new SR();
  recog.lang = 'ko-KR'; recog.continuous = true; recog.interimResults = false;
  recog.onresult = (e) => {
    const said = e.results[e.results.length - 1][0].transcript.trim();
    $('heard').textContent = `들은 말: "${said}"`;
    if (/처음|다시/.test(said)) $('btnReset').click();
    else if (/성적서/.test(said)) openCert();
    else if (/검사/.test(said)) doInspect();
    else if (/정상/.test(said)) doTeach(/이것도/.test(said));
  };
  recog.onerror = (e) => { $('heard').textContent = `음성 인식 오류: ${e.error}`; };
  recog.onend = () => { if (recog) recog.start(); };
  recog.start();
  $('btnVoice').textContent = '말로 하기 끄기';
  $('heard').textContent = '듣는 중 — "이게 정상", "검사", "성적서", "처음부터"';
}

// ───────────────────────── 대조(?selftest=1) ─────────────────────────
// 예시 사진을 흔들어 늘리기 없이 돌려 파이썬이 낸 값(examples/expected.json)과 견준다.
async function selftest() {
  const ex = await (await fetch('examples/expected.json')).json();
  const ref = ex.models[engine.name];
  const t = new Teacher(engine, 0);
  for (const f of ex.teach) await t.teach(await loadCanvas(`examples/${f}`));
  const rows = [];
  for (let i = 0; i < ex.test.length; i++) {
    const r = await t.inspect(await loadCanvas(`examples/${ex.test[i].file}`));
    const e = ref.results[i];
    rows.push({ file: ex.test[i].file, truth: ex.test[i].truth, score: r.score, expected: e.score,
      rel: Math.abs(r.score - e.score) / e.score, peak: [r.peak.x, r.peak.y], expectedPeak: e.peak,
      differs: r.differs, expectedDiffers: e.differs });
  }
  const out = { model: engine.name, ep: engine.ep, threshold: t.threshold, expectedThreshold: ref.threshold,
    maxRel: Math.max(...rows.map((r) => r.rel)), verdictsMatch: rows.every((r) => r.differs === r.expectedDiffers),
    peaksWithin1: rows.filter((r) => r.expectedDiffers).every((r) => Math.abs(r.peak[0] - r.expectedPeak[0]) <= 1 && Math.abs(r.peak[1] - r.expectedPeak[1]) <= 1),
    rows };
  window.__selftest = out;
  $('note').textContent = `대조: 판정 ${out.verdictsMatch ? '일치' : '불일치'} · 점수 최대 상대차 ${(out.maxRel * 100).toFixed(1)}% · 가장 다른 칸 ${out.peaksWithin1 ? '1칸 이내' : '어긋남'}`;
  return out;
}

// ───────────────────────── 시작 ─────────────────────────
function buildExampleStrip(ex) {
  const box = $('examples');
  for (const f of [...ex.teach, ...ex.test.map((t) => t.file)]) {
    const b = document.createElement('button');
    b.className = 'ex';
    b.innerHTML = `<img src="examples/${f}" alt="예시 병 사진" loading="lazy">`;
    b.onclick = async () => useStill(await loadCanvas(`examples/${f}`));
    box.append(b);
  }
}

async function main() {
  ctx.fillStyle = '#05080b'; ctx.fillRect(0, 0, VIEW, VIEW);
  setBanner('teach', '준비 중', '모델을 불러옵니다.');
  try { await engine.init(); } catch (e) { setBanner('ng', '열 수 없습니다', e.message); return; }
  teacher = new Teacher(engine);
  window.__teach = { engine, teacher, selftest, runDemo, startLive, useStill, doTeach, doInspect, loadCanvas, openCert, log, get lastResult() { return lastResult; } };
  renderPanel();
  $('btnCamera').onclick = () => startLive(true);
  $('btnPhotos').onclick = () => startLive(false);
  $('btnDemo').onclick = () => { stopCamera(); $('live').hidden = true; $('btnDemo').hidden = true; $('btnCamera').hidden = false; $('btnPhotos').hidden = false; runDemo(); };
  $('btnTeach').onclick = () => doTeach(false);
  $('btnInspect').onclick = doInspect;
  $('btnAlso').onclick = () => doTeach(true);
  $('btnAgain').onclick = again;
  $('btnReset').onclick = () => { teacher.reset(); again(); };
  $('btnVoice').onclick = toggleVoice;
  $('btnCert').onclick = () => openCert();
  $('btnCertClose').onclick = () => { $('cert').hidden = true; };
  $('btnCertPrint').onclick = () => window.print();
  $('file').onchange = async (e) => {
    const f = e.target.files[0];
    if (!f) return;
    const bmp = await createImageBitmap(f);
    await useStill(squareCanvas(bmp, bmp.width, bmp.height));
    e.target.value = '';
  };
  buildExampleStrip(await (await fetch('examples/expected.json')).json());
  if (params.get('selftest')) { await selftest(); return; }
  runDemo();
}

main();
