// core.js 순수 함수를 파이썬이 낸 정답(tests/fixture.json, scripts/build_teach_web.py)과 대조한다.
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { smooth3, thresholdFrom, verdict, peak, concat, upsample, toCHW, heatRGBA, whereOf, certificate } from '../core.js';

const fx = JSON.parse(readFileSync(new URL('./fixture.json', import.meta.url)));
const close = (a, b, tol = 1e-4) => assert.ok(Math.abs(a - b) <= tol, `${a} vs ${b}`);

test('3×3 평균이 파이썬 avg_pool2d(count_include_pad=False)와 같다', () => {
  const out = smooth3(Float32Array.from(fx.map));
  assert.equal(out.length, 1024);
  out.forEach((v, i) => close(v, fx.smooth3[i]));
});

test('문턱: 통계량 네 가지', () => {
  close(thresholdFrom(fx.loo, { stat: 'min', margin: 1.5 }), fx.thresholds['min_1.5']);
  close(thresholdFrom(fx.loo, { stat: 'median', margin: 1.0 }), fx.thresholds['median_1.0']);
  close(thresholdFrom(fx.loo, { stat: 'mean', margin: 1.2 }), fx.thresholds['mean_1.2']);
  close(thresholdFrom(fx.loo, { stat: 'max', margin: 0.8 }), fx.thresholds['max_0.8']);
  assert.throws(() => thresholdFrom([], { stat: 'min', margin: 1.5 }));
});

test('판정: 문턱과 같으면 다르다고 하지 않는다', () => {
  assert.equal(verdict(2.0, 2.0).differs, false);
  assert.equal(verdict(2.01, 2.0).differs, true);
  close(verdict(3, 2).ratio, 1.5);
});

test('가장 다른 칸의 좌표', () => {
  const m = new Float32Array(1024);
  m[5 * 32 + 17] = 9;
  assert.deepEqual(peak(m), { index: 177, x: 17, y: 5, value: 9 });
});

test('뱅크 잇기: 한 장 빼기', () => {
  const a = [Float32Array.of(1, 2), Float32Array.of(3), Float32Array.of(4, 5)];
  assert.deepEqual([...concat(a)], [1, 2, 3, 4, 5]);
  assert.deepEqual([...concat(a, 1)], [1, 2, 4, 5]);
});

test('키우기가 파이썬 bilinear(align_corners=False)와 같다', () => {
  const up = upsample(Float32Array.from(fx.map), 96);
  for (const [i, v] of fx.upsample96_samples) close(up[i], v, 1e-3);
});

test('RGBA → CHW', () => {
  const chw = toCHW(Uint8ClampedArray.of(255, 0, 51, 255), 1);
  close(chw[0], 1); close(chw[1], 0); close(chw[2], 0.2);
});

test('덧칠: 문턱의 0.7배 아래는 투명, 3배부터는 가장 진하다(반투명)', () => {
  const m = new Float32Array(1024).fill(0.6);
  assert.equal(heatRGBA(m, 32, 1.0)[3], 0);
  assert.equal(heatRGBA(m.fill(3), 32, 1.0)[3], 150);
  assert.equal(heatRGBA(m.fill(9), 32, 1.0)[3], 150);
});

test('위치를 말로: 구역 아홉 개', () => {
  assert.deepEqual(whereOf({ x: 0, y: 0 }), { xPct: 2, yPct: 2, word: '왼쪽 위' });
  assert.equal(whereOf({ x: 16, y: 16 }).word, '한가운데');
  assert.equal(whereOf({ x: 31, y: 16 }).word, '오른쪽');
  assert.equal(whereOf({ x: 16, y: 31 }).word, '아래');
  assert.equal(whereOf({ x: 31, y: 31 }).word, '오른쪽 아래');
});

test('성적서: 다르면 위치를 적고, 같으면 위치를 적지 않는다', () => {
  const info = { id: 'TI-1', issued: 't', shots: 5, aug: 3, loo: [4, 2, 3], rule: { stat: 'min', margin: 1.5 }, model: 'resnet18', ep: 'webgpu' };
  const ng = certificate({ peak: { x: 28, y: 3 }, score: 7.5, threshold: 3, ...verdict(7.5, 3) }, info);
  assert.equal(ng.verdict, '가르친 것과 다름');
  assert.equal(ng.where.word, '오른쪽 위');
  assert.match(ng.reasons[1], /기준은 3\.00입니다.*가장 작은 값\(2\.00\)의 1\.5배/);
  assert.match(ng.reasons[2], /기준의 2\.50배.*오른쪽 위\(왼쪽에서 89%, 위에서 11%\)/);
  assert.match(ng.limits[1], /18%.*19%/);
  const ok = certificate({ peak: { x: 28, y: 3 }, score: 2.4, threshold: 3, ...verdict(2.4, 3) }, info);
  assert.equal(ok.verdict, '가르친 것과 같음');
  assert.equal(ok.where, null);
  assert.match(ok.reasons[2], /기준 안/);
});

test('성적서: 5장보다 적게 가르쳤으면 한계에 적는다', () => {
  const info = { id: 'TI-1', issued: 't', shots: 3, aug: 0, loo: [2, 3, 4], rule: { stat: 'min', margin: 1.5 }, model: 'mobilenet_v3_small', ep: 'wasm' };
  const c = certificate({ peak: { x: 1, y: 1 }, score: 1, threshold: 3, ...verdict(1, 3) }, info);
  assert.equal(c.limits.length, 3);
  assert.match(c.limits[1], /25%.*23%/);
  assert.doesNotMatch(c.reasons[0], /사본/);
});
