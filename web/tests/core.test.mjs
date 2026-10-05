// core.js 순수 함수를 파이썬이 낸 정답(tests/fixture.json, scripts/build_teach_web.py)과 대조한다.
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { smooth3, thresholdFrom, verdict, peak, concat, upsample, toCHW, heatRGBA } from '../core.js';

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
