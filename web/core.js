// 판정에 쓰는 순수 계산. 화면·카메라·모델 호출은 app.js에 있고, 여기 함수들은 node에서 그대로 테스트한다.
// 파이썬 쪽 같은 정의: scripts/fewshot_teach_rule.py 의 web_maps (32×32 제곱거리 지도 → 3×3 평균 → 최댓값).

export const SIZE = 256;   // 모델 입력 한 변
export const GRID = 32;    // 거리 지도 한 변

/** RGBA(0~255, size×size) → CHW float32(0~1). 정규화는 모델 그래프 안에서 한다. */
export function toCHW(rgba, size = SIZE) {
  const n = size * size;
  const out = new Float32Array(3 * n);
  for (let i = 0; i < n; i++) {
    out[i] = rgba[4 * i] / 255;
    out[n + i] = rgba[4 * i + 1] / 255;
    out[2 * n + i] = rgba[4 * i + 2] / 255;
  }
  return out;
}

/** 3×3 평균. 가장자리는 있는 칸만으로 나눈다(파이썬 count_include_pad=False). */
export function smooth3(map, w = GRID, h = GRID) {
  const out = new Float32Array(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let s = 0, c = 0;
      for (let dy = -1; dy <= 1; dy++) {
        const yy = y + dy;
        if (yy < 0 || yy >= h) continue;
        for (let dx = -1; dx <= 1; dx++) {
          const xx = x + dx;
          if (xx < 0 || xx >= w) continue;
          s += map[yy * w + xx];
          c++;
        }
      }
      out[y * w + x] = s / c;
    }
  }
  return out;
}

/** 지도에서 가장 다른 칸. x·y는 칸 좌표. */
export function peak(map, w = GRID) {
  let i = 0;
  for (let k = 1; k < map.length; k++) if (map[k] > map[i]) i = k;
  return { index: i, x: i % w, y: Math.floor(i / w), value: map[i] };
}

const STATS = {
  min: (a) => Math.min(...a),
  max: (a) => Math.max(...a),
  mean: (a) => a.reduce((s, v) => s + v, 0) / a.length,
  median: (a) => {
    const s = [...a].sort((p, q) => p - q), m = s.length >> 1;
    return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
  },
};

/** 가르친 사진을 한 장씩 빼고 잰 점수들 → 문턱. rule = {stat, margin}. */
export function thresholdFrom(looScores, rule) {
  if (!looScores.length) throw new Error('가르친 사진이 없습니다');
  return STATS[rule.stat](looScores) * rule.margin;
}

/** 점수와 문턱 → 판정. ratio가 1을 넘으면 가르친 것과 다르다고 본다. */
export function verdict(score, threshold) {
  const ratio = score / threshold;
  return { ratio, differs: ratio > 1 };
}

/** 여러 Float32Array를 하나로 잇는다(뱅크 만들기). skip 번째는 뺀다. */
export function concat(arrays, skip = -1) {
  let n = 0;
  arrays.forEach((a, i) => { if (i !== skip) n += a.length; });
  const out = new Float32Array(n);
  let o = 0;
  arrays.forEach((a, i) => { if (i !== skip) { out.set(a, o); o += a.length; } });
  return out;
}

/** 지도를 out×out으로 키운다(양선형, 칸 중심 기준). 표시용. */
export function upsample(map, out, w = GRID, h = GRID) {
  const res = new Float32Array(out * out);
  for (let y = 0; y < out; y++) {
    const fy = Math.min(h - 1, Math.max(0, (y + 0.5) * h / out - 0.5));
    const y0 = Math.floor(fy), y1 = Math.min(h - 1, y0 + 1), ty = fy - y0;
    for (let x = 0; x < out; x++) {
      const fx = Math.min(w - 1, Math.max(0, (x + 0.5) * w / out - 0.5));
      const x0 = Math.floor(fx), x1 = Math.min(w - 1, x0 + 1), tx = fx - x0;
      res[y * out + x] = (map[y0 * w + x0] * (1 - tx) + map[y0 * w + x1] * tx) * (1 - ty)
        + (map[y1 * w + x0] * (1 - tx) + map[y1 * w + x1] * tx) * ty;
    }
  }
  return res;
}

/** 표시용 덧칠 RGBA. 문턱의 0.7배 아래는 투명, 3배에서 가장 진하다. 밑의 사진이 비쳐 보이게 반투명까지만. */
export function heatRGBA(map, out, threshold, w = GRID, h = GRID) {
  const up = upsample(map, out, w, h);
  const px = new Uint8ClampedArray(out * out * 4);
  for (let i = 0; i < up.length; i++) {
    const t = Math.min(1, Math.max(0, (up[i] / threshold - 0.7) / 2.3));   // 0.7배 → 0, 3배 → 1
    px[4 * i] = 255;
    px[4 * i + 1] = Math.round(200 * (1 - t));
    px[4 * i + 2] = Math.round(40 * (1 - t));
    px[4 * i + 3] = Math.round(150 * Math.sqrt(t));
  }
  return px;
}
