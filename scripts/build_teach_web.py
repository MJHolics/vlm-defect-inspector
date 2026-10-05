"""브라우저판(web/)에 들어갈 것을 만든다 — 모델, 예시 사진, 대조용 정답.

- web/models/: teach_mobilenet_v3_small.onnx · teach_resnet18.onnx · knn.onnx
- web/examples/: MVTec AD bottle에서 정상 5장(가르치기) + 시험 6장(정상 2·결함 4), 384px JPEG
- web/examples/expected.json: 같은 JPEG을 파이썬 경로로 돌린 점수·가장 다른 칸(흔들어 늘리기 없이)
- web/tests/fixture.json: core.js 순수 함수 대조용(3×3 평균, 문턱, 키우기)

    python scripts/build_teach_web.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_teach_backbone import export_knn  # noqa: E402
from fewshot_teach_bench import ROOT, SIZE, Backbone  # noqa: E402
from fewshot_teach_rule import web_maps  # noqa: E402

WEB = ROOT / "web"
SRC = ROOT / "data" / "mvtec" / "_full" / "bottle"
TEACH = ["000", "037", "081", "125", "190"]
TEST = [("good", "003"), ("broken_large", "004"), ("contamination", "007"),
        ("good", "011"), ("broken_small", "009"), ("contamination", "015")]
RULE = {"stat": "min", "margin": 1.5}      # reports/fewshot_teach_rule.json 에서 고른 값


def to_tensor(path: Path) -> torch.Tensor:
    im = Image.open(path).convert("RGB").resize((SIZE, SIZE), Image.BILINEAR)
    return torch.from_numpy(np.asarray(im).copy()).permute(2, 0, 1).float().div(255)


def main() -> None:
    (WEB / "models").mkdir(parents=True, exist_ok=True)
    (WEB / "examples").mkdir(parents=True, exist_ok=True)
    (WEB / "tests").mkdir(parents=True, exist_ok=True)

    nets = {}
    for b in ["mobilenet_v3_small", "resnet18"]:
        nets[b] = Backbone(b).eval()
        torch.onnx.export(nets[b], torch.rand(1, 3, SIZE, SIZE), str(WEB / "models" / f"teach_{b}.onnx"),
                          input_names=["image"], output_names=["patches"], opset_version=17, dynamo=False)
    export_knn(WEB / "models")

    names = {"teach": [], "test": []}
    for i, stem in enumerate(TEACH):
        out = WEB / "examples" / f"teach_{i + 1}.jpg"
        Image.open(SRC / "train" / "good" / f"{stem}.png").convert("RGB").resize((384, 384), Image.LANCZOS).save(out, quality=88)
        names["teach"].append(out.name)
    for i, (kind, stem) in enumerate(TEST):
        out = WEB / "examples" / f"test_{i + 1}.jpg"
        Image.open(SRC / "test" / kind / f"{stem}.png").convert("RGB").resize((384, 384), Image.LANCZOS).save(out, quality=88)
        names["test"].append({"file": out.name, "truth": "정상" if kind == "good" else "결함"})

    expected = {"rule": RULE, "teach": names["teach"], "test": names["test"], "models": {}}
    for b, net in nets.items():
        with torch.no_grad():
            tf = net(torch.stack([to_tensor(WEB / "examples" / n) for n in names["teach"]]))
            qf = net(torch.stack([to_tensor(WEB / "examples" / t["file"]) for t in names["test"]]))
        dim = tf.shape[-1]
        loo = [float(web_maps(torch.cat([tf[j].reshape(-1, dim) for j in range(len(tf)) if j != i]), tf[i:i + 1]).max())
               for i in range(len(tf))]
        maps = web_maps(tf.reshape(-1, dim), qf).numpy()
        th = min(loo) * RULE["margin"]
        expected["models"][b] = {"loo": loo, "threshold": th, "results": [
            {"score": float(m.max()), "peak": [int(m.argmax() % 32), int(m.argmax() // 32)], "differs": bool(m.max() > th)}
            for m in maps]}
        print(b, "문턱", round(th, 3), [(t["truth"], round(float(m.max()) / th, 2)) for t, m in zip(names["test"], maps)])
    (WEB / "examples" / "expected.json").write_text(json.dumps(expected, ensure_ascii=False, indent=1), encoding="utf-8")

    rng = np.random.default_rng(0)
    m = rng.random((32, 32), dtype=np.float32) * 5
    sm = F.avg_pool2d(torch.from_numpy(m)[None, None], 3, 1, 1, count_include_pad=False)[0, 0].numpy()
    up = F.interpolate(torch.from_numpy(m)[None, None], size=(96, 96), mode="bilinear", align_corners=False)[0, 0].numpy()
    loo = [3.2, 1.7, 2.4, 2.9, 2.0]
    fixture = {"map": m.reshape(-1).tolist(), "smooth3": sm.reshape(-1).tolist(),
               "upsample96_samples": [[int(i), float(up.reshape(-1)[i])] for i in rng.choice(96 * 96, 200, replace=False)],
               "loo": loo, "thresholds": {"min_1.5": min(loo) * 1.5, "median_1.0": float(np.median(loo)),
                                          "mean_1.2": float(np.mean(loo)) * 1.2, "max_0.8": max(loo) * 0.8}}
    (WEB / "tests" / "fixture.json").write_text(json.dumps(fixture), encoding="utf-8")
    for p in sorted((WEB / "models").glob("*.onnx")):
        print(p.name, f"{p.stat().st_size / 1e6:.2f}MB")


if __name__ == "__main__":
    for _s in (sys.stdout, sys.stderr):
        _s.reconfigure(encoding="utf-8")
    main()
