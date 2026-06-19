"""엣지 CNN 정직한 검증 — in-distribution vs 손상 vs 낯선 도메인(OOD).

데모에 올린 그 모델(space/model.onnx, MobileNetV3-S)이 NEU가 아닌 입력에 어떻게
반응하는지 실측한다. 핵심 질문: 낯선 입력에 **신뢰도가 떨어져 0.80 게이트에 걸리나**
(안전), 아니면 **엉뚱한 걸 자신있게 분류하나**(과신 = 정직한 약점)?

  A. NEU test (정상)        — 통제군: 정확도 + 신뢰도 분포
  B. NEU test + 손상         — 노이즈·블러(현장의 지저분한 캡처 시뮬): 정확도 강건성
  C. WM-811K 웨이퍼맵 (OOD)  — 완전히 낯선 도메인: 신뢰도 분포 + 0.80 게이트 통과율
                              (통과율이 높으면 = OOD 과신 = softmax 게이트만으론 부족)

사용:  python scripts/verify_ood_edge.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageFilter

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app import config  # noqa: E402

MODEL = ROOT / "space" / "model.onnx"
CLASSES = config.DEFECT_CLASSES
CLS2IDX = {c: i for i, c in enumerate(CLASSES)}
GATE = 0.80
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_sess = ort.InferenceSession(str(MODEL), providers=["CPUExecutionProvider"])
_name = _sess.get_inputs()[0].name


def _norm_chw(gray01: np.ndarray) -> np.ndarray:
    """(224,224) [0,1] grayscale → (1,3,224,224) ImageNet 정규화."""
    x = np.stack([gray01, gray01, gray01], axis=0)
    x = (x - MEAN[:, None, None]) / STD[:, None, None]
    return x[None].astype(np.float32)


def _from_pil(img: Image.Image, corrupt: bool = False) -> np.ndarray:
    img = img.convert("L").resize((224, 224), Image.BILINEAR)
    if corrupt:
        img = img.filter(ImageFilter.GaussianBlur(radius=1.5))   # 초점 흐림
    x = np.asarray(img, dtype=np.float32) / 255.0
    if corrupt:
        x = np.clip(x + np.random.normal(0, 0.12, x.shape), 0, 1)  # 센서 노이즈
    return _norm_chw(x)


def _from_wafer(m64: np.ndarray) -> np.ndarray:
    """웨이퍼맵 (64,64) {0,1,2} → 파이프라인 입력. 낯선 도메인 그대로 흘려보냄."""
    g = (m64.astype(np.float32) / 2.0)                            # [0,1]
    img = Image.fromarray((g * 255).astype(np.uint8)).resize((224, 224), Image.BILINEAR)
    return _norm_chw(np.asarray(img, dtype=np.float32) / 255.0)


def _softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _run(batch_inputs):
    """list of (1,3,224,224) → (preds, confs)."""
    X = np.concatenate(batch_inputs, axis=0)
    confs, preds = [], []
    for i in range(0, len(X), 64):
        probs = _softmax(_sess.run(None, {_name: X[i:i + 64]})[0])
        preds.append(probs.argmax(1))
        confs.append(probs.max(1))
    return np.concatenate(preds), np.concatenate(confs)


def _stats(confs, gate=GATE):
    c = np.asarray(confs)
    return {"mean_conf": round(float(c.mean()), 4),
            "median_conf": round(float(np.median(c)), 4),
            "pct_pass_gate": round(float((c >= gate).mean()), 4)}


def main():
    recs = json.loads((ROOT / "data" / "processed" / "test.json").read_text(encoding="utf-8"))
    neu = [(str(ROOT / r["image"]),
            CLS2IDX[json.loads(r["conversations"][1]["content"])["type"]]) for r in recs]
    y = np.array([lab for _, lab in neu])
    np.random.seed(42)

    report = {"model": "space/model.onnx (MobileNetV3-S)", "gate": GATE, "buckets": {}}

    # A. NEU 정상
    pa, ca = _run([_from_pil(Image.open(p)) for p, _ in neu])
    report["buckets"]["A_neu_clean"] = {"n": len(neu), "accuracy": round(float((pa == y).mean()), 4),
                                        **_stats(ca)}

    # B. NEU 손상 (블러+노이즈)
    pb, cb = _run([_from_pil(Image.open(p), corrupt=True) for p, _ in neu])
    report["buckets"]["B_neu_corrupted"] = {"n": len(neu), "accuracy": round(float((pb == y).mean()), 4),
                                            **_stats(cb)}

    # C. 웨이퍼맵 OOD
    waf = np.load(ROOT / "data" / "wm811k" / "wafer_prepped.npz")["X_test"]
    idx = np.random.choice(len(waf), size=min(2000, len(waf)), replace=False)
    pc, cc = _run([_from_wafer(waf[i]) for i in idx])
    pred_dist = {CLASSES[k]: int((pc == k).sum()) for k in range(len(CLASSES))}
    report["buckets"]["C_wafer_ood"] = {"n": int(len(idx)), "accuracy": None,
                                        **_stats(cc), "pred_distribution": pred_dist}

    out = ROOT / "data" / "results" / "edge_ood_verify.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 60)
    print(" 엣지 CNN 정직한 검증 (in-dist / 손상 / OOD)")
    print("=" * 60)
    for name, b in report["buckets"].items():
        acc = f"acc {b['accuracy']:.3f}" if b["accuracy"] is not None else "acc  —  (라벨 매핑 불가)"
        print(f" {name:18s} n={b['n']:4d}  {acc}  | 평균신뢰도 {b['mean_conf']:.3f}"
              f"  게이트(≥{GATE})통과 {b['pct_pass_gate']:.1%}")
    print("-" * 60)
    print(f" 저장: {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
