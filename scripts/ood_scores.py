"""전용 OOD(신규 결함) 점수 비교 — 생성 confidence를 넘어서 (GPU 필요).

배경: eval_ood.py는 생성 confidence(토큰 로그확률 기하평균)만으로 신규 결함을
가렸고, AUROC 0.68 / 임계값 0.80서 적발 33%에 그쳤다(생성 confidence는 OOD에
과신 → 신뢰할 탐지기 아님). 이 스크립트는 같은 holdout 어댑터·같은 test셋에서
'전용' OOD 점수 네 가지를 한꺼번에 뽑아 confidence 기준선과 공정 비교한다:

  1) confidence (MSP, 기준선)  : 1 - 생성토큰 로그확률 기하평균
  2) energy                    : 생성토큰별 -logsumexp(logits) 평균 (높을수록 OOD)
  3) entropy                   : 생성토큰별 분포 엔트로피 평균   (높을수록 OOD)
  4) mahalanobis (feature)     : 마지막 레이어·마지막 프롬프트 토큰 hidden state를
       known 클래스 train 특징으로 적합한 PCA+수축공분산 공간에서의 최소 클래스
       마할라노비스 거리 (높을수록 OOD). known-only 적합이라 test 누수 없음.
  5) ensemble                  : 위 네 점수를 test 전체에서 z정규화 후 평균(비지도).

비교 지표(클래스 라벨이 신규/기존인지로 채점):
  - AUROC                : 점수가 신규 vs 기존을 분리하는 능력
  - TPR@FPR5%            : 기존 오경보 5%만 허용할 때의 신규 적발률(운영 관점)
  - FPR95               : 신규 95% 적발 시 끌려오는 기존 오경보율 (낮을수록 좋음)

GPU 메모리 보호: generate는 scores만(hidden state 미요청), 특징은 별도 forward로
분리해 뽑는다(adapter 학습 때 본 표현과 동일 정의).

사용 (GPU):
    python scripts/ood_scores.py --adapter models/checkpoints/ood_no_inclusion \
        --holdout-class inclusion --fit-per-class 40
"""
import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
DATA = ROOT / "data" / "processed"


def _gt_type(ex: dict) -> str:
    for t in ex.get("conversations", []):
        if t.get("role") == "assistant":
            try:
                return (json.loads(t["content"]).get("type") or "").lower()
            except Exception:
                pass
    return ""


def _load_img(rec: dict) -> Image.Image:
    p = Path(rec["image"])
    ap = (ROOT / p) if not p.is_absolute() else p
    return Image.open(ap).convert("RGB")


def main():
    ap = argparse.ArgumentParser(description="전용 OOD 점수 비교")
    ap.add_argument("--adapter", required=True, help="holdout 학습된 LoRA 어댑터")
    ap.add_argument("--holdout-class", required=True, help="학습에서 제외된 신규 결함 유형")
    ap.add_argument("--testset", type=Path, default=DATA / "test.json")
    ap.add_argument("--trainset", type=Path, default=DATA / "train.json",
                    help="Mahalanobis 적합용 known-클래스 특징 출처")
    ap.add_argument("--fit-per-class", type=int, default=40,
                    help="클래스당 적합 표본 수(known 클래스만)")
    ap.add_argument("--pca-dim", type=int, default=64)
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "results" / "ood_scores.csv")
    ap.add_argument("--json-out", type=Path, default=ROOT / "data" / "results" / "ood_scores.json")
    ap.add_argument("--limit-test", type=int, default=None, help="스모크 테스트용")
    args = ap.parse_args()

    from app import config
    from app import main as appmain
    appmain.LORA_PATH = Path(args.adapter)
    appmain.USE_LORA = appmain.LORA_PATH.exists()
    if not appmain.USE_LORA:
        raise SystemExit(f"어댑터 경로가 없습니다: {args.adapter}")
    print(f"어댑터 로드: {args.adapter}  (신규 결함 = '{args.holdout_class}')")
    appmain.load_model()
    model, processor = appmain.model, appmain.processor
    device = model.device
    novel = args.holdout_class.lower()

    # ── 입력 빌더(appmain 프롬프트 재사용) ──────────────
    def _make_inputs(img: Image.Image):
        messages = [
            {"role": "system", "content": appmain.SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": appmain.INFERENCE_PROMPT},
            ]},
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return processor(text=[text], images=[img], return_tensors="pt", padding=True).to(device)

    # ── generate 1회: confidence/energy/entropy + pred ──
    def gen_signals(img: Image.Image):
        inputs = _make_inputs(img)
        plen = inputs["input_ids"].shape[1]
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=256, do_sample=False,
                temperature=None, top_p=None,
                output_scores=True, return_dict_in_generate=True,
            )
        gen = out.sequences[0, plen:]
        logps, energies, entropies = [], [], []
        for s, t in zip(out.scores, gen):
            logits = s[0].float()
            lse = torch.logsumexp(logits, dim=-1)
            logp = logits - lse  # = log_softmax
            energies.append((-lse).item())
            entropies.append((-(logp.exp() * logp).sum()).item())
            logps.append(logp[t].item())
        n = max(len(logps), 1)
        conf = math.exp(sum(logps) / n) if logps else 0.0
        raw = processor.batch_decode([gen], skip_special_tokens=True)[0].strip()
        pred = appmain._normalize_type(appmain.parse_output(raw)) or ""
        return conf, sum(energies) / n, sum(entropies) / n, pred

    # ── forward 1회: 특징(마지막 레이어·마지막 프롬프트 토큰) ──
    def feature(img: Image.Image) -> np.ndarray:
        inputs = _make_inputs(img)
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True, use_cache=False)
        return out.hidden_states[-1][0, -1, :].float().cpu().numpy()

    # ── Mahalanobis 적합: known 클래스 train 특징 ───────
    print(f"[fit] Mahalanobis 적합 특징 추출 (클래스당 {args.fit_per_class}, known만)...")
    train = json.loads(args.trainset.read_text(encoding="utf-8"))
    by_class: dict[str, list] = {}
    for ex in train:
        gt = _gt_type(ex)
        if gt and gt != novel:
            by_class.setdefault(gt, []).append(ex)
    fit_feats, fit_labels = [], []
    for cls, exs in sorted(by_class.items()):
        for ex in exs[: args.fit_per_class]:
            try:
                fit_feats.append(feature(_load_img(ex)))
                fit_labels.append(cls)
            except Exception as e:
                print(f"  [skip] {ex.get('id')}: {e}")
    fit_feats = np.asarray(fit_feats, dtype=np.float64)
    print(f"[fit] 특징 {fit_feats.shape} / 클래스 {sorted(set(fit_labels))}")

    from sklearn.decomposition import PCA
    from sklearn.covariance import LedoitWolf

    pca = PCA(n_components=min(args.pca_dim, fit_feats.shape[0] - 1)).fit(fit_feats)
    Z = pca.transform(fit_feats)
    class_means = {c: Z[[i for i, l in enumerate(fit_labels) if l == c]].mean(0)
                   for c in set(fit_labels)}
    # 클래스내 중심화 후 공유 공분산(수축) → 안정적 역행렬
    centered = np.vstack([Z[i] - class_means[fit_labels[i]] for i in range(len(Z))])
    prec = LedoitWolf().fit(centered).precision_  # 역공분산
    means_mat = np.vstack(list(class_means.values()))

    def mahalanobis(feat: np.ndarray) -> float:
        z = pca.transform(feat[None, :].astype(np.float64))[0]
        d = means_mat - z  # (C, k)
        m = np.einsum("ck,kj,cj->c", d, prec, d)  # 각 클래스 거리^2
        return float(m.min())

    # ── test 채점 ───────────────────────────────────────
    data = json.loads(args.testset.read_text(encoding="utf-8"))
    if args.limit_test:
        data = data[: args.limit_test]
    rows = []
    print(f"[test] {len(data)}건 채점...")
    for i, ex in enumerate(data):
        img = _load_img(ex)
        conf, energy, entropy, pred = gen_signals(img)
        maha = mahalanobis(feature(img))
        gt = _gt_type(ex)
        rows.append({
            "id": ex["id"], "gt_type": gt, "pred_type": pred,
            "is_novel": int(gt == novel),
            "confidence": round(conf, 5), "energy": round(energy, 5),
            "entropy": round(entropy, 5), "mahalanobis": round(maha, 5),
        })
        if (i + 1) % 30 == 0:
            print(f"  {i+1}/{len(data)}")

    # ── OOD 점수(높을수록 OOD) ──────────────────────────
    y = np.array([r["is_novel"] for r in rows])
    raw_scores = {
        "confidence": 1.0 - np.array([r["confidence"] for r in rows]),
        "energy": np.array([r["energy"] for r in rows]),
        "entropy": np.array([r["entropy"] for r in rows]),
        "mahalanobis": np.array([r["mahalanobis"] for r in rows]),
    }

    def _z(a):
        sd = a.std()
        return (a - a.mean()) / sd if sd > 1e-9 else a * 0.0

    raw_scores["ensemble"] = sum(_z(v) for v in [
        raw_scores["confidence"], raw_scores["energy"],
        raw_scores["entropy"], raw_scores["mahalanobis"]]) / 4.0

    from sklearn.metrics import roc_auc_score, roc_curve

    def _tpr_at_fpr(yt, sc, target=0.05):
        fpr, tpr, _ = roc_curve(yt, sc)
        ok = fpr <= target
        return float(tpr[ok].max()) if ok.any() else 0.0

    def _fpr_at_tpr(yt, sc, target=0.95):
        fpr, tpr, _ = roc_curve(yt, sc)
        ok = tpr >= target
        return float(fpr[ok].min()) if ok.any() else 1.0

    detectors = {}
    for name, sc in raw_scores.items():
        detectors[name] = {
            "auroc": round(float(roc_auc_score(y, sc)), 4),
            "tpr_at_fpr5": round(_tpr_at_fpr(y, sc, 0.05), 4),
            "fpr95": round(_fpr_at_tpr(y, sc, 0.95), 4),
        }

    # 기준선 confidence 임계값(0.80) 적발률 — eval_ood.py 재현 대조
    thr = config.CONFIDENCE_THRESHOLD
    nov = [r for r in rows if r["is_novel"]]
    kno = [r for r in rows if not r["is_novel"]]
    base_detect = sum(1 for r in nov if r["confidence"] < thr) / max(len(nov), 1)
    base_falsealarm = sum(1 for r in kno if r["confidence"] < thr) / max(len(kno), 1)

    report = {
        "adapter": args.adapter, "novel_class": novel,
        "n_test": len(rows), "n_novel": len(nov), "n_known": len(kno),
        "fit_per_class": args.fit_per_class, "pca_dim": int(pca.n_components_),
        "detectors": detectors,
        "confidence_threshold_baseline": {
            "threshold": thr,
            "novel_detection_rate": round(base_detect, 4),
            "known_false_alarm_rate": round(base_falsealarm, 4),
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    args.json_out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== 전용 OOD 점수 비교 (신규 = %s, %d/%d) ===" % (novel, len(nov), len(rows)))
    print(f"{'detector':<14}{'AUROC':>8}{'TPR@FPR5%':>12}{'FPR95':>9}")
    for name, m in detectors.items():
        print(f"{name:<14}{m['auroc']:>8.4f}{m['tpr_at_fpr5']:>12.4f}{m['fpr95']:>9.4f}")
    print(f"\n[기준선 재현] confidence<{thr}: 신규적발 {base_detect:.1%}, 기존오경보 {base_falsealarm:.1%}")
    print(f"저장: {args.out} / {args.json_out}")


if __name__ == "__main__":
    main()
