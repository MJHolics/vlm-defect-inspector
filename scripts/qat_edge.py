"""QAT(Quantization-Aware Training)로 INT8 PTQ 붕괴를 회복 — 저정밀 학습 트랙.

benchmark_edge.py의 ONNX INT8 PTQ에서 **MobileNetV3-Small은 99.6%→붕괴**한다
(depthwise/SE·hardswish의 넓은 활성 분포가 per-tensor 정적 양자화에서 깨짐).
포트폴리오는 그간 "PTQ 붕괴 → QAT 필요"까지만 *진단*했다. 이 스크립트는 그
진단을 **실측으로 닫는다** — QAT로 재학습하면 정확도가 회복되는가?

정직한 대조를 위해 **같은 torch FX 백엔드 안에서 PTQ와 QAT를 나란히** 측정한다
(ONNX-PTQ vs torch-QAT 같은 프레임워크 혼동 배제):

    fp32            — 학습 그대로(기준)
    PTQ (torch FX)  — 보정만(fake-quant 학습 없음) → 아키텍처 취약점 노출
    QAT (torch FX)  — fake-quant를 켠 채 소수 epoch 재학습 → 회복 측정

test는 VLM·PTQ와 동일한 270건(누수 0). 양자화 엔진은 onednn(x86 CPU).
fake-quant 학습은 GPU에서(부동소수 연산), 변환된 INT8 추론은 CPU에서 돈다.

사용:
    python scripts/qat_edge.py --arch mobilenet_v3_small --epochs 4
    python scripts/qat_edge.py --arch resnet18 --epochs 3
"""
import argparse
import copy
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app import config  # noqa: E402
from scripts.train_edge_cnn import _DS, _build_model, _load_split  # noqa: E402

CKPT_DIR = ROOT / "models" / "checkpoints" / "edge_cnn"
CLASSES = config.DEFECT_CLASSES


def _qconfig_mappings():
    """onednn(우선)·x86 순으로 사용 가능한 qconfig 매핑을 고른다."""
    from torch.ao.quantization import (get_default_qat_qconfig_mapping,
                                       get_default_qconfig_mapping)

    for backend in ("onednn", "x86", "fbgemm"):
        try:
            ptq = get_default_qconfig_mapping(backend)
            qat = get_default_qat_qconfig_mapping(backend)
            return backend, ptq, qat
        except Exception:
            continue
    raise SystemExit("사용 가능한 양자화 백엔드 없음")


def _eval_int8(model, X, y):
    """CPU INT8 모델의 test 정확도·클래스별 정확도(고정 test 270)."""
    import torch

    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), 64):
            xb = torch.from_numpy(X[i:i + 64])
            preds.append(model(xb).argmax(1).cpu().numpy())
    pred = np.concatenate(preds)
    acc = float((pred == y).mean())
    per_class = {}
    for ci, cname in enumerate(CLASSES):
        m = y == ci
        if m.any():
            per_class[cname] = round(float((pred[m] == y[m]).mean()), 4)
    return acc, per_class


def _int8_size_mb(model):
    """변환된 INT8 모델을 직렬화해 실제 디스크 크기(MB)를 잰다."""
    import torch

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        tmp = Path(f.name)
    try:
        torch.jit.save(torch.jit.script(model), str(tmp))
        return round(tmp.stat().st_size / 1e6, 2)
    except Exception:
        torch.save(model.state_dict(), str(tmp))
        return round(tmp.stat().st_size / 1e6, 2)
    finally:
        tmp.unlink(missing_ok=True)


def _cpu_latency_ms(model, X, runs=100):
    """batch=1 CPU 단건 지연 p50(ms). 엣지 단일건 처리 가정."""
    import torch

    model.eval()
    one = torch.from_numpy(X[:1])
    with torch.no_grad():
        for _ in range(5):
            model(one)
        times = []
        for i in range(runs):
            xb = torch.from_numpy(X[i % len(X):i % len(X) + 1])
            t = time.perf_counter()
            model(xb)
            times.append((time.perf_counter() - t) * 1000)
    return round(float(np.percentile(times, 50)), 3)


def main():
    import torch
    import torch.nn as nn
    from torch.ao.quantization.quantize_fx import (convert_fx, prepare_fx,
                                                   prepare_qat_fx)
    from torch.utils.data import DataLoader

    ap = argparse.ArgumentParser(description="QAT로 INT8 PTQ 붕괴 회복")
    ap.add_argument("--arch", default="mobilenet_v3_small",
                    choices=["resnet18", "mobilenet_v3_small"])
    ap.add_argument("--epochs", type=int, default=12, help="QAT fine-tune epoch")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--calib", type=int, default=256, help="PTQ 보정 샘플 수")
    ap.add_argument("--img-size", type=int, default=224)
    args = ap.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)

    ckpt = CKPT_DIR / f"{args.arch}.pt"
    if not ckpt.exists():
        raise SystemExit(f"체크포인트 없음: {ckpt} (먼저 train_edge_cnn.py)")

    backend, ptq_map, qat_map = _qconfig_mappings()
    torch.backends.quantized.engine = backend
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[env] backend={backend} | qat device={device} | int8 추론=CPU")

    # 고정 test 270 텐서 (benchmark_edge와 동일 전처리)
    from scripts.benchmark_edge import _preprocess
    te = _load_split("test")
    X = np.stack([_preprocess(p, args.img_size) for p, _ in te]).astype(np.float32)
    y = np.array([lab for _, lab in te])
    print(f"[data] test {len(te)}건 X{X.shape}")

    # fp32 로드
    fp32 = _build_model(args.arch, len(CLASSES), pretrained=False)
    fp32.load_state_dict(torch.load(ckpt, map_location="cpu"))
    fp32.eval()
    example = (torch.randn(1, 3, args.img_size, args.img_size),)

    results = {}

    # fp32 기준 (CPU)
    acc, per = _eval_int8(fp32, X, y)
    results["fp32"] = {"accuracy": round(acc, 4), "per_class": per,
                       "size_mb": round(ckpt.stat().st_size / 1e6, 2),
                       "latency_ms_p50": _cpu_latency_ms(fp32, X)}
    print(f"fp32           acc {acc:.4f}")

    # ── PTQ (torch FX, 보정만) ──────────────────────────────
    ptq_prep = prepare_fx(copy.deepcopy(fp32).eval(), ptq_map, example)
    calib = [p for p, _ in _load_split("train")[:args.calib]]
    with torch.no_grad():
        for p in calib:
            ptq_prep(torch.from_numpy(_preprocess(p, args.img_size)[None]))
    ptq_int8 = convert_fx(ptq_prep)
    acc, per = _eval_int8(ptq_int8, X, y)
    results["ptq_int8"] = {"accuracy": round(acc, 4), "per_class": per,
                           "size_mb": _int8_size_mb(ptq_int8),
                           "latency_ms_p50": _cpu_latency_ms(ptq_int8, X)}
    print(f"PTQ  INT8(FX)  acc {acc:.4f}   (보정만·재학습 없음)")

    # ── QAT (torch FX, fake-quant 재학습) ───────────────────
    qat = prepare_qat_fx(copy.deepcopy(fp32).train(), qat_map, example)
    qat.to(device)
    tr = _load_split("train")
    tr_loader = DataLoader(_DS(tr, True, args.img_size), batch_size=args.batch_size,
                           shuffle=True, num_workers=0)
    opt = torch.optim.AdamW(qat.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    # QAT 표준 레시피: 후반부에 관측자(양자화 범위)·BN 통계를 동결해 안정 수렴
    freeze_obs_at = max(1, int(args.epochs * 0.6))
    freeze_bn_at = max(1, int(args.epochs * 0.75))
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        qat.train()
        if ep == freeze_obs_at:
            from torch.ao.quantization import disable_observer
            qat.apply(disable_observer)
        if ep == freeze_bn_at:
            try:
                from torch.ao.nn.intrinsic.qat import freeze_bn_stats
                qat.apply(freeze_bn_stats)
            except Exception:
                pass
        tot = 0.0
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(qat(xb), yb)
            loss.backward()
            opt.step()
            tot += loss.item() * len(yb)
        sched.step()
        tag = ""
        if ep >= freeze_bn_at:
            tag = "  [obs+bn frozen]"
        elif ep >= freeze_obs_at:
            tag = "  [obs frozen]"
        print(f"  QAT epoch {ep:2d} | train_loss {tot/len(tr):.4f}{tag}")
    qat_sec = round(time.time() - t0, 1)

    qat.to("cpu").eval()
    qat_int8 = convert_fx(qat)
    acc, per = _eval_int8(qat_int8, X, y)
    results["qat_int8"] = {"accuracy": round(acc, 4), "per_class": per,
                           "size_mb": _int8_size_mb(qat_int8),
                           "latency_ms_p50": _cpu_latency_ms(qat_int8, X),
                           "qat_epochs": args.epochs, "qat_seconds": qat_sec}
    print(f"QAT  INT8(FX)  acc {acc:.4f}   ({args.epochs} epoch fake-quant 재학습)")

    # 요약
    f32, ptq, qat_r = results["fp32"], results["ptq_int8"], results["qat_int8"]
    summary = {
        "ptq_drop_from_fp32": round(f32["accuracy"] - ptq["accuracy"], 4),
        "qat_drop_from_fp32": round(f32["accuracy"] - qat_r["accuracy"], 4),
        "qat_recovery_over_ptq": round(qat_r["accuracy"] - ptq["accuracy"], 4),
        "size_reduction_x": round(f32["size_mb"] / max(qat_r["size_mb"], 1e-6), 2),
    }
    out = {
        "arch": args.arch, "backend": backend, "img_size": args.img_size,
        "test_n": len(te), "variants": results, "summary": summary,
    }
    (ROOT / "data" / "results").mkdir(parents=True, exist_ok=True)
    out_path = ROOT / "data" / "results" / f"qat_edge_{args.arch}.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 60)
    print(f" {args.arch}  (backend={backend}, test {len(te)})")
    print(f"   fp32      {f32['accuracy']*100:5.1f}%")
    print(f"   PTQ INT8  {ptq['accuracy']*100:5.1f}%   (fp32 대비 {summary['ptq_drop_from_fp32']*100:+.1f}%p)")
    print(f"   QAT INT8  {qat_r['accuracy']*100:5.1f}%   (fp32 대비 {summary['qat_drop_from_fp32']*100:+.1f}%p)")
    print(f"   → QAT가 PTQ 대비 {summary['qat_recovery_over_ptq']*100:+.1f}%p 회복")
    print("=" * 60)
    print(f"결과 저장: {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
