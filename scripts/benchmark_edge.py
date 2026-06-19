"""엣지 CNN 압축·배포 벤치마크 — 엣지 배포·경량화 트랙.

train_edge_cnn.py로 학습한 경량 CNN을 (1) ONNX로 익스포트하고 (2) INT8 정적
양자화(PTQ)한 뒤, 같은 test 270건으로 **정확도·지연·모델크기**를 조합별로 실측한다:

    torch fp32 (GPU)   — 학습 그대로
    torch fp32 (CPU)   — GPU 없는 라인 PC 가정
    ONNX  fp32 (CPU)   — 런타임 교체만
    ONNX  INT8 (CPU)   — 가중치/활성 8bit 양자화 (엣지 핵심)

엣지=대개 CPU/ARM이므로 INT8 이득(속도·4× 작은 크기)은 CPU에서 가장 의미 있다.
양산 타깃은 TensorRT(GPU INT8)지만 Windows 재현성을 위해 ONNX Runtime로 측정한다.
VLM(14.7s/img·6GB, scripts/benchmark_latency.py)과 한 표에서 대비한다.

사용:
    python scripts/benchmark_edge.py --arch resnet18
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app import config  # noqa: E402

SPLIT_DIR = ROOT / "data" / "processed"
CKPT_DIR = ROOT / "models" / "checkpoints" / "edge_cnn"
CLASSES = config.DEFECT_CLASSES
CLS2IDX = {c: i for i, c in enumerate(CLASSES)}
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _load_split(name):
    recs = json.loads((SPLIT_DIR / f"{name}.json").read_text(encoding="utf-8"))
    out = []
    for r in recs:
        label = json.loads(r["conversations"][1]["content"])["type"]
        out.append((str(ROOT / r["image"]), CLS2IDX[label]))
    return out


def _preprocess(path, img_size=224):
    """val 변환과 동일: grayscale→3ch, resize, [0,1], ImageNet 정규화 → (3,H,W) f32."""
    from PIL import Image

    img = Image.open(path).convert("L").resize((img_size, img_size), Image.BILINEAR)
    x = np.asarray(img, dtype=np.float32) / 255.0          # (H,W) [0,1]
    x = np.stack([x, x, x], axis=0)                        # (3,H,W)
    x = (x - IMAGENET_MEAN[:, None, None]) / IMAGENET_STD[:, None, None]
    return x.astype(np.float32)


def _pct(times_ms):
    a = np.array(times_ms)
    return {"mean": round(float(a.mean()), 3), "p50": round(float(np.percentile(a, 50)), 3),
            "p99": round(float(np.percentile(a, 99)), 3)}


# ── torch 경로 ────────────────────────────────────────────
def _build_model(arch, num_classes):
    import torch.nn as nn
    import torchvision.models as M

    if arch == "resnet18":
        m = M.resnet18(weights=None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    else:
        m = M.mobilenet_v3_small(weights=None)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
    return m


def _torch_bench(model, X, y, device, runs=200):
    import torch

    model.eval().to(device)
    xt = torch.from_numpy(X).to(device)
    # accuracy
    with torch.no_grad():
        preds = []
        for i in range(0, len(xt), 64):
            preds.append(model(xt[i:i + 64]).argmax(1).cpu().numpy())
    acc = float((np.concatenate(preds) == y).mean())
    # latency (batch=1)
    one = xt[:1]
    with torch.no_grad():
        for _ in range(5):  # warmup
            model(one)
            if device == "cuda":
                torch.cuda.synchronize()
        times = []
        for i in range(runs):
            t = time.perf_counter()
            model(xt[i % len(xt):i % len(xt) + 1])
            if device == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t) * 1000)
    return acc, _pct(times)


# ── ONNX 경로 ─────────────────────────────────────────────
def _export_onnx(model, arch, img_size=224):
    import torch

    model.eval().cpu()
    path = CKPT_DIR / f"{arch}_fp32.onnx"
    dummy = torch.randn(1, 3, img_size, img_size)
    torch.onnx.export(
        model, dummy, str(path), input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}}, opset_version=17,
        dynamo=False,
    )
    return path


class _CalibReader:
    """INT8 정적 양자화용 캘리브레이션: train 일부를 input_name으로 흘려준다."""

    def __init__(self, paths, input_name, img_size=224):
        self.data = [{input_name: _preprocess(p, img_size)[None]} for p in paths]
        self.it = iter(self.data)

    def get_next(self):
        return next(self.it, None)


def _quantize_int8(fp32_path, arch, calib_paths):
    from onnxruntime.quantization import (CalibrationMethod, QuantFormat,
                                          QuantType, quantize_static)
    from onnxruntime.quantization.preprocess import quant_pre_process

    prep = CKPT_DIR / f"{arch}_prep.onnx"
    quant_pre_process(str(fp32_path), str(prep))
    int8_path = CKPT_DIR / f"{arch}_int8.onnx"
    reader = _CalibReader(calib_paths, "input")
    quantize_static(
        str(prep), str(int8_path), reader,
        quant_format=QuantFormat.QDQ, per_channel=True,
        activation_type=QuantType.QInt8, weight_type=QuantType.QInt8,
        calibrate_method=CalibrationMethod.MinMax,
    )
    prep.unlink(missing_ok=True)
    return int8_path


def _ort_bench(onnx_path, X, y, runs=200):
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.intra_op_num_threads = 1  # 엣지 단일코어 가정(보수적)
    sess = ort.InferenceSession(str(onnx_path), so, providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    # accuracy
    preds = []
    for i in range(0, len(X), 64):
        out = sess.run(None, {name: X[i:i + 64]})[0]
        preds.append(out.argmax(1))
    acc = float((np.concatenate(preds) == y).mean())
    # latency (batch=1)
    one = X[:1]
    for _ in range(5):
        sess.run(None, {name: one})
    times = []
    for i in range(runs):
        x1 = X[i % len(X):i % len(X) + 1]
        t = time.perf_counter()
        sess.run(None, {name: x1})
        times.append((time.perf_counter() - t) * 1000)
    return acc, _pct(times)


def main():
    import torch

    ap = argparse.ArgumentParser(description="엣지 CNN INT8 압축·배포 벤치마크")
    ap.add_argument("--arch", default="resnet18",
                    choices=["resnet18", "mobilenet_v3_small"])
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--runs", type=int, default=200)
    ap.add_argument("--calib", type=int, default=128, help="INT8 캘리브 샘플 수")
    args = ap.parse_args()

    ckpt = CKPT_DIR / f"{args.arch}.pt"
    if not ckpt.exists():
        raise SystemExit(f"체크포인트 없음: {ckpt} (먼저 train_edge_cnn.py 실행)")

    te = _load_split("test")
    X = np.stack([_preprocess(p, args.img_size) for p, _ in te])
    y = np.array([lab for _, lab in te])
    print(f"[data] test {len(te)}건 전처리 완료  X{X.shape}")

    model = _build_model(args.arch, len(CLASSES))
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    fp32_pt_mb = ckpt.stat().st_size / 1e6

    results = {}

    # torch GPU
    if torch.cuda.is_available():
        acc, lat = _torch_bench(model, X, y, "cuda", args.runs)
        results["torch_fp32_gpu"] = {"accuracy": round(acc, 4), "latency_ms": lat,
                                     "throughput_img_per_sec": round(1000 / lat["p50"], 1),
                                     "size_mb": round(fp32_pt_mb, 2)}
        print("torch fp32 GPU:", results["torch_fp32_gpu"])

    # torch CPU
    acc, lat = _torch_bench(model, X, y, "cpu", args.runs)
    results["torch_fp32_cpu"] = {"accuracy": round(acc, 4), "latency_ms": lat,
                                 "throughput_img_per_sec": round(1000 / lat["p50"], 1),
                                 "size_mb": round(fp32_pt_mb, 2)}
    print("torch fp32 CPU:", results["torch_fp32_cpu"])

    # ONNX export + fp32 CPU
    fp32_onnx = _export_onnx(model, args.arch, args.img_size)
    acc, lat = _ort_bench(fp32_onnx, X, y, args.runs)
    results["onnx_fp32_cpu"] = {"accuracy": round(acc, 4), "latency_ms": lat,
                                "throughput_img_per_sec": round(1000 / lat["p50"], 1),
                                "size_mb": round(fp32_onnx.stat().st_size / 1e6, 2)}
    print("onnx  fp32 CPU:", results["onnx_fp32_cpu"])

    # INT8 quantize + CPU
    calib_paths = [p for p, _ in _load_split("train")[:args.calib]]
    int8_onnx = _quantize_int8(fp32_onnx, args.arch, calib_paths)
    acc, lat = _ort_bench(int8_onnx, X, y, args.runs)
    results["onnx_int8_cpu"] = {"accuracy": round(acc, 4), "latency_ms": lat,
                                "throughput_img_per_sec": round(1000 / lat["p50"], 1),
                                "size_mb": round(int8_onnx.stat().st_size / 1e6, 2)}
    print("onnx  INT8 CPU:", results["onnx_int8_cpu"])

    # 요약: INT8 vs fp32 CPU 압축효과
    f = results["onnx_fp32_cpu"]
    q = results["onnx_int8_cpu"]
    summary = {
        "cpu_int8_speedup_vs_fp32": round(f["latency_ms"]["p50"] / q["latency_ms"]["p50"], 2),
        "size_reduction_x": round(f["size_mb"] / q["size_mb"], 2),
        "accuracy_drop": round(f["accuracy"] - q["accuracy"], 4),
    }

    out = {
        "arch": args.arch,
        "img_size": args.img_size,
        "runs": args.runs,
        "device_gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "variants": results,
        "compression_summary": summary,
        "vlm_reference": {"latency_sec_mean": 14.681, "peak_mem_gb": 6.03,
                          "throughput_img_per_sec": 0.068, "test_accuracy": 0.9593},
    }
    out_path = ROOT / "data" / "results" / f"edge_deploy_{args.arch}.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n=== 압축 요약 ===", summary)
    print(f"결과 저장: {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
