"""추론 지연/처리량 벤치마크 — 양산 인라인 검사 현실성 지표 (GPU 필요).

운영 모델(레지스트리 active 어댑터)을 고정 평가셋 이미지로 반복 추론해
단건 latency 분포(p50/p90/p95/p99)·처리량(img/s)·토큰 생성 속도·GPU 메모리
풋프린트를 측정한다. app.main의 실제 추론 경로(run_inference)를 그대로 쓰므로
운영에서 보게 될 수치와 동일하다.

사용 (GPU):
    python scripts/benchmark_latency.py --n 60
    python scripts/benchmark_latency.py --adapter models/checkpoints/cand_v4 --n 60
"""
import argparse
import json
import statistics as stats
import sys
import time
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
DATA = ROOT / "data" / "processed"


def _active_adapter() -> str:
    reg = json.loads((ROOT / "models" / "registry.json").read_text(encoding="utf-8"))
    cur = reg.get("current")
    for m in reg.get("models", []):
        if m.get("version") == cur:
            return m["adapter_path"]
    raise SystemExit("registry에서 active 어댑터를 못 찾음")


def main():
    ap = argparse.ArgumentParser(description="추론 지연/처리량 벤치마크")
    ap.add_argument("--adapter", default=None, help="기본: registry의 active 어댑터")
    ap.add_argument("--testset", type=Path, default=DATA / "test.json")
    ap.add_argument("--n", type=int, default=60, help="측정 표본 수(워밍업 별도)")
    ap.add_argument("--warmup", type=int, default=3, help="측정 제외 워밍업 횟수")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "results" / "latency_benchmark.json")
    args = ap.parse_args()

    adapter = args.adapter or _active_adapter()
    from app import main as appmain
    appmain.LORA_PATH = Path(adapter)
    appmain.USE_LORA = appmain.LORA_PATH.exists()
    if not appmain.USE_LORA:
        raise SystemExit(f"어댑터 경로가 없습니다: {adapter}")

    import torch
    print(f"어댑터: {adapter}")
    load_t0 = time.time()
    appmain.load_model()
    load_sec = time.time() - load_t0
    print(f"모델 로드: {load_sec:.1f}s")

    data = json.loads(args.testset.read_text(encoding="utf-8"))
    imgs = []
    for ex in data[: args.warmup + args.n]:
        p = Path(ex["image"])
        ap_ = (ROOT / p) if not p.is_absolute() else p
        imgs.append(Image.open(ap_).convert("RGB"))

    dev = appmain.model.device
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # 워밍업 (CUDA 그래프/캐시 워밍업 — 측정 제외)
    for img in imgs[: args.warmup]:
        appmain.run_inference(img)

    lat, ntok = [], []
    for img in imgs[args.warmup : args.warmup + args.n]:
        parsed, raw, elapsed, conf = appmain.run_inference(img)
        lat.append(elapsed)
        ntok.append(len(appmain.processor.tokenizer(raw)["input_ids"]))

    lat_sorted = sorted(lat)

    def pct(p):
        return lat_sorted[min(len(lat_sorted) - 1, int(round(p / 100 * len(lat_sorted))) - 1)]

    total_tok = sum(ntok)
    total_time = sum(lat)
    peak_mem_gb = (
        torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else None
    )
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

    report = {
        "adapter": adapter,
        "gpu": gpu_name,
        "quantization": "nf4 4bit (double quant)",
        "n_samples": args.n,
        "warmup": args.warmup,
        "model_load_sec": round(load_sec, 2),
        "latency_sec": {
            "mean": round(stats.mean(lat), 3),
            "p50": round(pct(50), 3),
            "p90": round(pct(90), 3),
            "p95": round(pct(95), 3),
            "p99": round(pct(99), 3),
            "min": round(min(lat), 3),
            "max": round(max(lat), 3),
        },
        "throughput_img_per_sec": round(args.n / total_time, 3),
        "tokens_per_sec": round(total_tok / total_time, 1),
        "avg_tokens_per_response": round(total_tok / args.n, 1),
        "gpu_peak_mem_gb": round(peak_mem_gb, 2) if peak_mem_gb else None,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== 벤치마크 결과 ===")
    print(f"GPU            : {gpu_name}")
    print(f"단건 latency   : 평균 {report['latency_sec']['mean']}s "
          f"(p50 {report['latency_sec']['p50']} / p90 {report['latency_sec']['p90']} "
          f"/ p99 {report['latency_sec']['p99']})")
    print(f"처리량         : {report['throughput_img_per_sec']} img/s "
          f"({report['tokens_per_sec']} tok/s, 평균 {report['avg_tokens_per_response']} tok/응답)")
    print(f"GPU peak mem   : {report['gpu_peak_mem_gb']} GB")
    print(f"\n저장: {args.out}")


if __name__ == "__main__":
    main()
