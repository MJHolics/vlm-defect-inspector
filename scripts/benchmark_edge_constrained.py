"""엣지 제약 환경 벤치 — ONNX 4종을 CPU·메모리를 제한한 컨테이너에서 재고, arm64 이식성을 확인한다.

## 왜 하는가
기존 `benchmark_edge.py`는 x86 데스크톱 CPU(제한 없음)에서 쟀다. 엣지 보드는 코어 수·메모리가 훨씬 작다.
실물 Jetson·라즈베리파이가 없어서, Docker 로 **코어 수(--cpus)와 메모리(--memory)를 제한**해 그 영향을 재고,
`--platform linux/arm64`(QEMU 에뮬레이션)로 같은 모델이 aarch64 런타임에서도 같은 출력을 내는지 확인한다.

## 이 측정이 말하지 않는 것 (반드시 구분)
- 실 보드의 지연이 아니다. --cpus 는 CFS 쿼터로 x86 코어 시간을 깎는 것이라 ARM 코어의 IPC·캐시·메모리 대역폭·NEON 을 모른다.
- arm64 에뮬레이션(QEMU)의 지연은 의미가 없다(수 배 느림). **정확성(출력 일치)만** 본다.
- GPU/NPU(TensorRT, Jetson)는 쓰지 않았다. ONNX Runtime CPU 전용이다.

## 실행 (호스트에서)
    python scripts/benchmark_edge_constrained.py            # docs 결과: data/results/edge_constrained.json
컨테이너 안에서는 `--inside` 모드로 이 파일이 다시 실행된다.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CKPT = ROOT / "models" / "checkpoints" / "edge_cnn"
OUT = ROOT / "data" / "results" / "edge_constrained.json"
MODELS = ["mobilenet_v3_small_fp32", "mobilenet_v3_small_int8", "resnet18_fp32", "resnet18_int8"]
IMAGE = "edge-bench:py312"
IMAGE_ARM = "edge-bench:py312-arm64"


def inside(args):
    import resource  # 리눅스(컨테이너) 전용
    import numpy as np
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.intra_op_num_threads = args.threads
    so.inter_op_num_threads = 1
    sess = ort.InferenceSession(f"/models/{args.model}.onnx", so, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    shape = [d if isinstance(d, int) else 1 for d in inp.shape]
    rng = np.random.default_rng(0)
    x = rng.random(shape, dtype=np.float32)
    ref = sess.run(None, {inp.name: x})[0]
    for _ in range(args.warmup):
        sess.run(None, {inp.name: x})
    ts = []
    for _ in range(args.runs):
        t0 = time.perf_counter()
        sess.run(None, {inp.name: x})
        ts.append((time.perf_counter() - t0) * 1000)
    ts.sort()
    res = {
        "model": args.model, "threads": args.threads, "shape": shape,
        "p50_ms": round(ts[len(ts) // 2], 3), "p95_ms": round(ts[int(len(ts) * 0.95)], 3),
        "max_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
        "ref_logits_head": [round(float(v), 5) for v in ref.reshape(-1)[:6]],
        "arch": __import__("platform").machine(),
    }
    print("RESULT " + json.dumps(res))


def docker_run(model, cpus, mem, platform=None, runs=200, warmup=30, timeout=900):
    cmd = ["docker", "run", "--rm", f"--cpus={cpus}", f"--memory={mem}", "--memory-swap", mem,
           "-v", f"{CKPT}:/models:ro", "-v", f"{Path(__file__).resolve()}:/bench.py:ro"]
    if platform:
        cmd += ["--platform", platform]
    cmd += [IMAGE_ARM if platform else IMAGE, "python", "/bench.py", "--inside", "--model", model, "--threads", str(int(max(1, cpus))),
            "--runs", str(runs), "--warmup", str(warmup)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    for line in r.stdout.splitlines():
        if line.startswith("RESULT "):
            return json.loads(line[7:])
    return {"model": model, "error": f"exit={r.returncode}", "stderr": r.stderr.strip()[-200:]}


def host(args):
    subprocess.run(["docker", "build", "-t", IMAGE, "-"], input=(
        "FROM python:3.12-slim\nRUN pip install --no-cache-dir onnxruntime numpy\n"), text=True, check=True)
    subprocess.run(["docker", "build", "--platform", "linux/arm64", "-t", IMAGE_ARM, "-"], input=(
        "FROM python:3.12-slim\nRUN pip install --no-cache-dir onnxruntime numpy\n"), text=True, check=True)
    report = {"note": "CPU 쿼터·메모리 제한 컨테이너(x86) 측정. 실 보드 아님. arm64는 정확성만.", "cpu_sweep": [],
              "min_memory": [], "arm64_emulated": []}
    for m in MODELS:
        for cpus in (1, 2, 4):
            r = docker_run(m, cpus, "1g")
            r["cpus"] = cpus
            report["cpu_sweep"].append(r)
            print("cpu", m, cpus, r.get("p50_ms"), r.get("p95_ms"), r.get("max_rss_mb"), r.get("error", ""), flush=True)
    for m in MODELS:  # 메모리를 줄여 가며 OOM 경계 찾기(cpus=1)
        row = {"model": m, "limits": {}}
        for mem in ("512m", "256m", "160m", "128m", "96m", "64m"):
            r = docker_run(m, 1, mem, runs=60, warmup=10)
            row["limits"][mem] = "ok" if "p50_ms" in r else "fail(OOM/오류)"
            if "p50_ms" not in r:
                break
        report["min_memory"].append(row)
        print("mem", row, flush=True)
    x86_ref = {r["model"]: r["ref_logits_head"] for r in report["cpu_sweep"] if r.get("cpus") == 1 and "p50_ms" in r}
    for m in MODELS:
        r = docker_run(m, 2, "1g", platform="linux/arm64", runs=5, warmup=1, timeout=1500)
        if "ref_logits_head" in r and m in x86_ref:
            r["max_abs_diff_vs_x86"] = round(max(abs(a - b) for a, b in zip(r["ref_logits_head"], x86_ref[m])), 6)
        report["arm64_emulated"].append(r)
        print("arm64", m, r.get("arch"), r.get("max_abs_diff_vs_x86"), r.get("error", ""), flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("저장:", OUT)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--inside", action="store_true")
    ap.add_argument("--model")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--runs", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=30)
    a = ap.parse_args()
    inside(a) if a.inside else host(a)
