"""몇 장 가르치기용 백본을 ONNX로 낸다(브라우저 onnxruntime-web용).

입력 (1,3,256,256) 0~1 RGB → 출력 (1,32,32,D) 패치 특징. 정규화·층 결합·평활은 그래프 안에 넣어
브라우저 쪽 코드는 "사진 → 특징"만 부르면 되게 한다. 대조용 고정 입력과 그 출력도 같이 저장한다.

    python scripts/export_teach_backbone.py --out tools/teach_probe --backbone mobilenet_v3_small
    python scripts/export_teach_backbone.py --out tools/teach_probe --backbone resnet18
    cd tools/teach_probe && python -m http.server 8791   # 콘솔에서 await run('mobilenet_v3_small','wasm')
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fewshot_teach_bench import SIZE, Backbone  # noqa: E402


class NearestDistance(torch.nn.Module):
    """(N,D) 패치와 (M,D) 뱅크 → 패치별 최근접 제곱거리. 순수 JS 루프 대신 이 그래프를 부른다."""

    def forward(self, q: torch.Tensor, bank: torch.Tensor) -> torch.Tensor:
        d = (q * q).sum(1, keepdim=True) + (bank * bank).sum(1)[None] - 2 * q @ bank.T
        return d.min(1).values


def export_knn(out: Path) -> None:
    torch.onnx.export(NearestDistance(), (torch.rand(8, 4), torch.rand(16, 4)), str(out / "knn.onnx"),
                      input_names=["q", "bank"], output_names=["d"], opset_version=17, dynamo=False,
                      dynamic_axes={"q": {0: "n", 1: "dim"}, "bank": {0: "m", 1: "dim"}})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--backbone", default="mobilenet_v3_small")
    args = ap.parse_args()
    out = Path(args.out) / args.backbone
    out.mkdir(parents=True, exist_ok=True)
    export_knn(Path(args.out))
    net = Backbone(args.backbone).eval()
    rng = np.random.default_rng(0)
    x = torch.from_numpy(rng.random((1, 3, SIZE, SIZE), dtype=np.float32))
    path = out / f"teach_{args.backbone}.onnx"
    torch.onnx.export(net, x, str(path), input_names=["image"], output_names=["patches"],
                      opset_version=17, dynamo=False)
    with torch.no_grad():
        y = net(x).numpy()
    import onnxruntime as ort
    y2 = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"]).run(None, {"image": x.numpy()})[0]
    ref = {"shape": list(y.shape), "sum": float(y.sum()), "first": y.reshape(-1)[:16].tolist(),
           "torch_vs_ort_maxabs": float(np.abs(y - y2).max())}
    (out / "ref.json").write_text(json.dumps(ref), encoding="utf-8")
    x.numpy().tofile(out / "input.f32")
    y.tofile(out / "output.f32")
    print(path, f"{path.stat().st_size / 1e6:.2f}MB", ref["shape"], "torch-ort", ref["torch_vs_ort_maxabs"])


if __name__ == "__main__":
    main()
