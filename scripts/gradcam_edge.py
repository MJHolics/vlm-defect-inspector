"""Grad-CAM 설명가능성 — 엣지 CNN이 '어디를 보고' 결함을 판정했는지 시각화.

플래그십 서사의 마지막 조각: 정확도(98.9%/99.6%)·불확실성(OOD·confidence 게이트)에 더해
**판단 근거(saliency)**를 제시한다. Grad-CAM은 예측 클래스 로짓을 마지막 conv 특성맵으로
역전파해 기여도 가중 활성맵을 얻고, 입력 위에 히트맵으로 겹친다 — 모델이 실제 결함 영역(스크래치
선, 개재물 점 등)에 주목하는지 눈으로 검증할 수 있다.

같은 학습 코드(scripts/train_edge_cnn.py)의 모델 빌더·평가 전처리를 그대로 재사용해, 시각화가
배포 모델과 동일 입력 파이프라인 위에서 계산되도록 한다(설명의 충실도).

사용 (GPU 권장, CPU도 가능):
    python scripts/gradcam_edge.py --arch resnet18 --per-class 2
    python scripts/gradcam_edge.py --arch mobilenet_v3_small --per-class 2
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.train_edge_cnn import CLASSES, _build_model, _load_split  # noqa: E402

CKPT_DIR = ROOT / "models" / "checkpoints" / "edge_cnn"
OUT_DIR = ROOT / "data" / "results" / "gradcam"
_MEAN = np.array([0.485, 0.456, 0.406])
_STD = np.array([0.229, 0.224, 0.225])


def _eval_transform(img_size: int):
    from torchvision import transforms as T

    return T.Compose([
        T.Grayscale(num_output_channels=3),
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(_MEAN.tolist(), _STD.tolist()),
    ])


def _target_layer(model, arch: str):
    """Grad-CAM을 걸 마지막 conv 단계(GAP 직전 특성맵)."""
    if arch == "resnet18":
        return model.layer4[-1]
    if arch == "mobilenet_v3_small":
        return model.features[-1]
    raise ValueError(f"미지원 arch: {arch}")


class GradCAM:
    """예측 클래스 로짓을 타깃 conv 특성맵으로 역전파해 클래스 활성맵을 만든다."""

    def __init__(self, model, target_layer):
        self.model = model
        self.acts = None
        self.grads = None
        target_layer.register_forward_hook(self._fwd)
        target_layer.register_full_backward_hook(self._bwd)

    def _fwd(self, _m, _inp, out):
        self.acts = out.detach()

    def _bwd(self, _m, _gin, gout):
        self.grads = gout[0].detach()

    def __call__(self, x, class_idx: int | None = None):
        import torch
        import torch.nn.functional as F

        self.model.zero_grad()
        logits = self.model(x)  # (1, C)
        probs = F.softmax(logits, dim=1)[0]
        idx = int(logits.argmax(1)) if class_idx is None else class_idx
        logits[0, idx].backward()

        # 채널별 가중치 = 기울기의 공간 평균(GAP), cam = ReLU(Σ w_k A_k)
        weights = self.grads.mean(dim=(2, 3), keepdim=True)  # (1,K,1,1)
        cam = F.relu((weights * self.acts).sum(dim=1, keepdim=True))  # (1,1,h,w)
        cam = F.interpolate(cam, size=x.shape[2:], mode="bilinear", align_corners=False)
        cam = cam[0, 0].cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, idx, float(probs[idx])


def _denorm(x_tensor) -> np.ndarray:
    """정규화된 입력 텐서를 0~1 RGB 이미지로 되돌린다(시각화 배경용)."""
    img = x_tensor[0].cpu().numpy().transpose(1, 2, 0)
    return np.clip(img * _STD + _MEAN, 0, 1)


def main() -> None:
    import torch

    ap = argparse.ArgumentParser(description="엣지 CNN Grad-CAM 시각화")
    ap.add_argument("--arch", default="resnet18", choices=["resnet18", "mobilenet_v3_small"])
    ap.add_argument("--per-class", type=int, default=2, help="클래스당 시각화할 test 이미지 수")
    ap.add_argument("--img-size", type=int, default=224)
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.cm as cm
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = "Malgun Gothic"  # 한글 라벨 렌더(Windows)
    plt.rcParams["axes.unicode_minus"] = False
    from PIL import Image

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = CKPT_DIR / f"{args.arch}.pt"
    if not ckpt.exists():
        raise SystemExit(f"체크포인트 없음: {ckpt} (먼저 train_edge_cnn.py 실행)")

    model = _build_model(args.arch, len(CLASSES), pretrained=False).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    cam_engine = GradCAM(model, _target_layer(model, args.arch))
    tf = _eval_transform(args.img_size)

    # 클래스당 앞에서 per-class장씩 test 이미지 샘플(결정적)
    test = _load_split("test")
    by_class: dict[int, list[str]] = {}
    for path, y in test:
        by_class.setdefault(y, []).append(path)
    samples: list[tuple[str, int]] = []
    for y in range(len(CLASSES)):
        for path in by_class.get(y, [])[: args.per_class]:
            samples.append((path, y))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cols = args.per_class * 2  # (원본 | 오버레이) 쌍을 클래스별 행에
    rows = len(CLASSES)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.4))
    axes = np.atleast_2d(axes)

    manifest = []
    correct = 0
    for k, (path, gt) in enumerate(samples):
        pil = Image.open(path).convert("RGB")
        x = tf(pil).unsqueeze(0).to(device)
        cam, pred, conf = cam_engine(x)
        ok = pred == gt
        correct += int(ok)
        bg = _denorm(x)
        heat = cm.jet(cam)[:, :, :3]
        overlay = np.clip(0.45 * heat + 0.55 * bg, 0, 1)

        r = gt
        c = (k % args.per_class) * 2
        title_col = "green" if ok else "red"
        axes[r, c].imshow(bg)
        axes[r, c].set_title(f"{CLASSES[gt]}", fontsize=8)
        axes[r, c].axis("off")
        axes[r, c + 1].imshow(overlay)
        axes[r, c + 1].set_title(f"→ {CLASSES[pred]} ({conf:.2f})", fontsize=8, color=title_col)
        axes[r, c + 1].axis("off")

        manifest.append({
            "image": str(Path(path).relative_to(ROOT)) if str(ROOT) in str(path) else path,
            "gt": CLASSES[gt], "pred": CLASSES[pred], "confidence": round(conf, 4), "correct": ok,
        })

    fig.suptitle(
        f"Grad-CAM · {args.arch} (NEU test) — 모델이 주목한 영역(빨강=강)", fontsize=12
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    gallery = OUT_DIR / f"gradcam_gallery_{args.arch}.png"
    fig.savefig(gallery, dpi=120)
    plt.close(fig)

    man_path = OUT_DIR / f"gradcam_{args.arch}.json"
    man_path.write_text(json.dumps({
        "arch": args.arch,
        "n_samples": len(samples),
        "correct": correct,
        "accuracy_on_samples": round(correct / len(samples), 4) if samples else 0.0,
        "gallery": str(gallery.relative_to(ROOT)),
        "samples": manifest,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[gradcam] {args.arch}: {len(samples)}장 시각화, 샘플 정확도 {correct}/{len(samples)}")
    print(f"  갤러리: {gallery.relative_to(ROOT)}")
    print(f"  매니페스트: {man_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
