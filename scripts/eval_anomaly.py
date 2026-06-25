"""무지도 이상탐지 평가 — MVTec 카테고리에서 PatchCore/PaDiM 비교 실측.

두 축의 비교를 한 번에 낸다(트랙의 핵심 = "어떤 상황에 어떤 모델을 왜"의 근거):
  ① AD 내부 right-sizing: PatchCore(heavy, wide_resnet50_2) vs PatchCore(light, resnet18+
     강한 coreset 압축) vs PaDiM — 정확도 ↔ 지연·메모리 트레이드오프.
  ② 산출물: image AUROC, pixel AUROC, 단건 latency(ms), memory bank 크기.

이미지 점수=패치 이상점수의 최댓값. pixel AUROC=이상맵을 원본 크기로 올려 GT 마스크와
픽셀단위 비교(정상 test 이미지는 전부 0 마스크).

사용:
    python scripts/eval_anomaly.py --category screw
    python scripts/eval_anomaly.py --category screw --configs patchcore_heavy,padim
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

from scripts import anomaly_detect as ad  # noqa: E402

DATA = ROOT / "data" / "mvtec"
RESULTS = ROOT / "data" / "results"
GALLERY = RESULTS / "anomaly"

# 비교 설정 — 무거움→가벼움. (이름: (클래스, kwargs))
CONFIGS = {
    "patchcore_heavy": ("PatchCore", dict(backbone="wide_resnet50_2", coreset_ratio=0.10)),
    "patchcore_light": ("PatchCore", dict(backbone="resnet18", coreset_ratio=0.01)),
    "padim": ("PaDiM", dict(backbone="resnet18", n_dims=100)),
}


def _category_dir(category: str) -> Path:
    base = DATA / category
    for cand in [base / category, base]:
        if (cand / "train" / "good").is_dir():
            return cand
    raise SystemExit(f"{base} 에 train/good 없음 — scripts/fetch_mvtec.py 먼저 실행")


def load_mvtec(category: str):
    """반환: (train_good_paths, test_items). test_items=[{path,label,defect,mask}]."""
    cat = _category_dir(category)
    train = sorted((cat / "train" / "good").glob("*.png"))
    items = []
    for d in sorted((cat / "test").iterdir()):
        if not d.is_dir():
            continue
        defect = d.name
        for p in sorted(d.glob("*.png")):
            label = 0 if defect == "good" else 1
            mask = cat / "ground_truth" / defect / f"{p.stem}_mask.png"
            items.append({"path": p, "label": label, "defect": defect,
                          "mask": mask if mask.exists() else None})
    return train, items


def _pixel_auroc(maps, items, size: int):
    """이상맵(grid)을 원본 size로 올려 GT 마스크와 픽셀 AUROC. 마스크 없으면(정상) 0."""
    from PIL import Image
    from sklearn.metrics import roc_auc_score

    all_scores, all_labels = [], []
    for m, it in zip(maps, items):
        up = np.asarray(Image.fromarray(m).resize((size, size), Image.BILINEAR))
        if it["mask"] is not None:
            gt = np.asarray(Image.open(it["mask"]).convert("L").resize((size, size))) > 0
        else:
            gt = np.zeros((size, size), dtype=bool)
        all_scores.append(up.ravel())
        all_labels.append(gt.ravel().astype(np.uint8))
    s = np.concatenate(all_scores)
    y = np.concatenate(all_labels)
    if len(np.unique(y)) < 2:
        return float("nan")
    # 픽셀이 너무 많으면 표본추출(속도) — 라벨 비율 유지 무작위.
    if len(y) > 2_000_000:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(y), 2_000_000, replace=False)
        s, y = s[idx], y[idx]
    return float(roc_auc_score(y, s))


def _save_gallery(detector, items, size: int, name: str, n: int = 6):
    """결함 test 이미지 몇 장 + 이상맵 오버레이 갤러리 PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    # 갤러리는 폰트 의존 없는 영문 라벨로(환경마다 한글 폰트 깨짐 방지 — 꼼꼼함).
    plt.rcParams["axes.unicode_minus"] = False

    defects = [it for it in items if it["label"] == 1][:n]
    img_scores, maps, _ = detector.score([it["path"] for it in defects])
    fig, axes = plt.subplots(2, len(defects), figsize=(2.4 * len(defects), 5))
    for j, (it, m) in enumerate(zip(defects, maps)):
        img = Image.open(it["path"]).convert("L").resize((size, size))
        up = np.asarray(Image.fromarray(m).resize((size, size), Image.BILINEAR))
        axes[0, j].imshow(img, cmap="gray"); axes[0, j].set_title(it["defect"], fontsize=8)
        axes[1, j].imshow(img, cmap="gray"); axes[1, j].imshow(up, cmap="jet", alpha=0.5)
        for ax in (axes[0, j], axes[1, j]):
            ax.axis("off")
    fig.suptitle(f"{name}  —  defect anomaly maps (top: input / bottom: overlay)", fontsize=11)
    fig.tight_layout()
    GALLERY.mkdir(parents=True, exist_ok=True)
    out = GALLERY / f"{name}.png"
    fig.savefig(out, dpi=110, bbox_inches="tight"); plt.close(fig)
    return out


def run_config(name: str, category: str, train, items, gallery: bool):
    cls_name, kwargs = CONFIGS[name]
    detector = getattr(ad, cls_name)(**kwargs)
    size = detector.size
    print(f"\n[{name}] {cls_name}{kwargs} — fit (정상 {len(train)}장)…")
    t0 = time.perf_counter()
    detector.fit([str(p) for p in train])
    fit_s = time.perf_counter() - t0

    img_scores, maps, lat = detector.score([str(it["path"]) for it in items])
    labels = np.array([it["label"] for it in items])
    img_auroc = ad.image_auroc(img_scores, labels)
    px_auroc = _pixel_auroc(maps, items, size)
    gallery_path = _save_gallery(detector, items, size, name) if gallery else None
    rep = {
        "config": name, "method": cls_name, "backbone": kwargs.get("backbone"),
        "image_auroc": round(img_auroc, 4), "pixel_auroc": round(px_auroc, 4),
        "latency_ms": round(float(lat.mean()), 1), "bank_size": detector.bank_size,
        "fit_sec": round(fit_s, 1),
    }
    print(f"  image AUROC {rep['image_auroc']} · pixel AUROC {rep['pixel_auroc']} · "
          f"latency {rep['latency_ms']}ms · bank {rep['bank_size']} · fit {rep['fit_sec']}s")
    if gallery_path:
        print(f"  갤러리: {gallery_path.relative_to(ROOT)}")
    return rep


def main():
    ap = argparse.ArgumentParser(description="무지도 이상탐지 비교 평가 (MVTec)")
    ap.add_argument("--category", default="screw")
    ap.add_argument("--configs", default=",".join(CONFIGS),
                    help="쉼표구분 (기본 전체: patchcore_heavy,patchcore_light,padim)")
    ap.add_argument("--no-gallery", dest="gallery", action="store_false")
    ap.set_defaults(gallery=True)
    args = ap.parse_args()

    train, items = load_mvtec(args.category)
    n_def = sum(it["label"] for it in items)
    print(f"MVTec '{args.category}': 정상train {len(train)} · test {len(items)}"
          f"(정상 {len(items)-n_def}/결함 {n_def})")

    reps = [run_config(c.strip(), args.category, train, items, args.gallery)
            for c in args.configs.split(",") if c.strip()]

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / f"anomaly_{args.category}.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(reps[0].keys()))
        w.writeheader(); w.writerows(reps)
    print(f"\n결과표 저장: {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
