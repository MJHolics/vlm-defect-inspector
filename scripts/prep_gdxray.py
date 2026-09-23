"""GDXray Castings(주조품 X-ray) → 패치 분류 데이터셋 — X-ray 도메인 전이 트랙 (B).

GDXray는 주조·용접 방사선 검사의 공개 표준 데이터셋이다. 여기서는 Castings 중
어노테이션(BoundingBox.mat)이 있는 36개 시리즈, bbox 1,545개를 쓴다.

기존 트랙(NEU 광학, WM-811K 웨이퍼맵)과 같은 레코드 포맷으로 뱉어, 학습·Conformal·
Calibration·OOD 스크립트를 **코드 수정 없이** 그대로 태우는 것이 목적이다.

────────────────────────────────────────────────────────────
설계 판단 2가지 — 여기서 틀리면 정확도가 높아도 의미가 없다
────────────────────────────────────────────────────────────

**(1) 음성 패치를 어디서 뽑는가.** 이미지 전체에서 무작위로 뽑으면 음성 대부분이
공기(배경)라, 모델은 '결함이 있는가'가 아니라 '여기 금속이 있는가'를 배운다. 정확도는
잘 나오지만 검사기로는 쓸모가 없다. 그래서 음성은 **같은 이미지에서, 같은 결함 주변
환형(annulus)**에서만 뽑는다 — 노출·재질·두께가 같은 조건으로 통제되므로 모델이
쓸 수 있는 단서는 결함 텍스처뿐이다. 추가로 Otsu 임계로 소재 영역 비율을 확인해
공기가 섞인 패치는 버린다.

**(2) 어떻게 나누는가.** 같은 주조품 시리즈(C0001 등)는 같은 부품을 각도만 바꿔 찍은
연속 촬영이라, 패치 단위로 무작위 분할하면 **거의 같은 그림이 train과 test에 동시에**
들어간다(누수). 분할은 반드시 **시리즈 단위**로 한다. 시리즈별 bbox 수가 9~248로
편차가 커서, 목표 비율에 맞게 greedy로 배분한다.

사용:
    python scripts/prep_gdxray.py                  # 기본(패치 64px, 음성 2배)
    python scripts/prep_gdxray.py --patch 64 --neg-per-pos 2
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "gdxray" / "Castings kaggle" / "Castings (1)" / "Castings"
OUT_IMG = ROOT / "data" / "gdxray_patches"
OUT_SPLIT = ROOT / "data" / "gdxray_processed"

CLASSES = ["clean", "defect"]
SPLIT_TARGET = {"train": 0.70, "val": 0.15, "test": 0.15}


def _load_boxes(series_dir: Path) -> dict[int, list[tuple[float, float, float, float]]]:
    """시리즈의 BoundingBox*.mat을 병합 → {이미지번호: [(x1,x2,y1,y2), ...]}."""
    import scipy.io as sio

    per_img: dict[int, set] = defaultdict(set)
    for mat in sorted(series_dir.glob("BoundingBox*.mat")):
        bb = sio.loadmat(str(mat))["bb"]
        for row in bb:
            idx = int(round(row[0]))
            # 중복 어노테이션(여러 mat) 제거를 위해 소수 둘째자리로 반올림해 집합에 넣는다
            per_img[idx].add(tuple(np.round(row[1:5], 2)))
    return {k: [tuple(map(float, b)) for b in v] for k, v in per_img.items()}


def _material_mask(img: np.ndarray) -> np.ndarray:
    """Otsu로 소재(감쇠가 있는 영역) 분리.

    소재가 밝게 나올지 어둡게 나올지는 촬영·현상 방식에 따라 달라지므로 가정하지 않고,
    **경계에 많이 닿는 쪽을 배경으로** 판단한다. (GDXray Castings에서는 실측상 소재가
    밝게 나온다 — 패치 밝기 중앙값 소재 175 vs 공기 56.)
    """
    from skimage.filters import threshold_otsu

    try:
        t = threshold_otsu(img)
    except Exception:
        return np.ones_like(img, dtype=bool)
    # 밝은 쪽이 배경(공기)인지 소재인지 면적으로 판단 — 배경이 더 넓다고 가정하지 않고
    # 두 영역 중 '경계에 많이 닿는 쪽'을 배경으로 본다.
    dark = img < t
    border = np.zeros_like(dark)
    border[0, :] = border[-1, :] = border[:, 0] = border[:, -1] = True
    dark_border = (dark & border).sum() / max(border.sum(), 1)
    return ~dark if dark_border > 0.5 else dark


def _crop(img: np.ndarray, cy: float, cx: float, half: int):
    """(cy,cx) 중심 정사각 크롭. 경계를 벗어나면 None."""
    y0, y1 = int(round(cy)) - half, int(round(cy)) + half
    x0, x1 = int(round(cx)) - half, int(round(cx)) + half
    if y0 < 0 or x0 < 0 or y1 > img.shape[0] or x1 > img.shape[1]:
        return None
    return img[y0:y1, x0:x1]


def _overlaps(rect, boxes, margin: float) -> bool:
    """rect=(y0,y1,x0,x1)가 어떤 bbox와도 (margin 여유 포함) 겹치지 않아야 음성."""
    y0, y1, x0, x1 = rect
    for (bx1, bx2, by1, by2) in boxes:
        if (x0 < bx2 + margin and x1 > bx1 - margin
                and y0 < by2 + margin and y1 > by1 - margin):
            return True
    return False


def _assign_splits(counts: dict[str, int], seed: int) -> dict[str, str]:
    """시리즈를 bbox 수 기준 greedy로 배분 — 목표 비율에 가장 모자란 split에 큰 것부터."""
    total = sum(counts.values())
    want = {k: v * total for k, v in SPLIT_TARGET.items()}
    got = {k: 0.0 for k in SPLIT_TARGET}
    rng = np.random.default_rng(seed)
    order = sorted(counts, key=lambda s: (-counts[s], s))
    # 동점 흔들기(결정론 유지: seed 고정)
    order = list(order)
    rng.shuffle(order[len(order) // 2:])
    out = {}
    for s in order:
        pick = max(got, key=lambda k: (want[k] - got[k]) / max(want[k], 1e-9))
        out[s] = pick
        got[pick] += counts[s]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patch", type=int, default=64, help="패치 한 변(px)")
    ap.add_argument("--neg-per-pos", type=int, default=2)
    ap.add_argument("--min-material", type=float, default=0.90,
                    help="패치 내 소재 픽셀 최소 비율(공기 패치 배제)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--clahe", action="store_true",
                    help="CLAHE 대비제한 히스토그램 평활화를 적용해 별도 디렉터리로 저장. "
                         "소재 마스크와 패치 좌표는 원본 기준으로 계산하므로 "
                         "원본판과 패치 위치가 정확히 일치한다(통제된 비교)")
    ap.add_argument("--clip-limit", type=float, default=0.01, help="CLAHE clip limit")
    args = ap.parse_args()

    global OUT_IMG, OUT_SPLIT
    if args.clahe:
        OUT_IMG = OUT_IMG.with_name(OUT_IMG.name + "_clahe")
        OUT_SPLIT = OUT_SPLIT.with_name(OUT_SPLIT.name + "_clahe")

    from PIL import Image

    if not SRC.exists():
        raise SystemExit(f"GDXray 원본이 없다: {SRC}")

    half = args.patch // 2
    rng = np.random.default_rng(args.seed)

    series = [d for d in sorted(SRC.iterdir())
              if d.is_dir() and list(d.glob("BoundingBox*.mat"))]
    boxes_by_series = {d.name: _load_boxes(d) for d in series}
    counts = {name: sum(len(v) for v in bx.values())
              for name, bx in boxes_by_series.items()}
    counts = {k: v for k, v in counts.items() if v > 0}
    split_of = _assign_splits(counts, args.seed)

    print(f"시리즈 {len(counts)}개 · bbox {sum(counts.values())}개")
    for sp in SPLIT_TARGET:
        ss = [s for s in counts if split_of[s] == sp]
        print(f"  {sp:5s}: 시리즈 {len(ss):2d}개 · bbox {sum(counts[s] for s in ss):4d}  {sorted(ss)}")

    OUT_IMG.mkdir(parents=True, exist_ok=True)
    OUT_SPLIT.mkdir(parents=True, exist_ok=True)
    recs = {k: [] for k in SPLIT_TARGET}
    stat = defaultdict(int)

    for d in series:
        name = d.name
        if name not in counts:
            continue
        sp = split_of[name]
        for img_idx, bxs in sorted(boxes_by_series[name].items()):
            png = d / f"{name}_{img_idx:04d}.png"
            if not png.exists():
                stat["이미지없음"] += 1
                continue
            img = np.asarray(Image.open(png).convert("L"))
            # 소재 마스크·패치 좌표는 항상 원본에서 뽑는다 → CLAHE판과 위치가 동일해진다.
            mat = _material_mask(img)
            if args.clahe:
                from skimage.exposure import equalize_adapthist

                img = (equalize_adapthist(img, clip_limit=args.clip_limit) * 255
                       ).astype(np.uint8)

            for bi, (x1, x2, y1, y2) in enumerate(bxs):
                cy, cx = (y1 + y2) / 2.0, (x1 + x2) / 2.0
                pos = _crop(img, cy, cx, half)
                if pos is None:
                    stat["양성_경계밖"] += 1
                    continue
                stem = f"{name}_{img_idx:04d}_{bi:03d}"
                fp = OUT_IMG / f"{stem}_defect.png"
                Image.fromarray(pos).save(fp)
                recs[sp].append({
                    "image": str(fp.relative_to(ROOT)).replace("\\", "/"),
                    "series": name,
                    "conversations": [
                        {"role": "user", "content": "이 X-ray 패치에 주조 결함이 있는가?"},
                        {"role": "assistant", "content": json.dumps({"type": "defect"}, ensure_ascii=False)},
                    ],
                })
                stat["양성"] += 1

                # 음성: 같은 이미지, 같은 결함 주변 환형에서
                made = 0
                for _ in range(60):
                    if made >= args.neg_per_pos:
                        break
                    ang = rng.uniform(0, 2 * np.pi)
                    rad = rng.uniform(1.5 * args.patch, 4.0 * args.patch)
                    ny, nx = cy + rad * np.sin(ang), cx + rad * np.cos(ang)
                    rect = (ny - half, ny + half, nx - half, nx + half)
                    if _overlaps(rect, bxs, margin=half * 0.5):
                        continue
                    neg = _crop(img, ny, nx, half)
                    if neg is None:
                        continue
                    m = _crop(mat, ny, nx, half)
                    if m is None or m.mean() < args.min_material:
                        stat["음성_소재부족"] += 1
                        continue
                    fpn = OUT_IMG / f"{stem}_clean{made}.png"
                    Image.fromarray(neg).save(fpn)
                    recs[sp].append({
                        "image": str(fpn.relative_to(ROOT)).replace("\\", "/"),
                        "series": name,
                        "conversations": [
                            {"role": "user", "content": "이 X-ray 패치에 주조 결함이 있는가?"},
                            {"role": "assistant", "content": json.dumps({"type": "clean"}, ensure_ascii=False)},
                        ],
                    })
                    made += 1
                    stat["음성"] += 1
                if made < args.neg_per_pos:
                    stat["음성_부족"] += 1

    for sp, rs in recs.items():
        rng.shuffle(rs)
        (OUT_SPLIT / f"{sp}.json").write_text(
            json.dumps(rs, ensure_ascii=False, indent=1), encoding="utf-8")
        n_def = sum(1 for r in rs if '"defect"' in r["conversations"][1]["content"])
        print(f"{sp:5s}: {len(rs):5d}패치 (defect {n_def}, clean {len(rs)-n_def})")

    print("\n집계:", dict(stat))
    (OUT_SPLIT / "prep_meta.json").write_text(json.dumps({
        "classes": CLASSES, "patch": args.patch, "neg_per_pos": args.neg_per_pos,
        "min_material": args.min_material, "seed": args.seed,
        "split_of_series": split_of, "bbox_per_series": counts,
        "stats": dict(stat),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"저장: {OUT_SPLIT}")


if __name__ == "__main__":
    main()
