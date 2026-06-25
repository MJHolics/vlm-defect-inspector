"""MVTec AD(무지도 산업 이상탐지 표준 벤치) 카테고리 다운로드 — 이상탐지 트랙.

NEU-DET엔 정상(양품) 이미지가 없어(6클래스 전부 결함) 무지도 AD를 못 한다. MVTec AD는
카테고리별로 **정상 train + 결함 test + 픽셀 ground-truth 마스크**를 제공하는 표준셋이다.
금속 결을 유지하려고 기본 카테고리를 `screw`(금속 나사)로 둔다.

`scripts/fetch_wm811k.py`의 Kaggle 패턴을 따른다 — Kaggle API 토큰
(`~/.kaggle/kaggle.json`)이 있어야 한다.

캐노니컬 MVTec 폴더 구조(다운로드 후):
    data/mvtec/<category>/train/good/*.png         (정상만)
    data/mvtec/<category>/test/good/*.png           (정상)
    data/mvtec/<category>/test/<defect>/*.png        (결함)
    data/mvtec/<category>/ground_truth/<defect>/*_mask.png

사용:
    python scripts/fetch_mvtec.py                         # screw (기본)
    python scripts/fetch_mvtec.py --kaggle-ref <ref> --category <name>
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEST_ROOT = ROOT / "data" / "mvtec"

# 카테고리 → (Kaggle ref, 압축 해제 후 카테고리 폴더가 들어있는지). screw=금속, 캐노니컬 구조.
DEFAULT_REFS = {
    "screw": "thomasdubail/screwanomalies-detection",
}


def _find_category_dir(base: Path, category: str) -> Path | None:
    """압축 해제 결과에서 train/good 를 가진 카테고리 루트를 찾는다(중첩 대비)."""
    for cand in [base / category, base]:
        if (cand / "train" / "good").is_dir():
            return cand
    for p in base.rglob("train"):
        if (p / "good").is_dir():
            return p.parent
    return None


def main():
    ap = argparse.ArgumentParser(description="MVTec AD 카테고리 다운로드")
    ap.add_argument("--category", default="screw", help="카테고리명 (기본 screw=금속)")
    ap.add_argument("--kaggle-ref", default=None, help="Kaggle 데이터셋 ref (미지정시 기본 매핑)")
    args = ap.parse_args()

    ref = args.kaggle_ref or DEFAULT_REFS.get(args.category)
    if not ref:
        raise SystemExit(f"--kaggle-ref 를 지정하세요 ('{args.category}' 기본 매핑 없음)")

    dest = DEST_ROOT / args.category
    if _find_category_dir(dest, args.category):
        print(f"이미 존재: {dest} — 건너뜀")
        return

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except Exception as e:
        raise SystemExit(f"kaggle 패키지 필요: pip install kaggle ({e})")

    api = KaggleApi()
    try:
        api.authenticate()
    except Exception as e:
        raise SystemExit(
            "Kaggle 인증 실패. ~/.kaggle/kaggle.json 확인.\n"
            "  https://www.kaggle.com/settings/account → API → Create New Token\n"
            f"  원인: {e}"
        )

    dest.mkdir(parents=True, exist_ok=True)
    print(f"다운로드: {ref} → {dest}")
    api.dataset_download_files(ref, path=str(dest), unzip=True, quiet=False)

    cat_dir = _find_category_dir(dest, args.category)
    if not cat_dir:
        got = [p.name for p in dest.iterdir()]
        raise SystemExit(f"train/good 를 못 찾음. 받은 항목: {got}")
    n_train = len(list((cat_dir / "train" / "good").glob("*.*")))
    n_test = len(list((cat_dir / "test").rglob("*.*"))) if (cat_dir / "test").is_dir() else 0
    print(f"완료: {cat_dir}  (정상 train {n_train}장, test {n_test}장)")


if __name__ == "__main__":
    main()
