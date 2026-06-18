"""WM-811K 전처리 → 학습용 npz — 반도체 도메인 전이 트랙.

LSWMD.pkl(가변 크기 웨이퍼맵 811k장)에서 라벨된 결함패턴만 추려
고정 크기(기본 64x64)로 리사이즈하고 층화 분할해 npz로 저장한다.
웨이퍼맵 픽셀값은 {0=배경, 1=정상 die, 2=결함 die}.

기본은 8개 '결함패턴' 클래스(none 제외)로, 실제 fab에서 의미 있는
'어떤 결함패턴인가' 인식 태스크를 만든다. --include-none 으로 9-class 가능.

사용:
    python scripts/prep_wm811k.py --size 64
    python scripts/prep_wm811k.py --size 64 --include-none
"""
import argparse
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
WM = ROOT / "data" / "wm811k"

DEFECT8 = ["Center", "Donut", "Edge-Loc", "Edge-Ring",
           "Loc", "Near-full", "Random", "Scratch"]


def _load_lswmd(pkl: Path):
    """아주 오래된 pandas(<0.20)로 피클된 LSWMD.pkl을 pandas 2.x에서 로드.

    옛 피클은 제거된 모듈/클래스(pandas.indexes.*, Int64Index/Float64Index)를
    참조하므로 모듈 경로를 현재로 매핑하고 제거된 인덱스 클래스는 pd.Index로
    대체하는 호환 언피클러를 쓴다. (WM-811K의 잘 알려진 호환성 이슈)
    """
    import pickle

    import pandas as pd
    from pandas.core.indexes.base import Index, _new_Index

    class _CompatUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            m = module
            if m.startswith("pandas.indexes"):
                m = m.replace("pandas.indexes", "pandas.core.indexes")
            if m == "pandas.core.index":
                m = "pandas.core.indexes.base"
            try:
                return super().find_class(m, name)
            except (ModuleNotFoundError, AttributeError):
                if name in ("Int64Index", "Float64Index", "UInt64Index", "RangeIndex"):
                    return Index
                if name == "_new_Index":
                    return _new_Index
                raise

    with open(pkl, "rb") as f:
        # encoding=latin1: Python 2에서 만든 피클(바이트열)을 py3에서 읽기 위함
        return _CompatUnpickler(f, encoding="latin1").load()


def _flat_label(v) -> str:
    """failureType 셀(중첩 ndarray/문자열/빈값)에서 단일 라벨 문자열 추출."""
    if isinstance(v, np.ndarray):
        flat = v.ravel()
        if flat.size == 0:
            return ""
        v = flat[0]
    if v is None:
        return ""
    return str(v).strip()


def main():
    ap = argparse.ArgumentParser(description="WM-811K 전처리")
    ap.add_argument("--pkl", type=Path, default=WM / "LSWMD.pkl")
    ap.add_argument("--size", type=int, default=64, help="리사이즈 정사각 변 길이")
    ap.add_argument("--include-none", action="store_true",
                    help="결함없음(none)도 클래스로 포함(9-class). 기본은 결함패턴 8-class")
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--out", type=Path, default=WM / "wafer_prepped.npz")
    args = ap.parse_args()

    if not args.pkl.exists():
        raise SystemExit(f"피클 없음: {args.pkl} — 먼저 scripts/fetch_wm811k.py 실행")

    import pandas as pd
    from PIL import Image
    from sklearn.model_selection import train_test_split

    print(f"로드: {args.pkl}")
    df = _load_lswmd(args.pkl)
    print(f"전체 웨이퍼맵: {len(df):,}")

    classes = DEFECT8 + (["none"] if args.include_none else [])
    cls_idx = {c: i for i, c in enumerate(classes)}

    df = df.copy()
    df["_label"] = df["failureType"].apply(_flat_label)
    df = df[df["_label"].isin(classes)]
    print(f"대상 클래스({len(classes)}) 라벨된 샘플: {len(df):,}")
    print("클래스 분포:")
    for c in classes:
        print(f"  {c:12s} {(df['_label'] == c).sum():>7,}")

    S = args.size
    X = np.zeros((len(df), S, S), dtype=np.uint8)
    y = np.zeros(len(df), dtype=np.int64)
    for i, (wm, lab) in enumerate(zip(df["waferMap"].values, df["_label"].values)):
        arr = np.asarray(wm, dtype=np.uint8)
        img = Image.fromarray(arr).resize((S, S), Image.NEAREST)
        X[i] = np.asarray(img, dtype=np.uint8)
        y[i] = cls_idx[lab]
        if (i + 1) % 5000 == 0:
            print(f"  리사이즈 {i + 1:,}/{len(df):,}")

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=args.test_frac, stratify=y, random_state=42)
    print(f"분할: train {len(Xtr):,} / test {len(Xte):,}")

    np.savez_compressed(
        args.out, X_train=Xtr, y_train=ytr, X_test=Xte, y_test=yte,
        classes=np.array(classes))
    print(f"저장: {args.out} ({args.out.stat().st_size / 1024**2:.0f} MB)")


if __name__ == "__main__":
    main()
