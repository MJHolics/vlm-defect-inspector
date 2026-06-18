"""WM-811K(실제 반도체 fab 웨이퍼맵) 데이터셋 다운로드 — 반도체 도메인 전이 트랙.

Kaggle MIR-WM811K(`qingyi/wm811k-wafer-map`)의 `LSWMD.pkl`을 받아
`data/wm811k/`에 푼다. 811,457장 웨이퍼맵 중 라벨된 부분 + 9개 결함패턴
(Center/Donut/Edge-Loc/Edge-Ring/Loc/Random/Scratch/Near-full/none)을 담은
pandas DataFrame 피클이다.

전제: Kaggle API 토큰(`~/.kaggle/kaggle.json`)이 있어야 한다.
  1) https://www.kaggle.com/settings/account → API → Create New Token
  2) 받은 kaggle.json을 ~/.kaggle/kaggle.json 으로 이동

사용:
    python scripts/fetch_wm811k.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEST = ROOT / "data" / "wm811k"
SLUG = "qingyi/wm811k-wafer-map"


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    pkl = DEST / "LSWMD.pkl"
    if pkl.exists():
        print(f"이미 존재: {pkl} ({pkl.stat().st_size / 1024**2:.0f} MB) — 건너뜀")
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
            "Kaggle 인증 실패. ~/.kaggle/kaggle.json 을 확인하세요.\n"
            "  https://www.kaggle.com/settings/account → API → Create New Token\n"
            f"  원인: {e}"
        )

    print(f"다운로드: {SLUG} → {DEST} (~3.5GB, 시간 걸림)")
    api.dataset_download_files(SLUG, path=str(DEST), unzip=True, quiet=False)

    if not pkl.exists():
        # 일부 미러는 파일명이 다를 수 있음 — 받은 내용 안내
        got = [p.name for p in DEST.iterdir()]
        raise SystemExit(f"LSWMD.pkl 을 못 찾음. 받은 파일: {got}")
    print(f"완료: {pkl} ({pkl.stat().st_size / 1024**2:.0f} MB)")


if __name__ == "__main__":
    main()
