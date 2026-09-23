"""CT 재구성(Radon/FBP) — X-ray·CT 검사장비 도메인 전이 트랙 (A).

기존 트랙은 전부 **완성된 이미지**를 받아 판정했다(NEU 광학, WM-811K 웨이퍼맵).
CT 검사장비는 다르다 — 이미지가 주어지는 게 아니라 **투영(sinogram)에서 만들어진다.**
그 재구성 단계가 뒷단 결함검출의 상한을 정하는데, 지금까지 그 앞단을 다뤄본 적이 없었다.

이 스크립트는 검사장비의 실제 트레이드오프 하나를 정량화한다:

    투영 각도 수 ↓  →  스캔 시간 ↓ (= 검사 처리량 ↑)  →  화질 ↓  →  결함을 놓친다

어디까지 줄여도 되는가? 그걸 "화질이 나빠진다"는 형용사가 아니라 곡선으로 답한다.

**측정 설계의 핵심**: 전역 화질(RMSE·SSIM)과 **결함 검출성(CNR)을 따로 잰다.**
검사장비가 파는 것은 이미지가 아니라 결함을 찾아내는 능력이라, 전역 지표가 완만하게
나빠지는 구간에서도 검출성은 이미 무너져 있을 수 있다. 지표를 하나만 보면 그걸 놓친다.

CNR(contrast-to-noise ratio) = |mean(결함ROI) - mean(배경ROI)| / std(배경ROI)
방사선 검사에서 결함 검출성을 재는 표준 지표. Rose 기준으로 CNR≈3~5 미만이면
사람이든 알고리즘이든 신뢰성 있게 검출하기 어렵다고 본다.

팬텀: Shepp-Logan(CT 재구성의 표준 벤치마크)에 균질 영역을 골라 인공 void(기공)를
심는다. 주조·용접·배터리 X-ray에서 가장 흔한 결함 형태가 void다.

사용:
    python scripts/ct_reconstruction.py                    # 기본 스윕
    python scripts/ct_reconstruction.py --angles 180 90 45 20 10 --repeats 5
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "reports" / "ct"

# 검사장비에서 실제로 조절하는 축들
DEFAULT_ANGLES = [180, 120, 90, 60, 45, 30, 20, 15, 10]
DEFAULT_FILTERS = ["ramp", "shepp-logan", "hann"]
DEFAULT_NOISE = [0.0, 0.02, 0.05]  # sinogram 상대 표준편차(검출기 노이즈)

PHANTOM_SIZE = 256
DEFECT_RADIUS = 6          # px — 작은 기공
DEFECT_CONTRAST = -0.06    # 주변 대비 감쇠계수 감소(void = 재료 없음 = 덜 흡수)
ROSE_THRESHOLD = 4.0       # CNR 검출 가능 경계(Rose criterion 중앙값)


def _build_phantom(size: int = PHANTOM_SIZE):
    """Shepp-Logan에 인공 void를 심고, (팬텀, 결함ROI마스크, 배경ROI마스크)를 낸다.

    결함은 **균질한 영역**에 심어야 CNR이 의미를 갖는다(배경 std가 재구성 노이즈만
    반영하도록). 팬텀 내부에서 국소 표준편차가 가장 작은 위치를 자동 탐색한다.
    """
    from skimage.data import shepp_logan_phantom
    from skimage.transform import resize

    base = resize(shepp_logan_phantom(), (size, size), anti_aliasing=True)

    yy, xx = np.mgrid[:size, :size]
    cy = cx = size / 2.0
    # 팬텀 본체 안쪽(경계 아티팩트 회피 위해 반지름 여유)
    inside = (yy - cy) ** 2 + (xx - cx) ** 2 < (size * 0.30) ** 2

    # 국소 표준편차가 최소인 지점 = 가장 균질한 곳
    from scipy.ndimage import uniform_filter

    win = 2 * (DEFECT_RADIUS * 3) + 1
    local_mean = uniform_filter(base, win)
    local_var = uniform_filter(base**2, win) - local_mean**2
    local_std = np.sqrt(np.clip(local_var, 0, None))
    local_std[~inside] = np.inf
    dy, dx = np.unravel_index(np.argmin(local_std), local_std.shape)

    rr = (yy - dy) ** 2 + (xx - dx) ** 2
    defect_mask = rr <= DEFECT_RADIUS**2
    # 배경 ROI = 결함을 둘러싼 환형(같은 재질, 같은 재구성 조건)
    bg_mask = (rr > (DEFECT_RADIUS * 2) ** 2) & (rr <= (DEFECT_RADIUS * 4) ** 2)

    phantom = base.copy()
    phantom[defect_mask] += DEFECT_CONTRAST
    return phantom, defect_mask, bg_mask, (int(dy), int(dx))


def _cnr(img: np.ndarray, defect_mask: np.ndarray, bg_mask: np.ndarray) -> float:
    """결함 검출성. 배경 std가 0에 수렴하면(노이즈 없는 이상적 재구성) 발산하므로 상한을 둔다."""
    d = float(img[defect_mask].mean())
    b = float(img[bg_mask].mean())
    s = float(img[bg_mask].std())
    if s < 1e-9:
        return float("inf")
    return abs(d - b) / s


def _reconstruct(phantom, n_angles, filter_name, noise_sigma, rng):
    """전방 투영 → 검출기 노이즈 → FBP 재구성. (재구성이미지, 소요초)를 낸다."""
    from skimage.transform import radon, iradon

    theta = np.linspace(0.0, 180.0, n_angles, endpoint=False)
    sino = radon(phantom, theta=theta, circle=True)

    if noise_sigma > 0:
        sino = sino + rng.normal(0.0, noise_sigma * sino.max(), sino.shape)

    t0 = time.perf_counter()
    recon = iradon(
        sino, theta=theta, filter_name=filter_name,
        circle=True, output_size=phantom.shape[0],
    )
    return recon, time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--angles", type=int, nargs="+", default=DEFAULT_ANGLES)
    ap.add_argument("--filters", nargs="+", default=DEFAULT_FILTERS)
    ap.add_argument("--noise", type=float, nargs="+", default=DEFAULT_NOISE)
    ap.add_argument("--repeats", type=int, default=5, help="노이즈 시드 반복(평균±표준편차)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from skimage.metrics import structural_similarity

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    phantom, defect_mask, bg_mask, dpos = _build_phantom()
    drange = float(phantom.max() - phantom.min())

    print(f"팬텀 {PHANTOM_SIZE}x{PHANTOM_SIZE} · 결함 중심 {dpos} · 반지름 {DEFECT_RADIUS}px "
          f"· 대비 {DEFECT_CONTRAST:+.3f}")
    print(f"결함ROI {int(defect_mask.sum())}px · 배경ROI {int(bg_mask.sum())}px")
    print(f"참값 CNR(노이즈 없는 팬텀) = {_cnr(phantom, defect_mask, bg_mask):.2f}\n")

    rows = []
    total = len(args.angles) * len(args.filters) * len(args.noise)
    done = 0
    for noise in args.noise:
        for filt in args.filters:
            for n_ang in args.angles:
                # 노이즈 0이면 반복이 무의미(결정론적)
                reps = 1 if noise == 0 else args.repeats
                rmses, ssims, cnrs, secs = [], [], [], []
                for r in range(reps):
                    rng = np.random.default_rng(args.seed + r)
                    recon, sec = _reconstruct(phantom, n_ang, filt, noise, rng)
                    rmses.append(float(np.sqrt(np.mean((recon - phantom) ** 2))))
                    ssims.append(float(structural_similarity(phantom, recon, data_range=drange)))
                    cnrs.append(_cnr(recon, defect_mask, bg_mask))
                    secs.append(sec)
                rows.append({
                    "noise": noise, "filter": filt, "n_angles": n_ang, "repeats": reps,
                    "rmse": float(np.mean(rmses)), "rmse_std": float(np.std(rmses)),
                    "ssim": float(np.mean(ssims)), "ssim_std": float(np.std(ssims)),
                    "cnr": float(np.mean(cnrs)), "cnr_std": float(np.std(cnrs)),
                    "recon_sec": float(np.mean(secs)),
                    "detectable": bool(np.mean(cnrs) >= ROSE_THRESHOLD),
                })
                done += 1
                print(f"[{done:3d}/{total}] noise={noise:.2f} {filt:12s} angles={n_ang:3d} "
                      f"RMSE={rows[-1]['rmse']:.4f} SSIM={rows[-1]['ssim']:.4f} "
                      f"CNR={rows[-1]['cnr']:6.2f} {'검출가능' if rows[-1]['detectable'] else '검출불가'}")

    out = {
        "config": {
            "phantom_size": PHANTOM_SIZE, "defect_radius": DEFECT_RADIUS,
            "defect_contrast": DEFECT_CONTRAST, "defect_center": dpos,
            "rose_threshold": ROSE_THRESHOLD,
            "angles": args.angles, "filters": args.filters, "noise": args.noise,
            "repeats": args.repeats, "seed": args.seed,
        },
        "phantom_cnr": _cnr(phantom, defect_mask, bg_mask),
        "rows": rows,
    }
    (OUT_DIR / "ct_sweep.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n저장: {OUT_DIR / 'ct_sweep.json'} ({len(rows)}행)")


if __name__ == "__main__":
    main()
