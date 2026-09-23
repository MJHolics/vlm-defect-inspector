"""X-ray 검사기의 OOD 대응 — X-ray 도메인 전이 트랙 (C).

NEU 광학 트랙에서 얻은 결론(B3: softmax confidence 0.68 → Mahalanobis 0.97)이
X-ray 도메인에서도 성립하는가를 같은 방식으로 다시 잰다. 도메인이 바뀌어도 검증
척추가 그대로 서는지 확인하는 것이 목적이다.

**OOD를 무엇으로 잡는가** — 여기서 이 트랙의 값이 갈린다. '다른 데이터셋'을 아무거나
쓰면 숫자는 잘 나오지만 현장에서 실제로 일어나는 일이 아니다. 두 가지를 쓴다:

  OOD-A **빈 지그(공기)**: 검사 대상이 아예 없는데 촬영된 경우. 미투입·오정렬로
    현장에서 실제로 발생한다. 분류기는 이걸 defect가 아니니 **clean으로 자신 있게
    통과**시킬 유인이 있다 — 즉 '불량 없음'이 아니라 '검사를 안 한 것'인데 합격 판정이
    나가는 가짜 합격(false pass). 검사장비에서 가장 위험한 실패 모드다.
  OOD-B **다른 모달리티 오투입**: NEU 광학 표면 결함 이미지. 설비·설정이 잘못 연결된
    경우를 가정한다.

두 스코어를 head-to-head로 비교한다:
  - max softmax probability (현장에서 흔히 쓰는 신뢰도 게이트)
  - Mahalanobis 거리 (penultimate 특징공간, train으로 적합한 클래스별 평균 + 공유 공분산)

사용:
    python scripts/xray_ood.py --ckpt models/checkpoints/xray_orig/resnet18.pt
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SRC = ROOT / "data" / "gdxray" / "Castings kaggle" / "Castings (1)" / "Castings"
OOD_DIR = ROOT / "data" / "gdxray_ood" / "air"
OUT = ROOT / "reports" / "xray"
GATE = 0.80


def _dump_air_patches(test_series: list[str], patch: int, n_target: int, seed: int) -> list[Path]:
    """test 시리즈 이미지에서 '소재가 거의 없는' 패치(공기)를 뽑아 저장.

    train에 쓰인 시리즈는 피한다 — 배경이라도 같은 촬영 조건을 학습에서 본 적이
    있으면 OOD 판정이 관대해질 수 있어서다.
    """
    from PIL import Image
    from scripts.prep_gdxray import _material_mask, _crop

    OOD_DIR.mkdir(parents=True, exist_ok=True)
    have = sorted(OOD_DIR.glob("*.png"))
    if len(have) >= n_target:
        return have[:n_target]

    rng = np.random.default_rng(seed)
    half = patch // 2
    out: list[Path] = list(have)
    pngs = []
    for s in test_series:
        pngs += sorted((SRC / s).glob(f"{s}_*.png"))
    rng.shuffle(pngs)

    for png in pngs:
        if len(out) >= n_target:
            break
        img = np.asarray(Image.open(png).convert("L"))
        mat = _material_mask(img)
        for _ in range(40):
            if len(out) >= n_target:
                break
            cy = rng.integers(half, img.shape[0] - half)
            cx = rng.integers(half, img.shape[1] - half)
            m = _crop(mat, cy, cx, half)
            if m is None or m.mean() > 0.02:      # 소재가 2% 넘게 섞이면 공기 패치가 아니다
                continue
            p = _crop(img, cy, cx, half)
            if p is None or p.std() < 1e-6:        # 완전 균일(포화)한 패치는 제외
                continue
            fp = OOD_DIR / f"air_{png.stem}_{cy}_{cx}.png"
            Image.fromarray(p).save(fp)
            out.append(fp)
    return out


def _features_and_logits(model, paths: list[str], img_size: int, device):
    """penultimate 특징(512-d)과 logits를 함께 뽑는다."""
    import torch
    from torch.utils.data import DataLoader
    from scripts.train_edge_cnn import _DS

    feats: list[np.ndarray] = []
    hook_out = {}

    def _hook(_m, _i, o):
        hook_out["f"] = torch.flatten(o, 1).detach().cpu().numpy()

    h = model.avgpool.register_forward_hook(_hook)
    items = [(p, 0) for p in paths]
    loader = DataLoader(_DS(items, False, img_size), batch_size=64, shuffle=False, num_workers=0)
    logits: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for x, _ in loader:
            z = model(x.to(device))
            logits.append(z.detach().cpu().numpy())
            feats.append(hook_out["f"])
    h.remove()
    return np.concatenate(feats), np.concatenate(logits)


def _softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _auroc(id_scores: np.ndarray, ood_scores: np.ndarray) -> float:
    """OOD가 더 '이상하다'고 클수록 1에 가깝게 — Mann-Whitney U (동점 0.5 처리)."""
    a, b = np.asarray(ood_scores, float), np.asarray(id_scores, float)
    all_v = np.concatenate([a, b])
    order = all_v.argsort()
    ranks = np.empty(len(all_v), float)
    ranks[order] = np.arange(1, len(all_v) + 1)
    # 동점 평균순위
    _, inv, cnt = np.unique(all_v, return_inverse=True, return_counts=True)
    sums = np.zeros(len(cnt))
    np.add.at(sums, inv, ranks)
    ranks = (sums / cnt)[inv]
    r_a = ranks[: len(a)].sum()
    u = r_a - len(a) * (len(a) + 1) / 2
    return float(u / (len(a) * len(b)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="models/checkpoints/xray_orig/resnet18.pt")
    ap.add_argument("--split-dir", default="data/gdxray_processed")
    ap.add_argument("--arch", default="resnet18")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--n-air", type=int, default=400)
    ap.add_argument("--shrinkage", type=float, default=1e-3, help="공분산 정칙화")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # train_edge_cnn 의 CLASSES/_load_split 을 X-ray 설정으로 물린다
    os.environ["EDGE_CLASSES"] = "clean,defect"
    os.environ["EDGE_SPLIT_DIR"] = str(ROOT / args.split_dir)
    import torch
    from scripts.train_edge_cnn import CLASSES, _build_model, _load_split

    OUT.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _build_model(args.arch, len(CLASSES), pretrained=False).to(device)
    model.load_state_dict(torch.load(ROOT / args.ckpt, map_location=device))

    meta = json.loads((ROOT / args.split_dir / "prep_meta.json").read_text(encoding="utf-8"))
    test_series = sorted([s for s, sp in meta["split_of_series"].items() if sp == "test"])
    patch = meta["patch"]

    tr = _load_split("train")
    te = _load_split("test")
    print(f"train {len(tr)} · test(ID) {len(te)} · test 시리즈 {test_series}")

    air = _dump_air_patches(test_series, patch, args.n_air, args.seed)
    print(f"OOD-A 빈 지그(공기) 패치: {len(air)}장")

    neu_split = ROOT / "data" / "processed" / "test.json"
    neu_paths: list[str] = []
    if neu_split.exists():
        recs = json.loads(neu_split.read_text(encoding="utf-8"))
        neu_paths = [str(ROOT / r["image"]) for r in recs]
    print(f"OOD-B NEU 광학 오투입: {len(neu_paths)}장")

    # ── 특징·확률
    f_tr, z_tr = _features_and_logits(model, [p for p, _ in tr], args.img_size, device)
    y_tr = np.array([y for _, y in tr])
    f_id, z_id = _features_and_logits(model, [p for p, _ in te], args.img_size, device)
    f_a, z_a = _features_and_logits(model, [str(p) for p in air], args.img_size, device)
    sets = {"air(빈 지그)": (f_a, z_a)}
    if neu_paths:
        f_b, z_b = _features_and_logits(model, neu_paths, args.img_size, device)
        sets["NEU(광학 오투입)"] = (f_b, z_b)

    # ── Mahalanobis: 클래스별 평균 + 공유 공분산(train으로만 적합)
    mus = np.stack([f_tr[y_tr == c].mean(0) for c in range(len(CLASSES))])
    cen = np.concatenate([f_tr[y_tr == c] - mus[c] for c in range(len(CLASSES))])
    cov = np.cov(cen, rowvar=False) + args.shrinkage * np.eye(f_tr.shape[1])
    prec = np.linalg.pinv(cov)

    def maha(f):
        d = np.stack([np.einsum("ij,jk,ik->i", f - m, prec, f - m) for m in mus], 1)
        return d.min(1)   # 가장 가까운 클래스까지의 거리

    conf_id = _softmax(z_id).max(1)
    m_id = maha(f_id)

    rows = []
    print(f"\nID(정상 X-ray 패치) — 평균 confidence {conf_id.mean():.4f} · "
          f"게이트({GATE}) 통과율 {(conf_id >= GATE).mean():.4f} · Maha 중앙값 {np.median(m_id):.1f}")
    for name, (f_o, z_o) in sets.items():
        c_o = _softmax(z_o).max(1)
        m_o = maha(f_o)
        # confidence는 낮을수록 OOD → 부호를 뒤집어 두 스코어 모두 '클수록 이상함'으로 맞춘다
        au_conf = _auroc(-conf_id, -c_o)
        au_maha = _auroc(m_id, m_o)
        rows.append({
            "ood_set": name, "n": int(len(c_o)),
            "conf_mean": float(c_o.mean()),
            "gate_pass_rate": float((c_o >= GATE).mean()),
            "auroc_softmax": float(au_conf),
            "auroc_mahalanobis": float(au_maha),
            "maha_median": float(np.median(m_o)),
        })
        print(f"\n[{name}] n={len(c_o)}")
        print(f"  평균 confidence {c_o.mean():.4f} · 게이트({GATE}) 통과율 "
              f"{(c_o >= GATE).mean():.4f}  ← 높으면 가짜 합격 위험")
        print(f"  AUROC  softmax {au_conf:.4f}  vs  Mahalanobis {au_maha:.4f}"
              f"   ({'Maha 우세' if au_maha > au_conf else 'softmax 우세'})")

    # ── 물리 게이트: 학습이 아니라 '검사 대상이 거기 있는가'를 먼저 묻는다
    #
    # 빈 지그 문제는 모델의 불확실성 문제가 아니라 **입력 유효성** 문제다. 애초에
    # 학습 패치를 만들 때 공기를 걸러낸 그 판단(Otsu 소재 마스크)을 배포 시 전단
    # 게이트로 그대로 쓴다. 딥 특징이 아니라 패치의 물리적 성질(대비)만 본다.
    # 첫 가설(대비=표준편차)은 데이터가 기각했다 — 투과영상의 공기 영역도 검출기
    # 노이즈로 std가 소재와 비슷하게 나온다(ID 21.1 vs air 19.1, AUROC 0.568).
    # 실제 분리축은 대비가 아니라 **감쇠(밝기)**였다: 소재는 밝고 공기는 어둡다.
    # 두 통계를 모두 재서 기각된 가설도 함께 남긴다.
    def _patch_stats(paths):
        from PIL import Image
        arrs = [np.asarray(Image.open(p).convert("L"), dtype=np.float32) for p in paths]
        return (np.array([a.mean() for a in arrs]), np.array([a.std() for a in arrs]))

    m_idp, s_idp = _patch_stats([p for p, _ in te])
    print("\n── 물리 게이트 ──")
    gate_rows = []
    for name, paths in [("air(빈 지그)", [str(p) for p in air])] + (
            [("NEU(광학 오투입)", neu_paths)] if neu_paths else []):
        m_o, s_o = _patch_stats(paths)
        au_std = _auroc(-s_idp, -s_o)   # 대비 낮을수록 이상
        au_mean = _auroc(-m_idp, -m_o)  # 어두울수록(감쇠 없을수록) 이상
        gate_rows.append({
            "ood_set": name,
            "auroc_patch_std": float(au_std), "auroc_patch_mean": float(au_mean),
            "mean_id_median": float(np.median(m_idp)), "mean_ood_median": float(np.median(m_o)),
        })
        print(f"  [{name}] 대비(std) AUROC {au_std:.4f}  ·  감쇠(mean) AUROC {au_mean:.4f}"
              f"   (밝기 중앙값 ID {np.median(m_idp):.1f} vs OOD {np.median(m_o):.1f})")

    # 운영점: ID를 99% 통과시키는 임계에서 빈 지그를 얼마나 막는가
    thr = float(np.percentile(m_idp, 1))
    m_air, _ = _patch_stats([str(p) for p in air])
    print(f"  운영점 임계 mean≥{thr:.1f} (ID 99% 통과) → 빈 지그 차단율 "
          f"{(m_air < thr).mean():.4f}")

    out = {
        "physical_gate": {"rows": gate_rows, "threshold_mean": thr,
                          "id_pass_rate": float((m_idp >= thr).mean()),
                          "air_block_rate": float((m_air < thr).mean())},
        "ckpt": args.ckpt, "split_dir": args.split_dir, "classes": CLASSES,
        "gate": GATE, "test_series": test_series,
        "id": {"n": int(len(conf_id)), "conf_mean": float(conf_id.mean()),
               "gate_pass_rate": float((conf_id >= GATE).mean()),
               "maha_median": float(np.median(m_id))},
        "ood": rows,
    }
    (OUT / "xray_ood.json").write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                       encoding="utf-8")
    print(f"\n저장: {OUT / 'xray_ood.json'}")


if __name__ == "__main__":
    main()
