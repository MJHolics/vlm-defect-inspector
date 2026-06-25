"""무지도 이상탐지(unsupervised anomaly detection) — PatchCore · PaDiM.

레포의 다른 트랙은 전부 *지도학습 분류*(VLM/CNN)다. 실제 제조 검사에서는 결함 라벨이
귀해서, **정상(양품) 이미지만으로 학습하고 그로부터의 이탈을 탐지**하는 무지도 AD가
지배적이다. 이 모듈이 그 패러다임을 풀에 추가한다.

두 방법:
- **PatchCore**: 정상 패치 feature들을 memory bank에 모으고(greedy coreset로 압축),
  테스트 패치의 최근접 거리로 이상점수를 낸다. SOTA급, 메모리뱅크가 무겁다.
- **PaDiM**: 패치 위치별로 정상 feature의 가우시안(평균·공분산)을 적합하고,
  Mahalanobis 거리로 이상점수를 낸다. 가볍고 학습이 빠르다.

설계 원칙:
- **새 무거운 패키지 없음** — torchvision 백본 + sklearn kNN + numpy만 사용
  (anomalib·faiss·timm 안 씀; 레포의 "albumentations 대신 torchvision" 미니멀 철학과 정합).
- **무거움/가벼움 노브**(고객 니즈 조율 실증): `backbone`(wide_resnet50_2 무거움 ↔ resnet18
  가벼움), `coreset_ratio`(memory bank 압축률).
- 순수 수치 로직(coreset·kNN·mahalanobis)은 백본·네트워크와 분리 → 단위테스트 가능.

사용:
    python scripts/anomaly_detect.py --smoke           # 합성 feature로 자기점검
    (학습·평가 실데이터 경로는 scripts/eval_anomaly.py가 호출)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# 윈도우 콘솔(cp949) 유니코드 안전.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass


# ───────────────────────── 순수 수치 로직 (백본 불요, 테스트 대상) ─────────────────────────

def greedy_coreset_indices(features: np.ndarray, ratio: float, seed: int = 0) -> np.ndarray:
    """Greedy k-center coreset 서브샘플 인덱스.

    memory bank를 ratio 비율로 압축하되, 무작위가 아니라 **이미 뽑힌 점들에서 가장 먼**
    점을 반복 선택해 feature 공간을 고르게 덮는다(PatchCore 코어셋). 이게 무거움/가벼움
    조율의 핵심 노브 — ratio를 낮추면 메모리·지연이 줄고 정확도는 거의 유지된다.

    features: (N, D). 반환: 선택된 행 인덱스 (M = ceil(N*ratio),).
    """
    n = len(features)
    if ratio >= 1.0 or n == 0:
        return np.arange(n)
    m = max(1, int(np.ceil(n * ratio)))
    rng = np.random.default_rng(seed)
    start = int(rng.integers(n))
    selected = [start]
    # 각 점에서 '가장 가까운 선택점까지의 거리'를 유지하며 매번 그 최댓값을 새로 선택.
    min_dist = np.linalg.norm(features - features[start], axis=1)
    for _ in range(1, m):
        nxt = int(np.argmax(min_dist))
        selected.append(nxt)
        d = np.linalg.norm(features - features[nxt], axis=1)
        min_dist = np.minimum(min_dist, d)
    return np.array(selected, dtype=np.int64)


def knn_anomaly_scores(test_feats: np.ndarray, bank: np.ndarray, k: int = 1):
    """테스트 패치별 memory bank 최근접 거리(이상점수)와 그 이미지 점수.

    test_feats: (P, D) 한 이미지의 패치 feature들. bank: (M, D).
    반환: (patch_scores (P,), image_score float). 이미지 점수 = 패치 최대 거리.
    """
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=k, algorithm="auto").fit(bank)
    dist, _ = nn.kneighbors(test_feats)      # (P, k)
    patch_scores = dist.mean(axis=1)         # k>1이면 평균
    return patch_scores, float(patch_scores.max())


def fit_padim_gaussian(feats_per_pos: np.ndarray, eps: float = 0.01):
    """패치 위치별 가우시안 적합. feats_per_pos: (N, L, D) (N=이미지수, L=위치수, D=차원).

    반환: mean (L, D), inv_cov (L, D, D). 공분산에 eps*I 정칙화.
    """
    n, length, d = feats_per_pos.shape
    mean = feats_per_pos.mean(axis=0)                       # (L, D)
    inv_cov = np.empty((length, d, d), dtype=np.float64)
    ident = np.eye(d)
    for pos in range(length):
        x = feats_per_pos[:, pos, :] - mean[pos]           # (N, D)
        cov = (x.T @ x) / max(n - 1, 1) + eps * ident
        inv_cov[pos] = np.linalg.inv(cov)
    return mean.astype(np.float64), inv_cov


def mahalanobis_scores(feats: np.ndarray, mean: np.ndarray, inv_cov: np.ndarray):
    """위치별 Mahalanobis 거리. feats: (L, D). 반환: (patch_scores (L,), image_score)."""
    length = feats.shape[0]
    scores = np.empty(length, dtype=np.float64)
    for pos in range(length):
        delta = feats[pos] - mean[pos]
        scores[pos] = float(np.sqrt(max(delta @ inv_cov[pos] @ delta, 0.0)))
    return scores, float(scores.max())


def image_auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """image-level AUROC (labels: 0=정상, 1=이상)."""
    from sklearn.metrics import roc_auc_score

    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


# ───────────────────────── 백본 feature 추출 (torchvision) ─────────────────────────

_BACKBONE_LAYERS = {
    # (layer2 채널, layer3 채널) — concat 차원 = 합
    "wide_resnet50_2": ("layer2", "layer3"),
    "resnet18": ("layer2", "layer3"),
}


class BackboneFeatures:
    """torchvision ResNet 계열에서 중간 feature(layer2+layer3)를 뽑아 patch feature로 정렬.

    layer3를 layer2 해상도로 업샘플해 채널을 concat → 각 공간 위치가 하나의 패치 벡터.
    무거움/가벼움: wide_resnet50_2(무거움·고차원) ↔ resnet18(가벼움·저차원).
    """

    def __init__(self, backbone: str = "wide_resnet50_2", device: str | None = None):
        import torch
        import torchvision.models as tvm

        if backbone not in _BACKBONE_LAYERS:
            raise ValueError(f"지원 백본: {list(_BACKBONE_LAYERS)} (받음: {backbone})")
        self.backbone = backbone
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        weights = "IMAGENET1K_V1"
        model = getattr(tvm, backbone)(weights=weights)
        model.eval().to(self.device)
        self.model = model
        self._feats: dict[str, "torch.Tensor"] = {}
        l2, l3 = _BACKBONE_LAYERS[backbone]
        getattr(model, l2).register_forward_hook(self._hook(l2))
        getattr(model, l3).register_forward_hook(self._hook(l3))
        self._l2, self._l3 = l2, l3

    def _hook(self, name):
        def fn(_m, _i, out):
            self._feats[name] = out
        return fn

    @property
    def patch_grid(self) -> int:
        """layer2 출력 한 변의 패치 수 (입력 224 기준 28)."""
        return 28

    def extract(self, batch):
        """batch: (B,3,H,W) torch tensor(정규화 완료). 반환 (B, L, D) numpy — L=grid^2."""
        import torch
        import torch.nn.functional as F

        with torch.no_grad():
            self._feats.clear()
            self.model(batch.to(self.device))
            f2 = self._feats[self._l2]                      # (B,C2,h,w)
            f3 = self._feats[self._l3]                      # (B,C3,h/2,w/2)
            f3 = F.interpolate(f3, size=f2.shape[-2:], mode="bilinear", align_corners=False)
            feat = torch.cat([f2, f3], dim=1)               # (B, C2+C3, h, w)
            # 주: PatchCore의 국소 이웃평균(avg_pool)을 실험했으나 screw처럼 결함이 작은
            # 카테고리에선 신호를 흐려 image AUROC가 외려 떨어졌다(0.818→0.803) → 미적용.
            b, c, h, w = feat.shape
            feat = feat.permute(0, 2, 3, 1).reshape(b, h * w, c)  # (B, L, D)
            return feat.cpu().numpy().astype(np.float32)


# ───────────────────────── 전처리 + 고수준 탐지기 (백본 사용) ─────────────────────────

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def preprocess(paths, size: int = 224):
    """이미지 경로 리스트 → (B,3,size,size) torch tensor (ImageNet 정규화). 흑백도 3채널로."""
    import torch
    from PIL import Image
    from torchvision.transforms import v2 as T

    tf = T.Compose([
        T.Resize((size, size)),
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    imgs = [tf(Image.open(p).convert("RGB")) for p in paths]
    return torch.stack(imgs)


def _extract_all(backbone: "BackboneFeatures", paths, size: int, batch: int = 16):
    """경로 리스트의 patch feature를 배치로 모아 (N, L, D) numpy 반환."""
    out = []
    for i in range(0, len(paths), batch):
        out.append(backbone.extract(preprocess(paths[i:i + batch], size)))
    return np.concatenate(out, axis=0)


class PatchCore:
    """정상 패치 memory bank + 최근접 거리 이상점수. coreset로 무거움↔가벼움 조율."""

    def __init__(self, backbone: str = "wide_resnet50_2", coreset_ratio: float = 0.1,
                 max_pool: int = 20000, size: int = 224, seed: int = 0):
        self.bf = BackboneFeatures(backbone)
        self.coreset_ratio = coreset_ratio
        self.max_pool = max_pool
        self.size = size
        self.seed = seed
        self.grid = self.bf.patch_grid
        self.bank = None

    def fit(self, normal_paths):
        feats = _extract_all(self.bf, normal_paths, self.size)        # (N, L, D)
        pool = feats.reshape(-1, feats.shape[-1])                     # (N*L, D)
        # PatchCore도 근사다 — 너무 크면 무작위로 풀을 줄인 뒤 greedy coreset로 압축.
        rng = np.random.default_rng(self.seed)
        if len(pool) > self.max_pool:
            pool = pool[rng.choice(len(pool), self.max_pool, replace=False)]
        idx = greedy_coreset_indices(pool, self.coreset_ratio, seed=self.seed)
        self.bank = pool[idx].astype(np.float32)
        return self

    def score(self, paths, k: int = 1):
        """반환: image_scores (M,), pixel_maps (M, grid, grid)."""
        import time

        feats = _extract_all(self.bf, paths, self.size)              # (M, L, D)
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=k).fit(self.bank)
        img_scores, maps, lat = [], [], []
        for f in feats:
            t0 = time.perf_counter()
            dist, _ = nn.kneighbors(f)                                # (L, k)
            lat.append((time.perf_counter() - t0) * 1000)
            patch = dist.mean(axis=1)
            img_scores.append(float(patch.max()))
            maps.append(patch.reshape(self.grid, self.grid))
        return np.array(img_scores), np.stack(maps), np.array(lat)

    @property
    def bank_size(self) -> int:
        return 0 if self.bank is None else len(self.bank)

    def save_bank(self, path):
        """적합된 memory bank를 .npy로 저장(데모 콜드스타트 캐시용)."""
        np.save(path, self.bank)

    def load_bank(self, path):
        """저장된 memory bank를 불러와 fit을 건너뛴다."""
        self.bank = np.load(path).astype(np.float32)
        return self


class PaDiM:
    """패치 위치별 가우시안 + Mahalanobis. 무작위 차원축소로 공분산을 가볍게."""

    def __init__(self, backbone: str = "resnet18", n_dims: int = 100,
                 size: int = 224, seed: int = 0):
        self.bf = BackboneFeatures(backbone)
        self.n_dims = n_dims
        self.size = size
        self.seed = seed
        self.grid = self.bf.patch_grid
        self.dim_idx = None
        self.mean = None
        self.inv_cov = None

    def fit(self, normal_paths):
        feats = _extract_all(self.bf, normal_paths, self.size)        # (N, L, D)
        rng = np.random.default_rng(self.seed)
        d = feats.shape[-1]
        self.dim_idx = rng.choice(d, min(self.n_dims, d), replace=False)
        feats = feats[:, :, self.dim_idx].astype(np.float64)         # (N, L, d')
        self.mean, self.inv_cov = fit_padim_gaussian(feats)
        return self

    def score(self, paths):
        import time

        feats = _extract_all(self.bf, paths, self.size)[:, :, self.dim_idx].astype(np.float64)
        img_scores, maps, lat = [], [], []
        for f in feats:                                              # f: (L, d')
            t0 = time.perf_counter()
            patch, img = mahalanobis_scores(f, self.mean, self.inv_cov)
            lat.append((time.perf_counter() - t0) * 1000)
            img_scores.append(img)
            maps.append(patch.reshape(self.grid, self.grid))
        return np.array(img_scores), np.stack(maps), np.array(lat)

    @property
    def bank_size(self) -> int:
        # PaDiM '뱅크'=위치별 (mean+inv_cov) 파라미터 수 (비교표 메모리 칸용).
        if self.mean is None:
            return 0
        length, d = self.mean.shape
        return int(length * (d + d * d))


# ───────────────────────── smoke test (합성 feature) ─────────────────────────

def _smoke() -> int:
    """백본·데이터 없이 순수 로직만 자기점검."""
    rng = np.random.default_rng(0)
    # 정상: 원점 부근 클러스터. 이상: 멀리 떨어진 점.
    normal = rng.normal(0, 1, size=(200, 16)).astype(np.float32)
    bank_idx = greedy_coreset_indices(normal, ratio=0.25)
    bank = normal[bank_idx]
    assert len(bank) == int(np.ceil(200 * 0.25)), "coreset 크기 불일치"

    normal_test = rng.normal(0, 1, size=(10, 16)).astype(np.float32)
    anom_test = rng.normal(6, 1, size=(10, 16)).astype(np.float32)
    _, s_norm = knn_anomaly_scores(normal_test, bank)
    _, s_anom = knn_anomaly_scores(anom_test, bank)
    assert s_anom > s_norm, f"이상 점수가 정상보다 커야 함 ({s_anom} vs {s_norm})"

    # PaDiM 경로: 위치 1개로 단순화.
    feats = rng.normal(0, 1, size=(100, 1, 8)).astype(np.float32)
    mean, inv_cov = fit_padim_gaussian(feats)
    _, m_norm = mahalanobis_scores(rng.normal(0, 1, (1, 8)), mean, inv_cov)
    _, m_anom = mahalanobis_scores(np.full((1, 8), 6.0), mean, inv_cov)
    assert m_anom > m_norm, "Mahalanobis 이상>정상 위배"

    scores = np.concatenate([[s_norm] * 5, [s_anom] * 5])
    labels = np.array([0] * 5 + [1] * 5)
    assert abs(image_auroc(scores, labels) - 1.0) < 1e-9, "완전분리 AUROC 1.0 기대"
    print("smoke OK — coreset·kNN·PaDiM·AUROC 순수 로직 정상")
    return 0


def main():
    ap = argparse.ArgumentParser(description="무지도 이상탐지 (PatchCore/PaDiM)")
    ap.add_argument("--smoke", action="store_true", help="합성 feature로 순수 로직 자기점검")
    args = ap.parse_args()
    if args.smoke:
        raise SystemExit(_smoke())
    ap.print_help()


if __name__ == "__main__":
    main()
