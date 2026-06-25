"""모델 선택 비교 데모 (로컬) — 같은 검사 문제, 다른 패러다임.

두 패러다임을 나란히 보여준다:
  ① 지도학습 분류 (NEU 금속표면) — 엣지 CNN(MobileNetV3-S, ONNX). 라벨로 6클래스를 직접 분류.
  ② 무지도 이상탐지 (MVTec screw) — PatchCore. 정상만 학습해 이탈을 히트맵으로 위치화.

도메인이 다르므로(NEU 표면 ≠ 나사) '한 이미지를 둘 다'는 정직하지 않다 — 각 패러다임을
제 데이터로 보이고, 아래 플레이북이 "어떤 상황에 어떤 모델을 왜"를 잇는다.

7B VLM은 안 띄운다(무겁다). 지도 패널은 라인에 실제로 올리는 경량 CNN(ONNX)을 쓴다.

사용:
    python demo_compare.py            # http://localhost:7860
    python demo_compare.py --share
"""
import argparse
import sys
import time
from pathlib import Path

import gradio as gr
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# ── 지도분류 (NEU 엣지 CNN, ONNX) ────────────────────────────────────────
import onnxruntime as ort  # noqa: E402

CNN_ONNX = ROOT / "space" / "model.onnx"
NEU_EXAMPLES = ROOT / "space" / "examples"
CLASSES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]
CLASS_KO = {"crazing": "균열", "inclusion": "개재물", "patches": "패치결함",
            "pitted_surface": "피팅", "rolled-in_scale": "압연스케일", "scratches": "스크래치"}
SEVERITY = {"crazing": "low", "inclusion": "medium", "patches": "low",
            "pitted_surface": "high", "rolled-in_scale": "medium", "scratches": "high"}
SEV_LABEL = {"low": "🟢 low", "medium": "🟡 medium", "high": "🔴 high"}
REVIEW_THRESHOLD = 0.80
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_cnn = None
if CNN_ONNX.exists():
    _so = ort.SessionOptions(); _so.intra_op_num_threads = 1
    _cnn = ort.InferenceSession(str(CNN_ONNX), _so, providers=["CPUExecutionProvider"])
    _cnn_in = _cnn.get_inputs()[0].name


def _prep_cnn(img: Image.Image, size: int = 224) -> np.ndarray:
    img = img.convert("L").resize((size, size), Image.BILINEAR)
    x = np.asarray(img, dtype=np.float32) / 255.0
    x = np.stack([x, x, x], axis=0)
    x = (x - _MEAN[:, None, None]) / _STD[:, None, None]
    return x[None].astype(np.float32)


def _softmax(z):
    z = z - z.max(); e = np.exp(z); return e / e.sum()


def predict_supervised(image):
    if image is None:
        return "이미지를 올리거나 예시를 눌러주세요.", None, ""
    if _cnn is None:
        return "엣지 CNN ONNX(space/model.onnx)가 없습니다.", None, ""
    t0 = time.perf_counter()
    logits = _cnn.run(None, {_cnn_in: _prep_cnn(image)})[0][0]
    ms = (time.perf_counter() - t0) * 1000
    probs = _softmax(logits)
    i = int(probs.argmax()); cls = CLASSES[i]; conf = float(probs[i])
    head = f"## {cls} ({CLASS_KO[cls]}) · {SEV_LABEL[SEVERITY[cls]]} {SEVERITY[cls]}"
    conf_map = {f"{CLASSES[j]} ({CLASS_KO[CLASSES[j]]})": float(probs[j]) for j in range(6)}
    gate = ("✅ 자동 통과 (confidence ≥ 0.80)" if conf >= REVIEW_THRESHOLD
            else "⚠️ 사람 검토 큐 (confidence < 0.80)")
    note = f"**신뢰도 {conf:.1%}** · 추론 {ms:.1f} ms (CPU 단일코어)\n\n{gate}"
    return head, conf_map, note


# ── 무지도 이상탐지 (MVTec screw, PatchCore) ──────────────────────────────
_ad = None           # (detector, threshold, score_min, score_max)
_AD_ERR = ""


AD_CACHE = ROOT / "data" / "results" / "_ad_demo_screw.npz"


def _build_ad():
    """screw 정상으로 PatchCore(heavy)를 적합. 뱅크·임계값을 디스크 캐시해 콜드스타트 단축."""
    global _ad, _AD_ERR
    if _ad is not None or _AD_ERR:
        return
    try:
        from scripts import anomaly_detect as ad
        from scripts.eval_anomaly import load_mvtec
        from sklearn.metrics import roc_curve

        det = ad.PatchCore(backbone="wide_resnet50_2", coreset_ratio=0.10)
        if AD_CACHE.exists():
            c = np.load(AD_CACHE)
            det.bank = c["bank"].astype(np.float32)
            thr, lo, hi = float(c["thr"]), float(c["lo"]), float(c["hi"])
        else:
            train, items = load_mvtec("screw")
            det.fit([str(p) for p in train])
            scores, _, _ = det.score([str(it["path"]) for it in items])
            labels = np.array([it["label"] for it in items])
            # Youden's J 최적 임계값(정상/이상 분리 최대 지점).
            fpr, tpr, ths = roc_curve(labels, scores)
            thr = float(ths[np.argmax(tpr - fpr)])
            lo, hi = float(scores.min()), float(scores.max())
            AD_CACHE.parent.mkdir(parents=True, exist_ok=True)
            np.savez(AD_CACHE, bank=det.bank, thr=thr, lo=lo, hi=hi)
        _ad = (det, thr, lo, hi)
    except Exception as e:  # 데이터 미존재 등
        _AD_ERR = f"AD 초기화 실패: {e}\nscripts/fetch_mvtec.py --category screw 를 먼저 실행하세요."


def _overlay(image: Image.Image, amap: np.ndarray, size: int = 224) -> Image.Image:
    """이상맵(grid)을 원본 위에 jet 컬러로 반투명 오버레이한 PIL 이미지."""
    import matplotlib.cm as cm
    base = image.convert("L").resize((size, size)).convert("RGB")
    up = np.asarray(Image.fromarray(amap).resize((size, size), Image.BILINEAR), dtype=np.float32)
    up = (up - up.min()) / (np.ptp(up) + 1e-8)
    heat = (cm.jet(up)[:, :, :3] * 255).astype(np.uint8)
    blend = (0.55 * np.asarray(base) + 0.45 * heat).astype(np.uint8)
    return Image.fromarray(blend)


def predict_anomaly(image):
    if image is None:
        return None, "screw 이미지를 올리거나 예시를 눌러주세요."
    _build_ad()
    if _AD_ERR:
        return None, _AD_ERR
    det, thr, lo, hi = _ad
    tmp = ROOT / "data" / "results" / "_demo_upload.png"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(tmp)
    t0 = time.perf_counter()
    scores, maps, _ = det.score([str(tmp)])
    ms = (time.perf_counter() - t0) * 1000
    score = float(scores[0])
    norm = (score - lo) / (hi - lo + 1e-8)
    verdict = "🔴 이상(anomaly) 의심" if score >= thr else "🟢 정상에 가까움"
    note = (f"## {verdict}\n\n**이상 점수 {score:.3f}** (정규화 {norm:.0%}, 임계 {thr:.3f}) · "
            f"추론 {ms:.1f} ms\n\n정상 320장만 학습한 PatchCore가 *처음 보는* 이 이미지의 "
            f"정상 분포 이탈을 점수화했습니다. 히트맵=이탈이 큰 영역(결함 후보).")
    return _overlay(image, maps[0]), note


# ── 플레이북 ──────────────────────────────────────────────────────────────
_PLAYBOOK = """
### 🧭 모델 선택 플레이북 — 어떤 상황에 어떤 모델을 왜

| 상황 | 권장 모델 | 왜 |
|------|-----------|----|
| 폐쇄셋·라벨 충분·고물량 인라인 | **경량 CNN** (왼쪽 탭) | CPU 1.8 ms·99.6%·6 MB — 7B VLM보다 빠르고 정확 |
| 결함 라벨이 없거나 극소수(정상만 풍부) | **무지도 AD** (오른쪽 탭) | 정상만 학습해 결함 위치화 (pixel AUROC 0.98) |
| 자연어 근거·설명 필요 | 7B VLM (QLoRA) | 구조화 리포트·설명 (이 데모는 경량만 띄움) |
| 신규 결함·낯선 입력 차단 | OOD + confidence 게이트 | 분포 밖이면 사람 검토로 |

> 두 패널은 **도메인이 다르다**(NEU 표면 ≠ MVTec 나사) — 그래서 같은 이미지를 둘 다 돌리지
> 않고 각 패러다임을 제 데이터로 보인다. 핵심은 "정답 모델은 하나가 아니라 *데이터 상황·자원·
> 설명 요구*가 고른다"는 것. 전체 비교 수치·코드는 README의 '모델 선택 플레이북' 참고.
"""


def _examples(folder: Path, pattern: str):
    return [[str(p)] for p in sorted(folder.glob(pattern))[:8]] if folder.exists() else []


with gr.Blocks(title="모델 선택 비교 — 지도분류 vs 무지도 이상탐지",
               theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🔬 모델 선택 비교 데모 — 같은 검사 문제, 다른 패러다임")
    gr.Markdown(_PLAYBOOK)

    with gr.Tab("① 지도분류 (NEU 금속표면 · 엣지 CNN)"):
        gr.Markdown("라벨로 학습한 6-class 분류기(MobileNetV3-S, ONNX, CPU). "
                    "결함 *유형*을 직접 분류 — 라벨이 충분한 폐쇄셋 고물량 라인용.")
        with gr.Row():
            with gr.Column():
                s_in = gr.Image(type="pil", label="NEU 금속 표면 이미지", height=300)
                s_btn = gr.Button("분류", variant="primary")
                gr.Examples(_examples(NEU_EXAMPLES, "*.jpg"), inputs=s_in, label="NEU 예시")
            with gr.Column():
                s_head = gr.Markdown()
                s_conf = gr.Label(label="클래스별 확률", num_top_classes=6)
                s_note = gr.Markdown()
        s_btn.click(predict_supervised, s_in, [s_head, s_conf, s_note])
        s_in.change(predict_supervised, s_in, [s_head, s_conf, s_note])

    with gr.Tab("② 무지도 이상탐지 (MVTec screw · PatchCore)"):
        gr.Markdown("**정상(양품) 320장만** 학습한 PatchCore. 결함 라벨 0개로 *이탈*을 탐지·위치화 — "
                    "결함 샘플이 귀한 라인용. 히트맵이 결함 후보 영역을 가리킨다.")
        with gr.Row():
            with gr.Column():
                a_in = gr.Image(type="pil", label="나사(screw) 이미지", height=300)
                a_btn = gr.Button("이상 탐지", variant="primary")
                gr.Examples(
                    _examples(ROOT / "data/mvtec/screw/screw/test/scratch_neck", "*.png")
                    + _examples(ROOT / "data/mvtec/screw/screw/test/good", "*.png"),
                    inputs=a_in, label="screw 예시 (결함/정상)")
            with gr.Column():
                a_out = gr.Image(label="이상 히트맵 오버레이", height=300)
                a_note = gr.Markdown()
        a_btn.click(predict_anomaly, a_in, [a_out, a_note])

    gr.Markdown("---\n*로컬 비교 데모. 라이브 CNN 데모: "
                "huggingface.co/spaces/appleholics/metal-defect-inspector*")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--share", action="store_true")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()
    demo.launch(share=args.share, server_port=args.port)
