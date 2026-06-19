"""VLM Defect Inspector — 엣지 CNN 라이브 데모 (Hugging Face Spaces).

플래그십 Qwen2.5-VL 7B(QLoRA)는 정확도는 높지만 1장에 14.7초·6GB라 무료 CPU
Space엔 못 올린다. 그래서 같은 NEU 데이터·같은 test 분할(누수 0)로 증류·학습한
경량 CNN(MobileNetV3-Small, 1.52M 파라미터, 6MB)을 ONNX로 띄운다 — CPU 단일코어
수 ms로 도는, '라인에 실제로 올릴' 모델이다. 본 레포의 '엣지 배포·경량화' 트랙 결과물.

torch 없이 onnxruntime + numpy + pillow만으로 동작한다(가벼운 콜드스타트).
"""
import time
from pathlib import Path

import gradio as gr
import numpy as np
import onnxruntime as ort
from PIL import Image

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "model.onnx"
GITHUB_URL = "https://github.com/MJHolics/vlm-defect-inspector"

CLASSES = ["crazing", "inclusion", "patches",
           "pitted_surface", "rolled-in_scale", "scratches"]
CLASS_KO = {"crazing": "균열", "inclusion": "개재물", "patches": "패치결함",
            "pitted_surface": "피팅", "rolled-in_scale": "압연스케일", "scratches": "스크래치"}
# 본 프로젝트의 결함유형→심각도 고정 매핑 (acceptance 위험점수 산정과 동일 기준)
SEVERITY = {"crazing": "low", "inclusion": "medium", "patches": "low",
            "pitted_surface": "high", "rolled-in_scale": "medium", "scratches": "high"}
SEV_LABEL = {"low": "🟢 low (낮음)", "medium": "🟡 medium (보통)", "high": "🔴 high (높음)"}

# 운영 신뢰 임계값 — confidence가 이 아래면 사람 검토 큐로 (본 레포 Production Trust Layer)
REVIEW_THRESHOLD = 0.80
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_so = ort.SessionOptions()
_so.intra_op_num_threads = 1
_sess = ort.InferenceSession(str(MODEL_PATH), _so, providers=["CPUExecutionProvider"])
_input_name = _sess.get_inputs()[0].name


def _preprocess(img: Image.Image, size: int = 224) -> np.ndarray:
    img = img.convert("L").resize((size, size), Image.BILINEAR)  # grayscale
    x = np.asarray(img, dtype=np.float32) / 255.0
    x = np.stack([x, x, x], axis=0)                              # 3채널 복제
    x = (x - IMAGENET_MEAN[:, None, None]) / IMAGENET_STD[:, None, None]
    return x[None].astype(np.float32)                           # (1,3,H,W)


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def predict(image: Image.Image):
    if image is None:
        return "이미지를 업로드하거나 아래 예시를 눌러주세요.", None, ""

    t0 = time.perf_counter()
    logits = _sess.run(None, {_input_name: _preprocess(image)})[0][0]
    elapsed_ms = (time.perf_counter() - t0) * 1000

    probs = _softmax(logits)
    idx = int(probs.argmax())
    cls = CLASSES[idx]
    conf = float(probs[idx])
    sev = SEVERITY[cls]

    headline = f"## {cls}  ({CLASS_KO[cls]})  ·  {SEV_LABEL[sev]}"

    # 클래스별 확률 막대 (Gradio Label)
    conf_map = {f"{CLASSES[i]} ({CLASS_KO[CLASSES[i]]})": float(probs[i])
                for i in range(len(CLASSES))}

    gate = ("✅ 자동 통과 — confidence ≥ 0.80"
            if conf >= REVIEW_THRESHOLD else
            "⚠️ 사람 검토 필요 — confidence < 0.80 (운영 게이트가 검토 큐로 보냄)")
    note = (f"**신뢰도 {conf:.1%}**  ·  추론 {elapsed_ms:.1f} ms (CPU 단일코어)\n\n"
            f"{gate}\n\n"
            f"> 같은 이미지를 7B VLM으로 돌리면 약 14,700 ms. 이 엣지 CNN은 약 "
            f"{14681/elapsed_ms:,.0f}× 빠르고 GPU 없이 돈다.")

    return headline, conf_map, note


_DESC = f"""
# 🔍 Metal Surface Defect Inspector — 엣지 CNN 데모

NEU 금속 표면 결함 **6-class** 분류기. 업로드하거나 아래 예시를 누르면 즉시 판정합니다.

이 데모는 7B VLM이 아니라, 같은 데이터로 학습한 **경량 CNN(MobileNetV3-Small · 1.52M 파라미터 · 6 MB)**
을 ONNX로 띄운 것입니다. 폐쇄형 6-class에서 test 정확도 **99.6%**(7B VLM 95.9%보다 높음)를 내면서
**CPU 단일코어 수 ms**에 돌아 — 라인 인라인 검사에 실제로 올릴 수 있는 모델입니다.

VLM은 신규결함·콜드스타트·설명 트리아지에, 경량 CNN은 100% 인라인 분류에 — 역할 분담을 보여주는
포트폴리오의 *엣지 배포·경량화* 트랙 결과물입니다.  ·  [전체 코드·README →]({GITHUB_URL})
"""


with gr.Blocks(title="Metal Defect Inspector — Edge CNN") as demo:
    gr.Markdown(_DESC)
    with gr.Row():
        with gr.Column(scale=1):
            img_in = gr.Image(type="pil", label="금속 표면 이미지", height=300)
            btn = gr.Button("분석", variant="primary", size="lg")
            gr.Examples(
                examples=[[str(p)] for p in sorted((ROOT / "examples").glob("*.jpg"))],
                inputs=img_in, label="예시 (클릭)",
            )
        with gr.Column(scale=1):
            out_head = gr.Markdown()
            out_conf = gr.Label(label="클래스별 확률", num_top_classes=6)
            out_note = gr.Markdown()

    btn.click(predict, inputs=img_in, outputs=[out_head, out_conf, out_note])
    img_in.change(predict, inputs=img_in, outputs=[out_head, out_conf, out_note])

    gr.Markdown(
        "결함유형→심각도는 고정 매핑(crazing·patches=low, inclusion·rolled-in_scale=medium, "
        "pitted_surface·scratches=high). confidence < 0.80이면 운영 게이트가 사람 검토 큐로 보냅니다."
    )


if __name__ == "__main__":
    demo.launch()
