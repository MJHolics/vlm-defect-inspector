"""
VLM Defect Inspector — Gradio 데모

Usage:
    python demo.py               # 로컬 실행 (http://localhost:7860)
    python demo.py --share       # 공개 링크 생성 (포트폴리오 시연용)
"""
import argparse
import json
import math
import re
import time
from pathlib import Path

import gradio as gr
import torch
import torch.nn.functional as F
from PIL import Image
from peft import PeftModel
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2_5_VLForConditionalGeneration,
)

# ── 설정 ──────────────────────────────────────────────────────────────
ROOT       = Path(__file__).parent
LORA_PATH  = ROOT / "models" / "checkpoints" / "best"
MODEL_ID   = "Qwen/Qwen2.5-VL-7B-Instruct"

DEFECT_CLASSES = [
    "crazing", "inclusion", "patches",
    "pitted_surface", "rolled-in_scale", "scratches",
]
CLASS_KO = {
    "crazing": "균열", "inclusion": "개재물", "patches": "패치결함",
    "pitted_surface": "피팅", "rolled-in_scale": "압연스케일", "scratches": "스크래치",
}
SEVERITY_LABEL = {"low": "🟢 low (낮음)", "medium": "🟡 medium (보통)", "high": "🔴 high (높음)"}

SYSTEM_PROMPT = (
    "당신은 금속 제품 표면 불량을 분석하는 전문 AI입니다. "
    "주어진 이미지를 분석하여 불량 유형을 정확히 판단하고 "
    "반드시 JSON 형식으로만 답변하세요."
)
INFERENCE_PROMPT = (
    "이 금속 표면 이미지를 분석하고 불량 정보를 JSON 형식으로 출력해줘.\n"
    "불량 유형은 반드시 다음 중 하나여야 해: "
    "crazing, inclusion, patches, pitted_surface, rolled-in_scale, scratches\n"
    '출력 형식: {"type": "...", "type_ko": "...", "severity": "low|medium|high", "description": "..."}'
)

# ── 모델 로드 ──────────────────────────────────────────────────────────
print("모델 로드 중 (최초 실행 시 수 분 소요)...")
_bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)
_base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID, quantization_config=_bnb, device_map="auto", torch_dtype=torch.float16,
)
if LORA_PATH.exists():
    model     = PeftModel.from_pretrained(_base, str(LORA_PATH))
    processor = AutoProcessor.from_pretrained(str(LORA_PATH))
    MODEL_LABEL = "Qwen2.5-VL 7B + QLoRA (fine-tuned)"
else:
    model     = _base
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    MODEL_LABEL = "Qwen2.5-VL 7B (zero-shot)"
model.eval()
print(f"로드 완료: {MODEL_LABEL}")


# ── 추론 유틸 ──────────────────────────────────────────────────────────
def _parse(raw: str) -> dict | None:
    raw = re.sub(r"```json\s*", "", raw)
    raw = re.sub(r"```\s*", "", raw)
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group())
    except json.JSONDecodeError:
        return None


def _confidence(scores: tuple, sequences: torch.Tensor, prompt_len: int) -> float:
    """생성 토큰 로그확률 기하평균 → 0–1 confidence score"""
    if not scores:
        return 0.0
    gen_tokens = sequences[0, prompt_len:]
    log_probs = [
        F.log_softmax(s[0], dim=-1)[t].item()
        for s, t in zip(scores, gen_tokens)
    ]
    return math.exp(sum(log_probs) / len(log_probs))


def predict(image: Image.Image):
    if image is None:
        return "—", "—", 0.0, "이미지를 업로드해주세요.", 0.0

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text",  "text": INFERENCE_PROMPT},
        ]},
    ]
    text   = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True).to(model.device)

    prompt_len = inputs["input_ids"].shape[1]
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=256, do_sample=False,
            output_scores=True, return_dict_in_generate=True,
        )
    elapsed_ms = (time.time() - t0) * 1000

    conf = _confidence(out.scores, out.sequences, prompt_len)
    raw  = processor.batch_decode(out.sequences[:, prompt_len:], skip_special_tokens=True)[0].strip()
    data = _parse(raw)

    if data:
        pred = data.get("type", "").strip().lower()
        if pred not in DEFECT_CLASSES:
            pred = "unknown"
        ko       = CLASS_KO.get(pred, "-")
        severity = SEVERITY_LABEL.get(data.get("severity", ""), data.get("severity", "-"))
        type_str = f"{pred}  ({ko})"
        desc     = data.get("description", "")
    else:
        type_str = "파싱 실패"
        severity = "-"
        desc     = f"[raw]\n{raw[:300]}"

    return type_str, severity, round(conf, 4), desc, round(elapsed_ms, 1)


# ── UI ────────────────────────────────────────────────────────────────
_css = """
.result-box { font-size: 1.1rem; font-weight: 600; }
"""

with gr.Blocks(title="VLM Defect Inspector", theme=gr.themes.Soft(), css=_css) as demo:
    gr.Markdown(f"""
# 🔍 VLM Defect Inspector
**{MODEL_LABEL}** 기반 금속 표면 불량 분류 시스템

NEU Metal Surface Defects 6-class:
`crazing` · `inclusion` · `patches` · `pitted_surface` · `rolled-in_scale` · `scratches`
""")

    with gr.Row():
        with gr.Column(scale=1):
            img_in  = gr.Image(type="pil", label="금속 표면 이미지 업로드", height=300)
            btn     = gr.Button("분석 시작", variant="primary", size="lg")

        with gr.Column(scale=1):
            out_type = gr.Textbox(label="불량 유형", elem_classes="result-box")
            out_sev  = gr.Textbox(label="심각도")
            out_conf = gr.Number(label="신뢰도 (0–1)  ·  로그확률 기하평균")
            out_desc = gr.Textbox(label="설명", lines=4)
            out_time = gr.Number(label="추론 시간 (ms)")

    btn.click(
        predict,
        inputs=[img_in],
        outputs=[out_type, out_sev, out_conf, out_desc, out_time],
    )

    gr.Markdown("""
---
**신뢰도 해석 가이드**
| 범위 | 의미 |
|------|------|
| 0.8 이상 | 모델이 매우 확신하는 예측 |
| 0.5 – 0.8 | 보통 수준의 확신 |
| 0.5 미만 | 낮은 확신 — 결과를 재검토 권장 |
""")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true", help="공개 링크 생성 (ngrok)")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    demo.launch(share=args.share, server_port=args.port)
