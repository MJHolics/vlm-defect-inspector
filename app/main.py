import io
import json
import re
import time
import base64
from pathlib import Path

import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
)
from peft import PeftModel

# ── 설정 ──────────────────────────────────────────────
MODEL_ID   = "Qwen/Qwen2.5-VL-7B-Instruct"
LORA_PATH  = Path(__file__).parent.parent / "models" / "checkpoints" / "best"
USE_LORA   = LORA_PATH.exists()

DEFECT_CLASSES = [
    "crazing", "inclusion", "patches",
    "pitted_surface", "rolled-in_scale", "scratches"
]
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

# ── 모델 로드 (startup) ───────────────────────────────
app = FastAPI(
    title="VLM Defect Inspector",
    description="Qwen2.5-VL 7B (QLoRA) 기반 금속 표면 불량 분류 API",
    version="1.0.0",
)

model = None
processor = None


@app.on_event("startup")
def load_model():
    global model, processor
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    if USE_LORA:
        model = PeftModel.from_pretrained(base, str(LORA_PATH))
        processor = AutoProcessor.from_pretrained(str(LORA_PATH))
        print(f"QLoRA 어댑터 로드: {LORA_PATH}")
    else:
        model = base
        processor = AutoProcessor.from_pretrained(MODEL_ID)
        print("베이스 모델 로드 (LoRA 없음)")
    model.eval()


# ── 유틸 ──────────────────────────────────────────────
def parse_output(raw: str) -> dict | None:
    raw = re.sub(r"```json\s*", "", raw)
    raw = re.sub(r"```\s*", "", raw)
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        return None


def run_inference(img: Image.Image) -> tuple[dict | None, str, float]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text",  "text": INFERENCE_PROMPT},
            ],
        },
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[text], images=[img], return_tensors="pt", padding=True
    ).to(model.device)

    t0 = time.time()
    with torch.no_grad():
        out_ids = model.generate(
            **inputs, max_new_tokens=256, do_sample=False,
            temperature=None, top_p=None,
        )
    elapsed = time.time() - t0

    generated = out_ids[:, inputs["input_ids"].shape[1]:]
    raw = processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
    parsed = parse_output(raw)
    return parsed, raw, elapsed


# ── 엔드포인트 ────────────────────────────────────────
class InspectResponse(BaseModel):
    type: str | None
    type_ko: str | None
    severity: str | None
    description: str | None
    confidence: str
    elapsed_ms: float
    raw_output: str
    model: str


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": MODEL_ID,
        "lora": USE_LORA,
        "device": str(next(model.parameters()).device) if model else "not loaded",
    }


@app.post("/inspect", response_model=InspectResponse)
async def inspect(file: UploadFile = File(...)):
    """이미지 업로드 → 불량 유형 분류"""
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="이미지 파일만 업로드 가능합니다")

    contents = await file.read()
    try:
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="이미지 파일을 읽을 수 없습니다")

    parsed, raw, elapsed = run_inference(img)

    pred_type = None
    if parsed:
        pred_type = parsed.get("type", "").strip().lower()
        if pred_type not in DEFECT_CLASSES:
            pred_type = None

    return InspectResponse(
        type=pred_type,
        type_ko=parsed.get("type_ko") if parsed else None,
        severity=parsed.get("severity") if parsed else None,
        description=parsed.get("description") if parsed else None,
        confidence="high" if pred_type else "low",
        elapsed_ms=round(elapsed * 1000, 1),
        raw_output=raw,
        model=f"{MODEL_ID} {'+ QLoRA' if USE_LORA else '(zero-shot)'}",
    )


@app.post("/inspect/base64")
async def inspect_base64(payload: dict):
    """Base64 인코딩 이미지로 추론"""
    try:
        img_data = base64.b64decode(payload["image_b64"])
        img = Image.open(io.BytesIO(img_data)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="유효하지 않은 base64 이미지")

    parsed, raw, elapsed = run_inference(img)
    pred_type = None
    if parsed:
        pred_type = parsed.get("type", "").strip().lower()
        if pred_type not in DEFECT_CLASSES:
            pred_type = None

    return {
        "type": pred_type,
        "type_ko": parsed.get("type_ko") if parsed else None,
        "severity": parsed.get("severity") if parsed else None,
        "description": parsed.get("description") if parsed else None,
        "elapsed_ms": round(elapsed * 1000, 1),
        "raw_output": raw,
    }
