import io
import json
import math
import re
import time
import base64
from pathlib import Path

import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
from pydantic import BaseModel
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
)
from peft import PeftModel

from . import audit, config, registry

# ── 설정 ──────────────────────────────────────────────
MODEL_ID   = "Qwen/Qwen2.5-VL-7B-Instruct"
LORA_PATH  = Path(__file__).parent.parent / "models" / "checkpoints" / "best"
USE_LORA   = LORA_PATH.exists()
MODEL_VERSION = f"{MODEL_ID}{'+QLoRA(best)' if USE_LORA else '(zero-shot)'}"

DEFECT_CLASSES = config.DEFECT_CLASSES
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
    description="Qwen2.5-VL 7B (QLoRA) 기반 금속 표면 불량 분류 API — 감사 추적·검토 큐·드리프트 모니터링 포함",
    version="2.0.0",
)

model = None
processor = None


@app.on_event("startup")
def startup():
    audit.init_db()
    load_model()


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


def _compute_confidence(scores: tuple, sequences: torch.Tensor, prompt_len: int) -> float:
    """생성된 토큰들의 로그확률 기하평균 → 0–1 confidence score.

    scores : model.generate(..., output_scores=True) 반환값 (토큰별 logit tuple)
    높을수록 모델이 자신의 출력에 확신하고 있음을 의미한다.
    """
    if not scores:
        return 0.0
    gen_tokens = sequences[0, prompt_len:]
    log_probs = [
        F.log_softmax(s[0], dim=-1)[t].item()
        for s, t in zip(scores, gen_tokens)
    ]
    return math.exp(sum(log_probs) / len(log_probs))


def run_inference(img: Image.Image) -> tuple[dict | None, str, float, float]:
    """이미지 추론 → (parsed, raw_text, elapsed_sec, confidence 0–1)"""
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

    prompt_len = inputs["input_ids"].shape[1]
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=256, do_sample=False,
            temperature=None, top_p=None,
            output_scores=True, return_dict_in_generate=True,
        )
    elapsed = time.time() - t0

    confidence = _compute_confidence(out.scores, out.sequences, prompt_len)
    generated = out.sequences[:, prompt_len:]
    raw = processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
    parsed = parse_output(raw)
    return parsed, raw, elapsed, confidence


def _normalize_type(parsed: dict | None) -> str | None:
    if not parsed:
        return None
    t = (parsed.get("type") or "").strip().lower()
    return t if t in DEFECT_CLASSES else None


def inspect_and_log(raw_bytes: bytes, img: Image.Image) -> dict:
    """추론 → confidence 기반 검토판정 → 감사 기록 → 응답 dict."""
    parsed, raw, elapsed, confidence = run_inference(img)
    pred_type = _normalize_type(parsed)
    confidence = round(confidence, 4)
    elapsed_ms = round(elapsed * 1000, 1)

    review_status = audit.decide_review(confidence, pred_type)
    image_sha256 = audit.image_hash(raw_bytes)
    # 검토가 필요한 건은 이미지를 보관해 두어 재라벨링·재학습에 쓴다.
    if review_status == "needs_review":
        audit.save_review_image(image_sha256, img)
    record_id = audit.log_inference(
        image_sha256=image_sha256,
        predicted_type=pred_type,
        severity=parsed.get("severity") if parsed else None,
        confidence=confidence,
        latency_ms=elapsed_ms,
        model_version=MODEL_VERSION,
        review_status=review_status,
        raw_output=raw,
    )

    return {
        "record_id": record_id,
        "type": pred_type,
        "type_ko": parsed.get("type_ko") if parsed else None,
        "severity": parsed.get("severity") if parsed else None,
        "description": parsed.get("description") if parsed else None,
        "confidence": confidence,
        "review_status": review_status,
        "review_required": review_status == "needs_review",
        "elapsed_ms": elapsed_ms,
        "raw_output": raw,
        "model": MODEL_VERSION,
    }


# ── 응답/요청 모델 ────────────────────────────────────
class InspectResponse(BaseModel):
    record_id: int
    type: str | None
    type_ko: str | None
    severity: str | None
    description: str | None
    confidence: float            # 0.0–1.0 (생성 토큰 로그확률 기하평균)
    review_status: str           # auto_accepted | needs_review
    review_required: bool        # confidence가 임계값 미만이면 True
    elapsed_ms: float
    raw_output: str
    model: str


class CorrectionRequest(BaseModel):
    true_type: str
    reviewer: str


# ── 추론 엔드포인트 ───────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": MODEL_VERSION,
        "lora": USE_LORA,
        "device": str(next(model.parameters()).device) if model else "not loaded",
        "confidence_threshold": config.CONFIDENCE_THRESHOLD,
    }


@app.post("/inspect", response_model=InspectResponse)
async def inspect(file: UploadFile = File(...)):
    """이미지 업로드 → 불량 유형 분류 (+ 감사 기록·검토 판정)"""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="이미지 파일만 업로드 가능합니다")

    contents = await file.read()
    try:
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="이미지 파일을 읽을 수 없습니다")

    return inspect_and_log(contents, img)


@app.post("/inspect/base64")
async def inspect_base64(payload: dict):
    """Base64 인코딩 이미지로 추론"""
    try:
        img_data = base64.b64decode(payload["image_b64"])
        img = Image.open(io.BytesIO(img_data)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="유효하지 않은 base64 이미지")

    return inspect_and_log(img_data, img)


# ── 검토(human-in-the-loop) 엔드포인트 ────────────────
@app.get("/review/queue")
def review_queue(limit: int = 50):
    """확신도가 낮아 사람 검토가 필요한 건 목록."""
    items = audit.list_review_queue(limit=limit)
    return {"count": len(items), "items": items}


@app.post("/review/{record_id}/correct")
def review_correct(record_id: int, body: CorrectionRequest):
    """검토자가 정답을 확정 → 감사 기록 갱신."""
    true_type = body.true_type.strip().lower()
    if true_type not in DEFECT_CLASSES:
        raise HTTPException(
            status_code=400,
            detail=f"true_type은 다음 중 하나여야 합니다: {DEFECT_CLASSES}",
        )
    updated = audit.submit_correction(record_id, true_type, body.reviewer)
    if updated is None:
        raise HTTPException(status_code=404, detail="해당 record를 찾을 수 없습니다")
    return updated


# ── 모니터링 엔드포인트 ───────────────────────────────
@app.get("/monitor/stats")
def monitor_stats():
    """운영 현황: 총 건수, 상태별 분포, 평균 confidence, 검토건 실측 정확도."""
    return audit.stats()


@app.get("/monitor/drift")
def monitor_drift(window: int | None = None):
    """입력/출력 분포 드리프트 진단 (PSI + 평균 confidence 하락)."""
    return audit.drift_report(window=window)


# ── 모델 레지스트리 엔드포인트 ────────────────────────
@app.get("/registry")
def registry_state():
    """모델 이력: 현재 운영 버전 + 등록된 모든 버전(성능·승격 사유·데이터 ref)."""
    return registry.load()


@app.post("/registry/rollback")
def registry_rollback():
    """직전 보관 모델로 롤백 (D5)."""
    result = registry.rollback()
    if not result.get("ok"):
        raise HTTPException(status_code=409, detail=result.get("detail"))
    return result
