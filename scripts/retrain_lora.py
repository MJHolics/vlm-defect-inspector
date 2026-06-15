"""교정분을 합친 LoRA 재학습 (자가개선 루프 3단계, GPU 필요).

`notebooks/03_finetune.ipynb`의 학습 레시피를 스크립트로 옮겨, 감사 DB에서 나온
교정 라벨(retrain_manifest.jsonl)을 원본 train에 합쳐 LoRA를 재학습한다.

교정 타깃 복원:
  교정은 type만 확정하지만, 교정 샘플은 모두 풀(val.json) 이미지이므로 이미지
  해시로 val.json의 전체 정답(type/severity/description)을 되찾아 학습 타깃으로 쓴다.
  → test.json은 일절 건드리지 않는다(누수 방지).

사용 (GPU):
    python scripts/retrain_lora.py --out models/checkpoints/cand_v2 --epochs 3
    python scripts/retrain_lora.py --out models/checkpoints/cand_v2 --epochs 1 --max-extra 32
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
DATA = ROOT / "data" / "processed"

SYSTEM_PROMPT = (
    "당신은 금속 제품 표면 불량을 분석하는 전문 AI입니다. "
    "주어진 이미지를 분석하여 불량 유형을 정확히 판단하고 "
    "반드시 JSON 형식으로만 답변하세요."
)


def _img_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_records(manifest: Path) -> list[dict]:
    """train.json + (교정된 하드샘플의 val.json 원본 레코드). test는 제외."""
    train = json.loads((DATA / "train.json").read_text(encoding="utf-8"))
    val = json.loads((DATA / "val.json").read_text(encoding="utf-8"))

    # val 이미지 해시 → 전체 정답 레코드
    val_by_hash = {}
    for ex in val:
        p = ex["image"]
        ap = (ROOT / p) if not Path(p).is_absolute() else Path(p)
        if ap.exists():
            val_by_hash[_img_hash(ap)] = ex

    # 보관 이미지는 PNG로 재인코딩되므로 파일 내용 해시는 원본 jpg 해시와 다르다.
    # 파일명(<원본 sha256>.png)의 stem이 DB에 기록된 해시 = val 원본 해시와 일치한다.
    corrected_hashes = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        ip = Path(json.loads(line)["image_path"])
        if ip.exists():
            corrected_hashes.append(ip.stem)

    extra, missed = [], 0
    for h in corrected_hashes:
        if h in val_by_hash:
            extra.append(val_by_hash[h])
        else:
            missed += 1
    print(f"train {len(train)} + 교정 하드샘플 {len(extra)} "
          f"(매칭실패 {missed}) = {len(train)+len(extra)}건")
    return train + extra


class DefectVQADataset(Dataset):
    def __init__(self, records, processor, max_length=512):
        self.data = records
        self.processor = processor
        self.max_length = max_length
        self._asst_ids = torch.tensor(
            processor.tokenizer.encode("<|im_start|>assistant", add_special_tokens=False)
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        r = self.data[idx]
        p = Path(r["image"])
        ap = (ROOT / p) if not p.is_absolute() else p
        img = Image.open(ap).convert("RGB")
        user_text = r["conversations"][0]["content"]
        assistant_text = r["conversations"][1]["content"]
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": user_text}]},
            {"role": "assistant", "content": assistant_text},
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        inputs = self.processor(text=[text], images=[img], return_tensors="pt",
                                padding="max_length", max_length=self.max_length, truncation=True)
        input_ids = inputs["input_ids"].squeeze(0)
        attention_mask = inputs["attention_mask"].squeeze(0)
        pixel_values = inputs.get("pixel_values")
        if pixel_values is not None:
            pixel_values = pixel_values.squeeze(0)
        grid = inputs.get("image_grid_thw")
        if grid is not None:
            grid = grid.squeeze(0)
        labels = input_ids.clone()
        n = len(self._asst_ids)
        found = False
        for i in range(len(labels) - n):
            if (input_ids[i:i + n] == self._asst_ids).all():
                labels[:i + n] = -100
                found = True
                break
        if not found:
            labels[:] = -100
        return {"input_ids": input_ids, "attention_mask": attention_mask,
                "pixel_values": pixel_values, "image_grid_thw": grid, "labels": labels}


def collate_fn(batch):
    out = {}
    for k in batch[0]:
        vals = [b[k] for b in batch if b[k] is not None]
        if vals:
            out[k] = torch.stack(vals)
    return out


def main():
    ap = argparse.ArgumentParser(description="교정분 합친 LoRA 재학습")
    ap.add_argument("--out", type=Path, required=True, help="어댑터 저장 경로")
    ap.add_argument("--manifest", type=Path, default=DATA / "retrain_manifest.jsonl")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-extra", type=int, default=None,
                    help="교정 하드샘플 사용 상한(디버그용)")
    args = ap.parse_args()

    import bitsandbytes as bnb
    from transformers import (Qwen2_5_VLForConditionalGeneration, AutoProcessor,
                              BitsAndBytesConfig, get_cosine_schedule_with_warmup)
    from peft import (LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType)
    from tqdm import tqdm

    MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
    records = build_records(args.manifest)
    if args.max_extra is not None:
        base = len(json.loads((DATA / "train.json").read_text(encoding="utf-8")))
        records = records[:base + args.max_extra]

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID, quantization_config=bnb_config, device_map="auto", torch_dtype=torch.bfloat16)
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    lora_config = LoraConfig(
        r=16, lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05, bias="none", task_type=TaskType.CAUSAL_LM)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    ds = DefectVQADataset(records, processor)
    loader = DataLoader(ds, batch_size=1, shuffle=True, collate_fn=collate_fn, num_workers=0)
    optimizer = bnb.optim.AdamW8bit(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.01)
    total_steps = len(loader) * args.epochs // args.grad_accum
    scheduler = get_cosine_schedule_with_warmup(optimizer, int(total_steps * 0.1), total_steps)
    device = next(model.parameters()).device
    print(f"총 옵티마이저 스텝: {total_steps}  | 데이터 {len(ds)}건 × {args.epochs}epoch")

    def opt_step():
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        optimizer.step(); scheduler.step(); optimizer.zero_grad()

    model.train()
    for epoch in range(args.epochs):
        optimizer.zero_grad()
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        running = 0.0
        for step, batch in enumerate(pbar):
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            (out.loss / args.grad_accum).backward()
            running += out.loss.item()
            if (step + 1) % args.grad_accum == 0:
                opt_step()
            pbar.set_postfix(loss=f"{out.loss.item():.4f}")
        if len(loader) % args.grad_accum != 0:
            opt_step()
        print(f"Epoch {epoch+1}: avg_loss={running/len(loader):.4f}")

    args.out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.out)
    processor.save_pretrained(args.out)
    print(f"\n어댑터 저장: {args.out}")


if __name__ == "__main__":
    main()
