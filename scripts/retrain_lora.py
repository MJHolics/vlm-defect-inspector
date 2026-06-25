"""교정분을 합친 LoRA 재학습 (자가개선 루프 3단계, GPU 필요).

`notebooks/03_finetune.ipynb`의 학습 레시피를 스크립트로 옮겨, 감사 DB에서 나온
교정 라벨(retrain_manifest.jsonl)을 원본 train에 합쳐 LoRA를 재학습한다.

교정 타깃 복원:
  교정은 type만 확정하지만, 교정 샘플은 모두 풀(val.json) 이미지이므로 이미지
  해시로 val.json의 전체 정답(type/severity/description)을 되찾아 학습 타깃으로 쓴다.
  → test.json은 일절 건드리지 않는다(누수 방지).

레시피(현행 best_exp와 정합):
  rank=32 / lora_alpha=64 / label_smoothing=0.1 / 증강(flip·rot90·밝기대비·블러) /
  val early-stopping. early-stopping의 검증셋은 val.json에서 교정분(train에 흡수)을
  뺀 나머지를 쓴다(test 불사용). 검증 loss가 가장 낮은 체크포인트만 저장한다.
  rank16·스무딩X·증강X·고정에폭이던 구버전은 1epoch=저적합 / 2~3epoch=과적합으로
  정상 구간이 없었다(v2·v3 거부). 이 레시피는 그 진단에 대한 수정이다.

사용 (GPU):
    python scripts/retrain_lora.py --out models/checkpoints/cand_v4
    python scripts/retrain_lora.py --out models/checkpoints/cand_v4 --epochs 5 --patience 2
"""
import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# 윈도우 콘솔(cp949)은 일부 유니코드(—, ↳ 등)를 인코딩하지 못해 print에서
# UnicodeEncodeError로 죽는다. 출력 스트림을 UTF-8로 고정해 어디서든 안전하게 찍는다.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

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


def build_splits(manifest: Path) -> tuple[list[dict], list[dict]]:
    """(train, es_val) 반환.

    train    = train.json + 교정된 하드샘플의 val.json 원본 레코드
    es_val   = val.json 중 교정에 쓰이지 않은 나머지 (early-stopping 검증용)
    test.json은 어느 쪽에도 들어가지 않는다.
    """
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
    corrected_hashes = set()
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        ip = Path(json.loads(line)["image_path"])
        if ip.exists():
            corrected_hashes.add(ip.stem)

    extra = [val_by_hash[h] for h in corrected_hashes if h in val_by_hash]
    missed = len(corrected_hashes) - len(extra)
    es_val = [ex for h, ex in val_by_hash.items() if h not in corrected_hashes]
    print(f"train {len(train)} + 교정 하드샘플 {len(extra)} (매칭실패 {missed}) "
          f"= {len(train)+len(extra)}건  | early-stop 검증셋 {len(es_val)}건")
    return train + extra, es_val


def _rec_type(rec: dict) -> str:
    """레코드(conversations)에서 정답 결함유형 추출 (없으면 '')."""
    for t in rec.get("conversations", []):
        if t.get("role") == "assistant":
            try:
                return (json.loads(t["content"]).get("type") or "").lower()
            except Exception:
                return ""
    return ""


class _Rotate90:
    """0/90/180/270도 랜덤 회전 (정사각 NEU 이미지에 라벨 보존)."""

    def __call__(self, img):
        k = random.randint(0, 3)
        return img.rotate(90 * k) if k else img


def build_augment():
    """현행 best_exp 증강(albumentations)과 동등한 torchvision 파이프라인.

    이 환경(.venv)엔 albumentations가 없어 의존성을 늘리지 않고 torchvision으로
    동등 구성: rot90 · h/v flip · 밝기/대비 · 가우시안 블러.
    """
    from torchvision.transforms import v2 as T

    return T.Compose([
        _Rotate90(),
        T.RandomHorizontalFlip(0.5),
        T.RandomVerticalFlip(0.5),
        T.ColorJitter(brightness=0.2, contrast=0.2),
        T.RandomApply([T.GaussianBlur(kernel_size=3)], p=0.2),
    ])


class DefectVQADataset(Dataset):
    def __init__(self, records, processor, max_length=512, augment=False):
        self.data = records
        self.processor = processor
        self.max_length = max_length
        self.aug = build_augment() if augment else None
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
        if self.aug is not None:
            img = self.aug(img)
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


def _lm_loss(logits, labels, label_smoothing):
    """다음 토큰 예측 CE (라벨 스무딩 포함). 모델 내부 CE 대신 직접 계산해 스무딩 적용."""
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
        label_smoothing=label_smoothing,
    )


@torch.no_grad()
def eval_val_loss(model, loader, device, label_smoothing):
    model.eval()
    total, n = 0.0, 0
    for batch in loader:
        labels = batch.pop("labels").to(device)
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)
        total += _lm_loss(out.logits, labels, label_smoothing).item()
        n += 1
    return total / max(n, 1)


def main():
    ap = argparse.ArgumentParser(description="교정분 합친 LoRA 재학습 (검증된 레시피 + early-stopping)")
    ap.add_argument("--out", type=Path, required=True, help="어댑터 저장 경로(best 체크포인트)")
    ap.add_argument("--manifest", type=Path, default=DATA / "retrain_manifest.jsonl")
    ap.add_argument("--epochs", type=int, default=5, help="최대 epoch (early-stopping이 조기 종료)")
    ap.add_argument("--patience", type=int, default=2, help="val loss 미개선 허용 epoch 수")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--lora-alpha", type=int, default=64)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--no-augment", dest="augment", action="store_false",
                    help="데이터 증강 끄기")
    ap.set_defaults(augment=True)
    ap.add_argument("--val-limit", type=int, default=80,
                    help="early-stop 검증셋 표본 상한(속도용)")
    ap.add_argument("--max-extra", type=int, default=None,
                    help="교정 하드샘플 사용 상한(디버그용)")
    ap.add_argument("--limit-train", type=int, default=None,
                    help="원본 train 사용 상한(스모크 테스트용)")
    ap.add_argument("--holdout-class", default=None,
                    help="이 결함유형을 train/val에서 제외(open-set OOD 실험용). "
                         "제외된 클래스는 학습에 한 번도 안 쓰여 '신규 결함'이 된다.")
    ap.add_argument("--oversample-class", default=None,
                    help="이 결함유형(들)을 학습셋에서 N배로 오버샘플(진단된 약한 클래스 보강). "
                         "쉼표로 여러 클래스 지정 가능(예: inclusion,rolled-in_scale) — 각 클래스에 "
                         "같은 배수 적용. 증강과 결합되어 매 복제본이 다른 뷰로 들어가 경계를 강화한다.")
    ap.add_argument("--oversample-factor", type=int, default=1,
                    help="--oversample-class 총 등장 배수(1=변화 없음, 3=원본 포함 3배)")
    args = ap.parse_args()

    random.seed(42)
    torch.manual_seed(42)

    import bitsandbytes as bnb
    from transformers import (Qwen2_5_VLForConditionalGeneration, AutoProcessor,
                              BitsAndBytesConfig, get_cosine_schedule_with_warmup)
    from peft import (LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType)
    from tqdm import tqdm

    MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
    train_records, es_val = build_splits(args.manifest)
    if args.holdout_class:
        hc = args.holdout_class.lower()
        n0t, n0v = len(train_records), len(es_val)
        train_records = [r for r in train_records if _rec_type(r) != hc]
        es_val = [r for r in es_val if _rec_type(r) != hc]
        print(f"[holdout] '{hc}' 제외: train {n0t}→{len(train_records)}, "
              f"val {n0v}→{len(es_val)} (이 클래스는 학습에 안 쓰임 = 신규 결함)")
    base = len([r for r in json.loads((DATA / "train.json").read_text(encoding="utf-8"))
                if not args.holdout_class or _rec_type(r) != args.holdout_class.lower()])
    train_part, extra_part = train_records[:base], train_records[base:]
    if args.limit_train is not None:
        train_part = train_part[: args.limit_train]
    if args.max_extra is not None:
        extra_part = extra_part[: args.max_extra]
    train_records = train_part + extra_part
    oversampled = 0
    if args.oversample_class and args.oversample_factor > 1:
        classes = [c.strip().lower() for c in args.oversample_class.split(",") if c.strip()]
        all_copies = []
        for oc in classes:
            base_recs = [r for r in train_records if _rec_type(r) == oc]
            extra_copies = base_recs * (args.oversample_factor - 1)
            all_copies += extra_copies
            print(f"[oversample] '{oc}' {len(base_recs)}건 × {args.oversample_factor} "
                  f"→ +{len(extra_copies)}건 (증강이 매 복제본을 다른 뷰로 만듦)")
        train_records = train_records + all_copies
        oversampled = len(all_copies)
    if args.val_limit is not None:
        es_val = es_val[: args.val_limit]
    print(f"사용: train {len(train_part)} + 교정 {len(extra_part)} + 오버샘플 {oversampled} "
          f"= {len(train_records)}건 | 검증 {len(es_val)}건 | rank={args.rank} "
          f"smoothing={args.label_smoothing} augment={args.augment}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID, quantization_config=bnb_config, device_map="auto", torch_dtype=torch.bfloat16)
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    lora_config = LoraConfig(
        r=args.rank, lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05, bias="none", task_type=TaskType.CAUSAL_LM)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_ds = DefectVQADataset(train_records, processor, augment=args.augment)
    val_ds = DefectVQADataset(es_val, processor, augment=False)
    loader = DataLoader(train_ds, batch_size=1, shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=collate_fn, num_workers=0)
    optimizer = bnb.optim.AdamW8bit(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.01)
    total_steps = len(loader) * args.epochs // args.grad_accum
    scheduler = get_cosine_schedule_with_warmup(optimizer, int(total_steps * 0.1), total_steps)
    device = next(model.parameters()).device
    print(f"총 옵티마이저 스텝(최대): {total_steps}  | 데이터 {len(train_ds)}건 × 최대 {args.epochs}epoch")

    def opt_step():
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        optimizer.step(); scheduler.step(); optimizer.zero_grad()

    best_val = float("inf")
    bad_epochs = 0
    args.out.mkdir(parents=True, exist_ok=True)
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        running = 0.0
        for step, batch in enumerate(pbar):
            labels = batch.pop("labels").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            loss = _lm_loss(out.logits, labels, args.label_smoothing)
            (loss / args.grad_accum).backward()
            running += loss.item()
            if (step + 1) % args.grad_accum == 0:
                opt_step()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        if len(loader) % args.grad_accum != 0:
            opt_step()
        train_loss = running / len(loader)
        val_loss = eval_val_loss(model, val_loader, device, args.label_smoothing)
        print(f"Epoch {epoch+1}: train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

        if val_loss < best_val - 1e-4:
            best_val = val_loss
            bad_epochs = 0
            model.save_pretrained(args.out)
            processor.save_pretrained(args.out)
            print(f"  ↳ val 개선 → best 체크포인트 저장 (best_val={best_val:.4f})")
        else:
            bad_epochs += 1
            print(f"  ↳ val 미개선 ({bad_epochs}/{args.patience})")
            if bad_epochs >= args.patience:
                print(f"조기 종료(early stopping) — best_val={best_val:.4f}")
                break

    print(f"\n최종 best 어댑터: {args.out}  (best_val_loss={best_val:.4f})")


if __name__ == "__main__":
    main()
