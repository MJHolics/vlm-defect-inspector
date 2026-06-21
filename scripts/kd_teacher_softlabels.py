"""VLM teacher → 엣지 CNN 지식증류용 soft label 생성 (GPU 필요).

플래그십 Qwen2.5-VL(v4, 7B QLoRA)을 **teacher**로 써서 train 이미지마다
6클래스에 대한 soft 확률분포를 만든다. 생성된 student CNN 학습(train_edge_kd.py)이
이 분포를 증류 타깃으로 쓴다.

핵심 — 생성(generate)이 아니라 **제약된 클래스명 스코어링**:
  추론 프롬프트 뒤에 강제 어시스턴트 접두사 `{"type": "<클래스>"` 를 붙여
  teacher-forcing 으로 흘려보내고, 그 <클래스> 토큰들의 **토큰당 평균 logprob**을
  6개 클래스에 대해 구한 뒤 softmax → soft 분포.
  · 단발 forward라 generate(14.7s)보다 빠르고 결정적.
  · 합logprob은 토큰수 많은 이름("rolled-in_scale")에 불리한 길이편향이 있어
    **평균 logprob**으로 정규화한 뒤 softmax 한다.

teacher는 폐쇄셋서 student보다 약하다(v4 95.9% vs CNN 99.6%). 그래서 이 증류의
값어치는 풀데이터 정확도가 아니라 **라벨이 적을 때의 데이터효율**에서 검증한다
(train_edge_kd.py의 라벨예산 스윕).

사용 (GPU):
    python scripts/kd_teacher_softlabels.py --limit 12      # 검증용
    python scripts/kd_teacher_softlabels.py                 # 전체 train 캐시
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app import config  # noqa: E402

SPLIT_DIR = ROOT / "data" / "processed"
TEACHER_ADAPTER = ROOT / "models" / "checkpoints" / "cand_v4"   # registry current=v4
CLASSES = config.DEFECT_CLASSES
DEFAULT_OUT = ROOT / "data" / "results" / "kd_teacher_softlabels.json"


def _load_split(name):
    recs = json.loads((SPLIT_DIR / f"{name}.json").read_text(encoding="utf-8"))
    out = []
    for r in recs:
        gt = json.loads(r["conversations"][1]["content"])["type"]
        out.append({"id": r["id"], "image": r["image"], "gt_type": gt})
    return out


def main():
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image

    ap = argparse.ArgumentParser(description="VLM teacher soft label 생성")
    ap.add_argument("--split", default="train")
    ap.add_argument("--adapter", type=Path, default=TEACHER_ADAPTER)
    ap.add_argument("--out", type=lambda s: Path(s).resolve(), default=DEFAULT_OUT)
    ap.add_argument("--limit", type=int, default=None, help="표본 수 제한(검증)")
    args = ap.parse_args()

    # app.main 의 4-bit 로딩/프롬프트를 그대로 재사용 (eval_on_test.py 와 동일 패턴)
    from app import main as appmain
    appmain.LORA_PATH = args.adapter
    appmain.USE_LORA = args.adapter.exists()
    if not appmain.USE_LORA:
        raise SystemExit(f"teacher 어댑터 경로가 없습니다: {args.adapter}")
    print(f"teacher 로드: {args.adapter}")
    appmain.load_model()
    model, processor = appmain.model, appmain.processor
    device = model.device

    # 강제 어시스턴트 접두사 + 각 클래스 완성문. softmax 는 후보 간 공통 접두사가
    # 상쇄되므로 클래스명 토큰의 '평균 logprob'만 신호로 남긴다.
    PREFIX = '{"type": "'
    completions = {c: PREFIX + c + '"' for c in CLASSES}

    data = _load_split(args.split)
    if args.limit:
        data = data[: args.limit]

    results = []
    n_argmax_ok = 0
    t0 = time.time()
    for i, ex in enumerate(data):
        p = Path(ex["image"])
        img = Image.open((ROOT / p) if not p.is_absolute() else p).convert("RGB")

        messages = [
            {"role": "system", "content": appmain.SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": appmain.INFERENCE_PROMPT},
            ]},
        ]
        prompt_text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        # 프롬프트(이미지 토큰 포함) 길이 — 완성문은 이 뒤에 깨끗이 이어붙는다.
        prompt_len = processor(text=[prompt_text], images=[img],
                               return_tensors="pt").input_ids.shape[1]

        mean_logp = {}
        for c, comp in completions.items():
            inputs = processor(text=[prompt_text + comp], images=[img],
                               return_tensors="pt").to(device)
            ids = inputs["input_ids"][0]
            with torch.no_grad():
                logits = model(**inputs).logits[0]          # [L, V]
            logp = F.log_softmax(logits.float(), dim=-1)
            # 클래스명+닫는따옴표 토큰 logprob (위치 i 토큰은 logits[i-1]이 예측)
            comp_ids = ids[prompt_len:]
            steps = [logp[j - 1, ids[j]].item() for j in range(prompt_len, len(ids))]
            mean_logp[c] = float(np.mean(steps)) if steps else -1e9

        scores = np.array([mean_logp[c] for c in CLASSES])
        probs = F.softmax(torch.tensor(scores), dim=-1).numpy()
        pred = CLASSES[int(probs.argmax())]
        if pred == ex["gt_type"]:
            n_argmax_ok += 1

        results.append({
            "id": ex["id"], "image": ex["image"], "gt_type": ex["gt_type"],
            "teacher_pred": pred,
            "mean_logp": {c: round(mean_logp[c], 5) for c in CLASSES},
            "soft": {c: round(float(probs[k]), 5) for k, c in enumerate(CLASSES)},
        })
        if (i + 1) % 20 == 0 or (i + 1) == len(data):
            print(f"  {i+1}/{len(data)}  teacher argmax acc {n_argmax_ok/(i+1):.3f}"
                  f"  ({(time.time()-t0)/(i+1):.2f}s/img)")

    acc = n_argmax_ok / len(data)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "split": args.split,
        "teacher_adapter": str(args.adapter.relative_to(ROOT)),
        "classes": CLASSES,
        "n": len(data),
        "scoring": "mean_logprob_over_classname_tokens -> softmax(T=1)",
        "teacher_argmax_accuracy": round(acc, 4),
        "seconds": round(time.time() - t0, 1),
        "items": results,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n완료: {len(data)}건, teacher argmax acc {acc:.4f}")
    print(f"저장: {args.out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
