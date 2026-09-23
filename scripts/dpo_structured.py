"""DPO(RL post-training) 미니 데모 — 구조화 JSON 결함 리포트 선호 정렬.

VLM Defect Inspector의 "구조화 JSON 리포트" 주제를 LLM 후처리로 확장한다. SFT(QLoRA)만이
아니라 **RL 계열(DPO)** 까지 다룬다는 것을 작은 모델로 정직하게 실측한다.

문제: 결함 분류 질문에 모델이 **유효한 JSON**(필수필드 defect_type·severity·confidence·evidence)으로
답하게 만든다. chosen = 간결·유효 JSON, rejected = 산문/깨진 JSON. DPO로 선호를 정렬한 뒤 두 층위로 측정:
  (1) 행동 지표 — **JSON 유효율·필드 충족률을 base→DPO로 비교**(실제 생성물 채점).
  (2) DPO-native 지표 — held-out 프롬프트에서 **선호정확도·implicit reward margin**
      (r=beta·(logπ_policy−logπ_ref)). DPO가 직접 최적화하는 양이라, 선호를 암기 아닌
      일반화로 내면화했는지 검증한다.

설계 원칙(KD·head_to_head 트랙과 동일): 순수 로직은 데이터·모델 없이 자기점검(--smoke),
실측은 단일 GPU에서 LoRA 4-bit로. 합성 데이터임을 숨기지 않는다(정직한 데모).

사용:
    python scripts/dpo_structured.py --smoke                 # 데이터·파싱 로직 점검(모델 불요)
    python scripts/dpo_structured.py --pipeline-smoke        # 2-step DPO로 파이프라인 검증(GPU)
    python scripts/dpo_structured.py --epochs 3              # 전체 실측(GPU)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

RESULTS = ROOT / "data" / "results"
DEFECTS = ["scratch", "dent", "crack", "pitting", "discoloration", "contamination"]
SURFACES = ["강판", "알루미늄 패널", "스테인리스 표면", "도금 부품", "주물 표면", "용접 비드"]
SEVERITIES = ["low", "medium", "high"]
REQUIRED_FIELDS = ["defect_type", "severity", "confidence", "evidence"]


# ───────────────────────── 데이터 생성 (결정적) ─────────────────────────

def _prompt(surface: str, defect: str) -> str:
    return (f"다음 금속 표면 검사 결과를 구조화 JSON으로 보고하라. "
            f"표면: {surface}. 관측: {defect} 의심 패턴. "
            f"필수 키: defect_type, severity(low/medium/high), confidence(0~1), evidence(한 문장).")


def _chosen(defect: str, severity: str, conf: float) -> str:
    obj = {"defect_type": defect, "severity": severity, "confidence": conf,
           "evidence": f"표면에서 {defect} 특유의 국소 패턴이 관측됨."}
    return json.dumps(obj, ensure_ascii=False)


def _rejected(defect: str, kind: int) -> str:
    # 세 가지 나쁜 출력 유형: 산문 / 깨진 JSON / 필드 누락.
    if kind == 0:
        return f"이 표면에는 {defect} 결함이 있어 보입니다. 심각도는 중간 정도이고 추가 검사가 필요합니다."
    if kind == 1:
        return f'{{"defect_type": "{defect}", "severity": medium, confidence 0.7'  # 깨진 JSON
    return json.dumps({"defect": defect, "note": "확인 필요"}, ensure_ascii=False)  # 필드 누락


def build_pairs(n: int, seed: int = 0):
    """결정적 선호쌍 리스트(conversational 포맷). 반환: list[dict]."""
    import random
    rng = random.Random(seed)
    pairs = []
    for i in range(n):
        surface = SURFACES[i % len(SURFACES)]
        defect = DEFECTS[i % len(DEFECTS)]
        severity = SEVERITIES[rng.randint(0, 2)]
        conf = round(rng.uniform(0.55, 0.97), 2)
        pairs.append({
            "prompt": [{"role": "user", "content": _prompt(surface, defect)}],
            "chosen": [{"role": "assistant", "content": _chosen(defect, severity, conf)}],
            "rejected": [{"role": "assistant", "content": _rejected(defect, rng.randint(0, 2))}],
        })
    return pairs


# ───────────────────────── 평가 (JSON 유효율·필드 충족률) ─────────────────────────

def parse_score(text: str) -> tuple[bool, float]:
    """모델 출력에서 JSON을 추출·검증. 반환: (유효여부, 필수필드 충족비율)."""
    s = text.strip()
    start, end = s.find("{"), s.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return False, 0.0
    try:
        obj = json.loads(s[start:end + 1])
    except (json.JSONDecodeError, ValueError):
        return False, 0.0
    if not isinstance(obj, dict):
        return False, 0.0
    hit = sum(1 for k in REQUIRED_FIELDS if k in obj)
    return True, hit / len(REQUIRED_FIELDS)


# ─────────────── DPO-native 지표 (선호정확도·reward margin) 순수 집계 ───────────────
# DPO의 implicit reward: r(x,y) = beta·(logπ_policy(y|x) − logπ_ref(y|x)).
# margin = r(x, chosen) − r(x, rejected). 이 margin의 부호가 선호 방향, 크기가 분리 강도.
# JSON 유효율(행동 지표)과 달리, 이 둘은 DPO가 *직접* 최적화하는 양이라 학습이 실제로
# 선호를 내면화했는지를 held-out 프롬프트에서 검증한다.

def preference_accuracy(margins) -> float:
    """held-out 선호정확도 = mean[margin>0], 동률(==0)은 0.5로 계수(무학습=참조와 동일 시)."""
    if not margins:
        return 0.0
    wins = sum(1.0 if m > 0 else (0.5 if m == 0 else 0.0) for m in margins)
    return wins / len(margins)


def mean_margin(margins) -> float:
    """평균 reward margin. 0 = 참조와 구분 못 함(학습 전), 양수 = chosen을 선호."""
    return sum(margins) / len(margins) if margins else 0.0


# ───────────────────────── smoke (순수 로직) ─────────────────────────

def _smoke() -> int:
    pairs = build_pairs(12, seed=0)
    assert len(pairs) == 12
    for p in pairs:
        assert set(p) == {"prompt", "chosen", "rejected"}
        # chosen 은 유효 JSON·필드 충족, rejected 는 그렇지 않아야 학습 신호가 성립.
        ok_c, cov_c = parse_score(p["chosen"][0]["content"])
        ok_r, cov_r = parse_score(p["rejected"][0]["content"])
        assert ok_c and cov_c == 1.0, "chosen 은 완전한 JSON 이어야"
        assert (not ok_r) or cov_r < 1.0, "rejected 는 무효이거나 필드 누락이어야"
    # 깨진 JSON·필드 누락·산문 각각 무효 처리되는지.
    assert parse_score("그냥 설명입니다")[0] is False
    assert parse_score('{"defect_type":"x"}')[1] == 0.25
    # DPO-native 지표 집계 로직 검증(동률 0.5 계수·평균 margin).
    assert preference_accuracy([0.5, -0.2, 0.0, 1.0]) == 0.625
    assert abs(mean_margin([0.5, -0.2, 0.0, 1.0]) - 0.325) < 1e-9
    assert preference_accuracy([]) == 0.0 and mean_margin([]) == 0.0
    assert preference_accuracy([0.0, 0.0]) == 0.5  # 참조와 동일(학습 전) = 선호정확도 0.5
    print("smoke OK — 선호쌍 생성·JSON 파싱·필드 채점·DPO 선호지표 집계 순수 로직 정상")
    return 0


# ───────────────────────── 학습·평가 (GPU) ─────────────────────────

def _seq_logprob(model, tok, prompt_msgs, completion, device) -> float:
    """완성(completion) 토큰들의 로그확률 합 = logπ(completion | prompt). 프롬프트는 마스킹."""
    import torch
    prompt_ids = tok.apply_chat_template(
        prompt_msgs, add_generation_prompt=True, return_tensors="pt",
        return_dict=True)["input_ids"].to(device)
    comp_ids = tok(completion, return_tensors="pt",
                   add_special_tokens=False).input_ids.to(device)
    input_ids = torch.cat([prompt_ids, comp_ids], dim=1)
    with torch.no_grad():
        logits = model(input_ids).logits  # [1, T, V]
    logprobs = torch.log_softmax(logits.float(), dim=-1)
    plen, clen = prompt_ids.shape[1], comp_ids.shape[1]
    # 위치 t-1의 로짓이 토큰 t를 예측 → 완성 토큰은 [plen-1 : plen-1+clen] 위치가 담당.
    idx = torch.arange(plen - 1, plen - 1 + clen, device=device)
    tok_lp = logprobs[0, idx, :].gather(-1, comp_ids[0].unsqueeze(-1)).squeeze(-1)
    return float(tok_lp.sum().item())


def eval_preference(peft_model, tok, eval_pairs, beta, device):
    """held-out 선호정확도·평균 reward margin. 정책=어댑터 ON, 참조=어댑터 OFF(같은 base)."""
    margins = []
    for p in eval_pairs:
        pm, ch, rj = p["prompt"], p["chosen"][0]["content"], p["rejected"][0]["content"]
        lp_pol_c = _seq_logprob(peft_model, tok, pm, ch, device)
        lp_pol_r = _seq_logprob(peft_model, tok, pm, rj, device)
        with peft_model.disable_adapter():  # 어댑터 끄면 곧 참조 모델(DPO ref).
            lp_ref_c = _seq_logprob(peft_model, tok, pm, ch, device)
            lp_ref_r = _seq_logprob(peft_model, tok, pm, rj, device)
        r_c = beta * (lp_pol_c - lp_ref_c)
        r_r = beta * (lp_pol_r - lp_ref_r)
        margins.append(r_c - r_r)
    return preference_accuracy(margins), mean_margin(margins)


def run(model_name, epochs, n_train, n_eval, beta, pipeline_smoke):
    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig
    from trl import DPOConfig, DPOTrainer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"DPO 구조화 정렬 | model={model_name} | device={device} | "
          f"epochs={epochs} | train={n_train} eval={n_eval} | beta={beta}")

    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=bnb, torch_dtype=torch.bfloat16, device_map=device)

    train_ds = Dataset.from_list(build_pairs(n_train, seed=0))
    eval_pairs = build_pairs(n_eval, seed=999)

    # base(학습 전) JSON 유효율 측정 — 같은 평가 프롬프트로 before/after 비교.
    def eval_model(m, tag):
        m.eval()
        valid, cov = 0, 0.0
        for p in eval_pairs:
            msgs = p["prompt"]
            enc = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                          return_tensors="pt", return_dict=True)
            enc = {k: v.to(device) for k, v in enc.items()}
            with torch.no_grad():
                out = m.generate(**enc, max_new_tokens=128, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
            text = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
            ok, c = parse_score(text)
            valid += int(ok)
            cov += c
        vr, cr = valid / len(eval_pairs), cov / len(eval_pairs)
        print(f"  [{tag}] JSON 유효율 {vr:.3f} · 필드 충족률 {cr:.3f}")
        return vr, cr

    base_vr, base_cr = eval_model(model, "base")

    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                      task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
    cfg = DPOConfig(
        output_dir=str(ROOT / "models" / "dpo_structured"),
        per_device_train_batch_size=2, gradient_accumulation_steps=4,
        num_train_epochs=(1 if pipeline_smoke else epochs),
        max_steps=(2 if pipeline_smoke else -1),
        learning_rate=5e-5, beta=beta, logging_steps=5, save_strategy="no",
        report_to=[], bf16=True, max_length=512,
        warmup_ratio=0.1, lr_scheduler_type="cosine")
    trainer = DPOTrainer(model=model, args=cfg, train_dataset=train_ds,
                         processing_class=tok, peft_config=lora)
    trainer.train()

    after_vr, after_cr = eval_model(trainer.model, "DPO")

    # DPO-native 지표: held-out 프롬프트에서 선호를 실제로 내면화했는지(정책 vs 참조).
    pref_acc, pref_margin = eval_preference(trainer.model, tok, eval_pairs, beta, device)
    print(f"  [DPO] held-out 선호정확도 {pref_acc:.3f} · 평균 reward margin {pref_margin:+.3f}")

    if pipeline_smoke:
        print("pipeline-smoke OK — DPO 2-step 학습·평가·선호지표 전 경로 동작")
        return

    RESULTS.mkdir(parents=True, exist_ok=True)
    md = RESULTS / "dpo_structured.md"
    md.write_text(
        "# DPO 구조화 JSON 정렬 — base vs DPO (실측)\n\n"
        f"- 모델: `{model_name}` · 4-bit QLoRA + DPO(beta={beta}) · "
        f"train {n_train}쌍 · eval {n_eval}프롬프트 · epochs {epochs}\n\n"
        "### 행동 지표 (생성물 채점, greedy)\n\n"
        "| 지표 | base | DPO | Δ |\n|---|---|---|---|\n"
        f"| JSON 유효율 | {base_vr:.3f} | {after_vr:.3f} | {after_vr-base_vr:+.3f} |\n"
        f"| 필드 충족률 | {base_cr:.3f} | {after_cr:.3f} | {after_cr-base_cr:+.3f} |\n\n"
        "### DPO-native 지표 (held-out 선호, 정책 vs 참조 implicit reward)\n\n"
        "| 지표 | 학습 전(참조=정책) | DPO | 의미 |\n|---|---|---|---|\n"
        f"| 선호정확도 | 0.500 | {pref_acc:.3f} | margin>0 비율(동률 0.5) |\n"
        f"| 평균 reward margin | 0.000 | {pref_margin:+.3f} | r(chosen)−r(rejected) |\n\n"
        "> 합성 선호쌍(chosen=유효 JSON / rejected=산문·깨진 JSON·필드누락) 기반의 정직한 데모. "
        "SFT(QLoRA)에 더해 RL 계열(DPO)로 출력 형식을 정렬할 수 있음을 작은 모델로 실측. "
        "학습 전에는 정책=참조라 margin=0·선호정확도=0.5(구성상). held-out(seed 999, 미학습 프롬프트)에서 "
        "측정하므로 암기가 아닌 선호의 일반화를 본다.\n",
        encoding="utf-8")
    print(f"\n▶ JSON 유효율 base {base_vr:.3f} → DPO {after_vr:.3f} · "
          f"held-out 선호정확도 {pref_acc:.3f} | 저장: {md.relative_to(ROOT)}")


def main():
    ap = argparse.ArgumentParser(description="DPO 구조화 JSON 정렬 미니 데모")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--pipeline-smoke", action="store_true", help="2-step DPO로 전 경로 검증(GPU)")
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--n-train", type=int, default=120)
    ap.add_argument("--n-eval", type=int, default=24)
    ap.add_argument("--beta", type=float, default=0.1)
    args = ap.parse_args()

    if args.smoke:
        raise SystemExit(_smoke())
    run(args.model, args.epochs, args.n_train, args.n_eval, args.beta, args.pipeline_smoke)


if __name__ == "__main__":
    main()
