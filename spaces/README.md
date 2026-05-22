---
title: VLM Defect Inspector
emoji: 🔍
colorFrom: blue
colorTo: gray
sdk: gradio
sdk_version: "4.44.0"
app_file: app.py
pinned: false
license: mit
---

# VLM Defect Inspector

Qwen2.5-VL 7B + QLoRA 기반 금속 표면 불량 분류 시스템.

Zero-shot 33.7% → Best Combo (rank32 + aug + label smoothing) **82.6%** Type Accuracy.

## 배포 방법

1. LoRA 어댑터 업로드:
   ```bash
   huggingface-cli login
   python spaces/upload_adapter.py --repo your-username/vlm-defect-inspector-lora
   ```

2. HuggingFace Spaces 생성 후 `spaces/` 폴더 내용을 업로드

3. Spaces > Settings > Environment Variables에 추가:
   ```
   HF_ADAPTER_REPO = your-username/vlm-defect-inspector-lora
   ```

4. Spaces > Settings > GPU 요청 (ZeroGPU 또는 T4 이상 권장)
