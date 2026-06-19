---
title: Metal Defect Inspector (Edge CNN)
emoji: 🔍
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 4.44.1
python_version: "3.11"
app_file: app.py
pinned: false
license: mit
---

# Metal Surface Defect Inspector — 엣지 CNN 라이브 데모

NEU 금속 표면 결함 6-class(`crazing`·`inclusion`·`patches`·`pitted_surface`·`rolled-in_scale`·`scratches`)
분류기. 이미지를 올리면 결함 유형·심각도·신뢰도를 즉시 판정합니다.

이 Space는 7B VLM이 아니라, 같은 데이터로 학습한 **경량 CNN(MobileNetV3-Small · 1.52M 파라미터 ·
6 MB)** 을 ONNX Runtime으로 띄운 것입니다. 폐쇄형 6-class에서 **test 정확도 99.6%**(같은 분할에서 7B
VLM 95.9%보다 높음)를 내면서 **CPU 단일코어 수 ms**에 돌아, 라인 인라인 검사에 실제로 올릴 수 있는
모델입니다. confidence < 0.80이면 운영 게이트가 사람 검토 큐로 보냅니다.

**전체 코드·README·실험:** https://github.com/MJHolics/vlm-defect-inspector

## 로컬 실행

```bash
pip install -r requirements.txt
python app.py   # http://localhost:7860
```
