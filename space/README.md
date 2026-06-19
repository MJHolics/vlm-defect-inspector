---
title: Metal Defect Inspector (Edge CNN)
emoji: 🔍
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.19.0
python_version: "3.13"
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

> ⚠️ **적용 범위·한계.** NEU-DET(200×200 흑백, 균일 조명의 미세 표면 텍스처) 6개 결함만 학습했습니다.
> 웹의 일반 사진(컬러·다른 배율/조명, 거시 균열, 6클래스 밖 결함)은 학습 분포 밖(OOD)이라 라벨이
> 부정확할 수 있습니다. 같은 NEU 분포 정확도는 99.6%, 낯선 입력은 confidence < 0.80에서 사람 검토로
> 보냅니다. 폐쇄셋 성능과 안전 게이트 동작 시연용이며 임의 현장 이미지 진단을 보장하지 않습니다.

**전체 코드·README·실험:** https://github.com/MJHolics/vlm-defect-inspector

## 로컬 실행

```bash
pip install -r requirements.txt
python app.py   # http://localhost:7860
```
