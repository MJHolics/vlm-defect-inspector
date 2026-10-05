"""NEU-DET 결함 검출기 학습 — 분류기가 못 내는 "결함이 어디 있나"를 박스로 낸다.

엣지 CNN(MobileNetV3)은 종류만 맞히고 위치를 내지 않는다. NEU-DET에는 정답 박스가 있으므로
YOLOv8n을 학습해 위치를 직접 낸다. 입력은 원본이 200x200이라 320으로 둔다(CPU 추론 대상).

사용: python scripts/train_detector.py [--epochs 100] [--imgsz 320]
산출: data/neu_det_yolo/(변환본) · models/detector/(가중치·ONNX) · reports/detector_val.json
"""
from __future__ import annotations

import argparse
import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "NEU-DET"
DST = ROOT / "data" / "neu_det_yolo"
CLASSES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]


def convert() -> Path:
    """VOC xml → YOLO txt. 이미지는 평평한 폴더로 복사한다."""
    for split, name in (("train", "train"), ("validation", "val")):
        img_out = DST / "images" / name
        lbl_out = DST / "labels" / name
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)
        for img in (SRC / split / "images").rglob("*.jpg"):
            xml = SRC / split / "annotations" / (img.stem + ".xml")
            if not xml.exists():
                continue
            r = ET.parse(xml).getroot()
            w, h = int(r.findtext("size/width")), int(r.findtext("size/height"))
            lines = []
            for o in r.iter("object"):
                b = o.find("bndbox")
                x0, y0, x1, y1 = (float(b.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax"))
                lines.append(f"{CLASSES.index(o.findtext('name'))} {(x0 + x1) / 2 / w:.6f} "
                             f"{(y0 + y1) / 2 / h:.6f} {(x1 - x0) / w:.6f} {(y1 - y0) / h:.6f}")
            shutil.copy(img, img_out / img.name)
            (lbl_out / (img.stem + ".txt")).write_text("\n".join(lines), encoding="utf-8")
    yaml = DST / "data.yaml"
    names = "\n".join(f"  {i}: {c}" for i, c in enumerate(CLASSES))
    yaml.write_text(f"path: {DST.as_posix()}\ntrain: images/train\nval: images/val\nnames:\n{names}\n",
                    encoding="utf-8")
    return yaml


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=320)
    args = ap.parse_args()

    from ultralytics import YOLO

    yaml = convert()
    model = YOLO("yolov8n.pt")
    model.train(data=str(yaml), epochs=args.epochs, imgsz=args.imgsz, batch=32, seed=0,
                project=str(ROOT / "models" / "detector"), name="yolov8n_neu", exist_ok=True,
                workers=0, verbose=False)
    best = YOLO(str(ROOT / "models" / "detector" / "yolov8n_neu" / "weights" / "best.pt"))
    m = best.val(data=str(yaml), imgsz=args.imgsz, workers=0, verbose=False)
    out = {
        "imgsz": args.imgsz, "epochs": args.epochs,
        "map50": round(float(m.box.map50), 4), "map50_95": round(float(m.box.map), 4),
        "per_class_ap50": {CLASSES[i]: round(float(v), 4) for i, v in zip(m.box.ap_class_index, m.box.ap50)},
    }
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "detector_val.json").write_text(json.dumps(out, ensure_ascii=False, indent=1),
                                                        encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=1))
    print("onnx:", best.export(format="onnx", imgsz=args.imgsz, opset=12, simplify=True))


if __name__ == "__main__":
    main()
