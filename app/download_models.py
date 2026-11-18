"""Unduh bobot model ke app/models/ (sekali saja).

- ppe_best.pt : keremberke/yolov8m-protective-equipment-detection (HuggingFace)
                kelas: glove, goggles, helmet, mask, shoes + versi no_* (helmet tidak dipakai)
- helmet_best.pt : keremberke/yolov8s-hard-hat-detection (HuggingFace), kelas Hardhat / NO-Hardhat
- yolov8n.pt  : Ultralytics COCO nano, dipakai untuk orang + kendaraan + tracking
"""
import os
import shutil
from pathlib import Path

MODELS = Path(__file__).resolve().parent / "models"
MODELS.mkdir(exist_ok=True)


def main():
    ppe = MODELS / "ppe_best.pt"
    if not ppe.exists():
        from huggingface_hub import hf_hub_download
        p = hf_hub_download("keremberke/yolov8m-protective-equipment-detection", "best.pt")
        shutil.copy(p, ppe)
    print("PPE  :", ppe, ppe.stat().st_size // 1024, "KB")

    helmet = MODELS / "helmet_best.pt"
    if not helmet.exists():
        from huggingface_hub import hf_hub_download
        p = hf_hub_download("keremberke/yolov8s-hard-hat-detection", "best.pt")
        shutil.copy(p, helmet)
    print("Helm :", helmet, helmet.stat().st_size // 1024, "KB")

    coco = MODELS / "yolov8n.pt"
    if not coco.exists():
        from ultralytics import YOLO
        cwd = os.getcwd()
        os.chdir(MODELS)  # ultralytics menyimpan bobot di folder kerja
        try:
            YOLO("yolov8n.pt")
        finally:
            os.chdir(cwd)
    print("COCO :", coco, coco.stat().st_size // 1024, "KB")


if __name__ == "__main__":
    main()
