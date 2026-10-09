"""Exporta un .pt a OpenVINO y ONNX. Ej.: python scripts/export.py --model models/best.pt"""
import argparse

from ultralytics import YOLO


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="models/best.pt")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--formatos", nargs="+", default=["openvino", "onnx"])
    p.add_argument("--half", action="store_true", help="FP16 (OpenVINO)")
    args = p.parse_args()

    modelo = YOLO(args.model)
    for fmt in args.formatos:
        ruta = modelo.export(format=fmt, imgsz=args.imgsz, half=args.half and fmt == "openvino")
        print(f"[{fmt}] -> {ruta}")


if __name__ == "__main__":
    main()
