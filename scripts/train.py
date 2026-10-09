"""Entrenamiento con ultralytics. Ej.: python scripts/train.py --epochs 80 --imgsz 640"""
import argparse
from pathlib import Path

from ultralytics import YOLO

RAIZ = Path(__file__).resolve().parent.parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="yolo11n.pt", help="pesos iniciales (transfer learning)")
    p.add_argument("--data", default=str(RAIZ / "dataset" / "data.yaml"))
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr0", type=float, default=0.01)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--device", default="cpu", help="cpu, 0 (GPU)...")
    p.add_argument("--name", default="mechatronics")
    args = p.parse_args()

    modelo = YOLO(args.model)
    modelo.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                 lr0=args.lr0, patience=args.patience, device=args.device,
                 project=str(RAIZ / "runs"), name=args.name, exist_ok=True)
    print(f"Mejor modelo: {RAIZ / 'runs' / args.name / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()
