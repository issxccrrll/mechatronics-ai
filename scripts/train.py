"""Entrenamiento con ultralytics. Ej.: python scripts/train.py --epochs 80 --imgsz 640

Por defecto aplica augmentation de "cámara barata" (desenfoque, ruido, JPEG,
baja resolución, poca luz) para que el modelo sea robusto a webcams e
industriales de baja calidad. Desactivar con --sin-aug-camara.
"""
import argparse
from pathlib import Path

from ultralytics import YOLO

RAIZ = Path(__file__).resolve().parent.parent


def augment_camara_barata():
    """Transformaciones Albumentations que simulan una cámara de baja calidad."""
    try:
        import albumentations as A
    except ImportError:
        raise SystemExit("Falta albumentations: pip install albumentations "
                         "(en local luego: pip uninstall -y opencv-python-headless; "
                         "pip install --force-reinstall --no-deps opencv-python)")
    return [
        A.OneOf([A.MotionBlur(blur_limit=(3, 9)), A.GaussianBlur(blur_limit=(3, 7)),
                 A.Defocus(radius=(1, 3))], p=0.3),
        A.OneOf([A.GaussNoise(std_range=(0.02, 0.08)),
                 A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5))], p=0.3),
        A.ImageCompression(quality_range=(30, 80), p=0.3),
        A.Downscale(scale_range=(0.5, 0.9), p=0.2),
        A.RandomBrightnessContrast(brightness_limit=(-0.35, 0.15), contrast_limit=0.2, p=0.3),
        A.ToGray(p=0.05),
    ]


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
    p.add_argument("--sin-aug-camara", action="store_true", help="desactiva la augmentation de cámara barata")
    args = p.parse_args()

    extra = {} if args.sin_aug_camara else {"augmentations": augment_camara_barata()}
    modelo = YOLO(args.model)
    modelo.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                 lr0=args.lr0, patience=args.patience, device=args.device,
                 project=str(RAIZ / "runs"), name=args.name, exist_ok=True, **extra)
    print(f"Mejor modelo: {RAIZ / 'runs' / args.name / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()
