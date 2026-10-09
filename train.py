"""Entrenamiento con ultralytics (detección) con Data Augmentation ajustado.
Ej.: python scripts/train.py --epochs 80 --imgsz 640 --device 0
"""
import argparse
from pathlib import Path

from ultralytics import YOLO

RAIZ = Path(__file__).resolve().parent.parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="yolo11n.pt", help="pesos iniciales (transfer learning); prueba yolo11s.pt si tienes GPU")
    p.add_argument("--data", default=str(RAIZ / "dataset" / "data.yaml"))
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr0", type=float, default=0.01)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--device", default="cpu", help="cpu, 0 (GPU)...")
    p.add_argument("--name", default="mechatronics")
    p.add_argument("--close-mosaic", type=int, default=10,
                   help="Desactiva mosaic en las últimas N épocas para afinar con imágenes reales")

    # --- Parámetros de Data Augmentation (valores conservadores para detección) ---
    # 1. Color e iluminación
    p.add_argument("--hsv-h", type=float, default=0.015, help="Variación de tono (subir solo si el color NO define la clase)")
    p.add_argument("--hsv-s", type=float, default=0.7, help="Variación de saturación")
    p.add_argument("--hsv-v", type=float, default=0.4, help="Simula sombras y brillos de luz")

    # 2. Transformaciones geométricas
    p.add_argument("--degrees", type=float, default=10.0, help="Rotación +/- grados")
    p.add_argument("--translate", type=float, default=0.1, help="Desplaza la botella fuera del centro")
    p.add_argument("--scale", type=float, default=0.5, help="Botellas más grandes o más lejanas")
    p.add_argument("--shear", type=float, default=2.0, help="Deformación leve para simular perspectiva")
    p.add_argument("--perspective", type=float, default=0.0, help="Perspectiva de cámara (0.0–0.001)")

    # 3. Volteos
    p.add_argument("--fliplr", type=float, default=0.5, help="Volteo horizontal (espejo)")
    p.add_argument("--flipud", type=float, default=0.0, help="Volteo vertical (solo si hay botellas de cabeza reales)")

    # 4. Mezcla y recortes
    p.add_argument("--mosaic", type=float, default=1.0, help="Combina 4 imágenes en 1 (ayuda con oclusiones)")
    p.add_argument("--mixup", type=float, default=0.0, help="Mezcla dos imágenes y sus etiquetas (0.0–0.1 en datasets pequeños)")
    p.add_argument("--copy-paste", type=float, default=0.0,
                   help="Solo funciona con etiquetas de segmentación (polígonos); sin efecto con cajas")

    args = p.parse_args()

    modelo = YOLO(args.model)
    modelo.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        lr0=args.lr0,
        patience=args.patience,
        device=args.device,
        project=str(RAIZ / "runs"),
        name=args.name,
        exist_ok=True,
        close_mosaic=args.close_mosaic,
        # Augmentations
        hsv_h=args.hsv_h,
        hsv_s=args.hsv_s,
        hsv_v=args.hsv_v,
        degrees=args.degrees,
        translate=args.translate,
        scale=args.scale,
        shear=args.shear,
        perspective=args.perspective,
        fliplr=args.fliplr,
        flipud=args.flipud,
        mosaic=args.mosaic,
        mixup=args.mixup,
        copy_paste=args.copy_paste,
    )
    print(f"Mejor modelo: {modelo.trainer.best}")


if __name__ == "__main__":
    main()