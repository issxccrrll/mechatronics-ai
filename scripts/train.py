"""Entrenamiento con ultralytics. Ej.: python scripts/train.py --epochs 80 --device 0

Pensado para correr en GPU (Colab/Kaggle, ver notebooks/train_colab.ipynb); en CPU es
viable solo para pruebas cortas.

Augmentation para que el modelo GENERALICE a ejemplares que no vio (ej. tijeras de
otro color) y aprenda la FORMA en vez de atajos:
- Color: tono muy variable (hsv_h), canales intercambiados y escala de grises.
- Sombras e iluminación: sombras artificiales y brillo/contraste aleatorios.
- Geometría: rotación, perspectiva y volteos (objetos acostados o de cabeza).
- Cámara barata: desenfoque, ruido, JPEG y baja resolución.
Desactivar con --sin-aug-extra para comparar.
"""
import argparse
from pathlib import Path

from ultralytics import YOLO

RAIZ = Path(__file__).resolve().parent.parent

# Hiperparámetros de augmentation nativos de ultralytics (los defaults entre paréntesis)
AUG_NATIVA = dict(
    hsv_h=0.4,         # (0.015) tono: un mango rojo puede verse azul/verde -> el color deja de ser pista
    hsv_s=0.7,         # (0.7) saturación
    hsv_v=0.5,         # (0.4) brillo: luz tenue / fuerte
    degrees=15.0,      # (0.0) rotación
    perspective=0.0005,  # (0.0) ligera perspectiva (cámara inclinada)
    flipud=0.2,        # (0.0) de cabeza: tazas boca abajo, tijeras invertidas
    fliplr=0.5,        # (0.5)
    mosaic=1.0,        # (1.0) mezcla 4 fotos: fondos y escalas variados
    close_mosaic=10,   # (10) últimas épocas sin mosaico
)


def augment_albumentations():
    """Transformaciones de píxel (no mueven cajas) con Albumentations."""
    try:
        import albumentations as A
    except ImportError:
        raise SystemExit("Falta albumentations: pip install albumentations "
                         "(en local luego: pip uninstall -y opencv-python-headless; "
                         "pip install --force-reinstall --no-deps opencv-python)")
    return [
        # Color: obliga a usar la forma
        A.ChannelShuffle(p=0.15),
        A.ToGray(p=0.15),
        # Sombras e iluminación
        A.RandomShadow(shadow_roi=(0, 0.3, 1, 1), num_shadows_limit=(1, 2), shadow_dimension=5,
                       shadow_intensity_range=(0.3, 0.6), p=0.25),
        A.RandomBrightnessContrast(brightness_limit=(-0.35, 0.15), contrast_limit=0.2, p=0.3),
        # Cámara barata
        A.OneOf([A.MotionBlur(blur_limit=(3, 9)), A.GaussianBlur(blur_limit=(3, 7)),
                 A.Defocus(radius=(1, 3))], p=0.3),
        A.OneOf([A.GaussNoise(std_range=(0.02, 0.08)),
                 A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5))], p=0.3),
        A.ImageCompression(quality_range=(30, 80), p=0.3),
        A.Downscale(scale_range=(0.5, 0.9), p=0.2),
    ]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="yolo11n.pt", help="pesos iniciales (transfer learning desde COCO)")
    p.add_argument("--data", default=str(RAIZ / "dataset" / "data.yaml"))
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16, help="-1 = automático según memoria de la GPU")
    p.add_argument("--lr0", type=float, default=0.01)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--device", default="cpu", help="cpu, 0 (GPU)...")
    p.add_argument("--cache", default=False, help="ram o disk: acelera las épocas (útil en Colab)")
    p.add_argument("--project", default=str(RAIZ / "runs"), help="carpeta de salida (ej. en Google Drive)")
    p.add_argument("--name", default="mechatronics")
    p.add_argument("--sin-aug-extra", action="store_true", help="usa solo la augmentation por defecto")
    args = p.parse_args()

    extra = {} if args.sin_aug_extra else {**AUG_NATIVA, "augmentations": augment_albumentations()}
    modelo = YOLO(args.model)
    modelo.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                 lr0=args.lr0, patience=args.patience, device=args.device, cache=args.cache,
                 project=args.project, name=args.name, exist_ok=True, **extra)
    print(f"Mejor modelo: {Path(args.project) / args.name / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()
