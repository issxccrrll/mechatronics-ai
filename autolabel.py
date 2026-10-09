"""Autoetiqueta fotos de dataset/raw/<clase>/ usando YOLO preentrenado en COCO.
Crea un .txt (formato YOLO) junto a cada foto donde detecte el objeto.

Ej.: python scripts/autolabel.py --clase botella --preview 10
"""
import argparse
from pathlib import Path

from ultralytics import YOLO

RAIZ = Path(__file__).resolve().parent.parent
EXT = {".jpg", ".jpeg", ".png"}

# ids según tu data.yaml
IDS = {"taza": 0, "botella": 1, "tijeras": 2, "destornillador": 3, "celular": 4}
# clase equivalente en COCO (destornillador no existe en COCO)
COCO = {"taza": 41, "botella": 39, "tijeras": 76, "celular": 67}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--clase", required=True, choices=list(COCO), help="carpeta dentro de dataset/raw/")
    p.add_argument("--raw", default=str(RAIZ / "dataset" / "raw"))
    p.add_argument("--model", default="yolo11n.pt")
    p.add_argument("--conf", type=float, default=0.4, help="confianza mínima de la detección")
    p.add_argument("--preview", type=int, default=10, help="guarda N fotos con la caja dibujada para revisar")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    carpeta = Path(args.raw) / args.clase
    fotos = sorted(f for f in carpeta.iterdir() if f.suffix.lower() in EXT)
    if not fotos:
        raise SystemExit(f"No hay fotos en {carpeta}")

    prev_dir = RAIZ / "runs" / "autolabel_preview"
    prev_dir.mkdir(parents=True, exist_ok=True)

    modelo = YOLO(args.model)
    etiquetadas, sin_deteccion, previews = 0, [], 0

    for foto in fotos:
        r = modelo.predict(str(foto), classes=[COCO[args.clase]], conf=args.conf,
                           device=args.device, verbose=False)[0]
        if len(r.boxes) == 0:
            sin_deteccion.append(foto.name)
            continue
        # una sola botella/objeto por foto: nos quedamos con la caja más segura
        mejor = int(r.boxes.conf.argmax())
        x, y, w, h = r.boxes.xywhn[mejor].tolist()
        foto.with_suffix(".txt").write_text(f"{IDS[args.clase]} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
        etiquetadas += 1
        if previews < args.preview:
            r.save(filename=str(prev_dir / foto.name))
            previews += 1

    print(f"Etiquetadas: {etiquetadas} / {len(fotos)}")
    if sin_deteccion:
        print(f"Sin detección ({len(sin_deteccion)}): se quedan sin .txt. Ejemplos: {sin_deteccion[:5]}")
    print(f"Revisa las muestras en: {prev_dir}")


if __name__ == "__main__":
    main()
