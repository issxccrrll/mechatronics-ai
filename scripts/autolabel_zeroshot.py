"""Pre-etiquetado zero-shot con OWLv2 (detector de vocabulario abierto por texto).

Para clases que el modelo COCO no conoce (destornillador) o donde falla (celular por
detrás). Escribe UNA caja (la más probable) por imagen SIN .txt; nunca sobrescribe.
Es lento en CPU (~7 s/img): correrlo en segundo plano o en Colab.

Dependencia opcional (no está en requirements.txt): pip install transformers
Modelo: google/owlv2-base-patch16-ensemble (Minderer et al., 2023), se descarga solo.

Ej.: python scripts/autolabel_zeroshot.py --carpeta destornillador --texto "a photo of a screwdriver"
     python scripts/autolabel_zeroshot.py --carpeta celular --texto "a photo of a smartphone"
"""
import argparse
from pathlib import Path

import cv2
import yaml

RAIZ = Path(__file__).resolve().parent.parent
EXT = {".jpg", ".jpeg", ".png"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--carpeta", required=True, help="carpeta de dataset/raw (= clase oficial)")
    p.add_argument("--texto", required=True, help='descripción en inglés, ej. "a photo of a screwdriver"')
    p.add_argument("--raw", default=str(RAIZ / "dataset" / "raw"))
    p.add_argument("--umbral", type=float, default=0.20, help="confianza mínima para escribir la caja")
    p.add_argument("--revisar", type=float, default=0.35, help="debajo de esto se marca REVISAR en la vista")
    p.add_argument("--vista", action="store_true")
    args = p.parse_args()

    import torch
    from PIL import Image
    from transformers import Owlv2ForObjectDetection, Owlv2Processor

    with open(RAIZ / "dataset" / "data.yaml", encoding="utf-8") as f:
        names = yaml.safe_load(f)["names"]
    clases = [names[i] for i in sorted(names)]
    if args.carpeta not in clases:
        raise SystemExit(f"'{args.carpeta}' no es una clase oficial: {clases}")
    cls = clases.index(args.carpeta)

    proc = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
    modelo = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble").eval()

    carpeta = Path(args.raw) / args.carpeta
    pendientes = [f for f in sorted(carpeta.glob("*")) if f.suffix.lower() in EXT and not f.with_suffix(".txt").exists()]
    print(f"{args.carpeta}: {len(pendientes)} imágenes sin etiqueta")
    escritas, bajas = 0, 0
    for i, img in enumerate(pendientes, 1):
        im = Image.open(img).convert("RGB")
        ancho, alto = im.size
        with torch.no_grad():
            salida = modelo(**proc(text=[[args.texto]], images=im, return_tensors="pt"))
        lado = max(ancho, alto)  # OWLv2 rellena la imagen a un cuadrado
        r = proc.post_process_object_detection(salida, threshold=args.umbral,
                                               target_sizes=torch.tensor([[lado, lado]]))[0]
        if len(r["scores"]):
            k = int(r["scores"].argmax())
            score = float(r["scores"][k])
            x1, y1, x2, y2 = r["boxes"][k].tolist()
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(ancho, x2), min(alto, y2)
            cx, cy, w, h = (x1 + x2) / 2 / ancho, (y1 + y2) / 2 / alto, (x2 - x1) / ancho, (y2 - y1) / alto
            img.with_suffix(".txt").write_text(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n", encoding="utf-8")
            escritas += 1
            bajas += score < args.revisar
            if args.vista:
                guardar_vista(img, (x1, y1, x2, y2), args.carpeta, score, score < args.revisar)
        if i % 25 == 0 or i == len(pendientes):
            print(f"  {i}/{len(pendientes)} | con caja: {escritas} (dudosas: {bajas})", flush=True)
    print(f"Listo: {escritas} cajas escritas, {len(pendientes) - escritas} siguen sin etiqueta.")


def guardar_vista(img: Path, caja, nombre: str, score: float, dudosa: bool):
    destino = RAIZ / "dataset" / "revision" / img.parent.name
    destino.mkdir(parents=True, exist_ok=True)
    im = cv2.imread(str(img))
    x1, y1, x2, y2 = map(int, caja)
    cv2.rectangle(im, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(im, f"{nombre} {score:.2f}", (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    if dudosa:
        cv2.putText(im, "REVISAR (zero-shot)", (8, im.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.imwrite(str(destino / img.name), im)


if __name__ == "__main__":
    main()
