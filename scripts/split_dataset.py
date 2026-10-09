"""Divide el dataset en train/val/test POR SESIÓN de captura (no al azar).

Así fotos casi idénticas de una misma sesión no quedan a la vez en train y test,
lo que inflaría las métricas.

Espera:  dataset/raw/<clase>/<sesion>_<n>.jpg  y etiquetas YOLO con el mismo nombre (.txt)
         en dataset/raw/<clase>/ (por ejemplo exportadas desde Roboflow/LabelImg).
Genera:  dataset/{train,val,test}/{images,labels}/
"""
import argparse
import random
import shutil
from collections import defaultdict
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
EXT = {".jpg", ".jpeg", ".png"}


def sesion_de(ruta: Path) -> str:
    return f"{ruta.parent.name}/{ruta.stem.rsplit('_', 1)[0]}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw", default=str(RAIZ / "dataset" / "raw"))
    p.add_argument("--salida", default=str(RAIZ / "dataset"))
    p.add_argument("--val", type=float, default=0.15)
    p.add_argument("--test", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    por_sesion = defaultdict(list)
    for img in Path(args.raw).rglob("*"):
        if img.suffix.lower() in EXT:
            por_sesion[sesion_de(img)].append(img)
    if not por_sesion:
        raise SystemExit(f"No hay imágenes en {args.raw}")

    # Repartir sesiones por clase para que cada split tenga todas las clases
    por_clase = defaultdict(list)
    for s in por_sesion:
        por_clase[s.split("/")[0]].append(s)

    rng = random.Random(args.seed)
    asignacion = {}
    for clase, sesiones in por_clase.items():
        rng.shuffle(sesiones)
        k = len(sesiones)
        n_test = max(1, round(k * args.test)) if k >= 3 else 0
        n_val = max(1, round(k * args.val)) if k >= 3 else 0
        for i, s in enumerate(sesiones):
            asignacion[s] = "test" if i < n_test else "val" if i < n_test + n_val else "train"
        if k < 3:
            print(f"[aviso] {clase}: solo {k} sesión(es); todo va a train. Captura más sesiones.")

    conteo = defaultdict(int)
    for s, imgs in por_sesion.items():
        split = asignacion[s]
        for img in imgs:
            for sub, archivo in (("images", img), ("labels", img.with_suffix(".txt"))):
                if archivo.exists():
                    destino = Path(args.salida) / split / sub
                    destino.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(archivo, destino / f"{img.parent.name}_{archivo.name}")
            conteo[split] += 1
    print("Imágenes por split:", dict(conteo))


if __name__ == "__main__":
    main()
