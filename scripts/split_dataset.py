"""Divide el dataset en train/val/test POR SESIÓN de captura (no al azar).

Así fotos casi idénticas de una misma sesión no quedan a la vez en train y test,
lo que inflaría las métricas.

--ejemplares-test tijeras-verdes taza-azul: manda TODAS las sesiones de esos objetos
físicos (capturados con capture.py --ejemplar) a test, y ninguna a train/val. Así el test
mide si el modelo reconoce ejemplares que nunca vio (ej. tijeras de otro color).

Respaldo: si una clase tiene menos de 3 sesiones, cada sesión se divide en BLOQUES
consecutivos (primer 70% train, siguiente 15% val, último 15% test) para que todas las
clases tengan val y test. Las fotos seguidas (casi idénticas) quedan en el mismo bloque.

Espera:  dataset/raw/<clase>/<sesion>_<n>.jpg  y etiquetas YOLO con el mismo nombre (.txt)
         en dataset/raw/<clase>/ (por ejemplo exportadas desde Roboflow/LabelImg).
Genera:  dataset/{train,val,test}/{images,labels}/
"""
import argparse
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
EXT = {".jpg", ".jpeg", ".png"}


PATRON_SESION = re.compile(r"s\d{8}_\d{6}(?:_e-[a-z0-9-]+)?")
PATRON_EJEMPLAR = re.compile(r"_e-([a-z0-9-]+)")


def sesion_de(ruta: Path) -> str:
    """'<carpeta>/<sesion>'. La sesión se busca con regex para tolerar archivos renombrados
    por herramientas de etiquetado (ej. Roboflow: s20261008_232241_0001_jpg.rf.abc123.jpg)."""
    m = PATRON_SESION.search(ruta.name)
    sesion = m.group(0) if m else ruta.stem.rsplit("_", 1)[0]
    return f"{ruta.parent.name}/{sesion}"


def etiqueta_de(img: Path) -> Path:
    """Etiqueta junto a la imagen, o en la carpeta labels/ hermana (formato de exportación YOLO)."""
    junto = img.with_suffix(".txt")
    if junto.exists() or img.parent.name != "images":
        return junto
    return img.parent.parent / "labels" / f"{img.stem}.txt"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw", default=str(RAIZ / "dataset" / "raw"))
    p.add_argument("--salida", default=str(RAIZ / "dataset"))
    p.add_argument("--val", type=float, default=0.15)
    p.add_argument("--test", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--bloques", action="store_true",
                   help="divide TODAS las clases por bloques 70/15/15 dentro de cada sesión "
                        "(más datos de entrenamiento; test menos estricto)")
    p.add_argument("--reemplazar", action="store_true",
                   help="borra train/, val/ y test/ de la salida antes de dividir")
    p.add_argument("--ejemplares-test", nargs="*", default=[],
                   help="objetos físicos reservados solo para test (ej. tijeras-verdes)")
    args = p.parse_args()
    reservados = {e.lower() for e in args.ejemplares_test}

    salida = Path(args.salida)
    previas = [salida / sp for sp in ("train", "val", "test") if (salida / sp).exists()]
    if previas and not args.reemplazar:
        raise SystemExit(f"Ya existen {[str(d) for d in previas]}. Mezclarían fotos viejas con nuevas; "
                         "usa --reemplazar para borrarlas y volver a dividir.")
    for d in previas:
        shutil.rmtree(d)

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
        # Ejemplares reservados -> test completo; la clase ya tiene test "no visto"
        fijas = [s for s in sesiones if (m := PATRON_EJEMPLAR.search(s)) and m.group(1) in reservados]
        for s in fijas:
            asignacion[s] = "test"
        sesiones = [s for s in sesiones if s not in fijas]
        rng.shuffle(sesiones)
        k = len(sesiones)
        if fijas:  # el resto solo se reparte entre train y val
            n_val = max(1, round(k * args.val)) if k >= 2 else 0
            for i, s in enumerate(sesiones):
                asignacion[s] = "val" if i < n_val else "train"
            print(f"[test no visto] {clase}: {len(fijas)} sesión(es) de ejemplares reservados")
            continue
        n_test = max(1, round(k * args.test)) if k >= 3 else 0
        n_val = max(1, round(k * args.val)) if k >= 3 else 0
        for i, s in enumerate(sesiones):
            asignacion[s] = "test" if i < n_test else "val" if i < n_test + n_val else "train"
        if k < 3 or args.bloques:
            for s in sesiones:
                asignacion[s] = "bloques"
            if not args.bloques:
                print(f"[aviso] {clase}: solo {k} sesión(es); se divide cada sesión en bloques 70/15/15. "
                      "Captura más sesiones para un test más honesto.")

    conteo, sin_etiqueta = defaultdict(int), []
    for s, imgs in por_sesion.items():
        imgs = sorted(imgs)
        for i, img in enumerate(imgs):
            split = asignacion[s]
            if split == "bloques":
                pos = i / len(imgs)
                split = "train" if pos < 0.70 else "val" if pos < 0.85 else "test"
            etiqueta = etiqueta_de(img)
            # Sin .txt YOLO la tomaría como fondo: solo se permite en la carpeta "fondo"
            if not etiqueta.exists() and img.parent.name != "fondo":
                sin_etiqueta.append(img)
                continue
            for sub, archivo in (("images", img), ("labels", etiqueta)):
                if archivo.exists():
                    destino = Path(args.salida) / split / sub
                    destino.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(archivo, destino / f"{img.parent.name}_{archivo.name}")
            conteo[split] += 1
    print("Imágenes por split:", dict(conteo))
    if sin_etiqueta:
        print(f"[aviso] {len(sin_etiqueta)} imágenes omitidas por no tener etiqueta (.txt). Ej.: {sin_etiqueta[0]}")


if __name__ == "__main__":
    main()
