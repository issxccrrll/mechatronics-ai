"""Integra etiquetas exportadas de Roboflow (formato YOLO) al dataset principal.

Resuelve dos diferencias con nuestro proyecto:
- Roboflow RENOMBRA los archivos: s20261009_230346_0001_jpg.rf.abc123.jpg -> se busca la foto
  original s20261009_230346_0001.jpg en dataset/raw/<clase>/.
- Roboflow ORDENA las clases alfabéticamente: los ids se traducen por NOMBRE usando el data.yaml
  de la exportación y el nuestro (dataset/data.yaml).

Acepta zips o carpetas exportadas (con data.yaml y */labels/*.txt).
Etiquetas vacías: solo se aceptan para fotos de "fondo" (en otras carpetas significarían
"sin objetos" y se omiten). No sobrescribe .txt existentes salvo --sobrescribir.

Ej.: python scripts/integrar_roboflow.py descargas/botella.zip descargas/tijeras.zip
"""
import argparse
import re
import tempfile
import zipfile
from collections import Counter
from pathlib import Path

import yaml

RAIZ = Path(__file__).resolve().parent.parent
PATRON = re.compile(r"(s\d{8}_\d{6}(?:_e-[a-z0-9-]+)?_\d+)")


def nombres_de(yaml_path: Path) -> list[str]:
    names = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))["names"]
    return [names[i] for i in sorted(names)] if isinstance(names, dict) else list(names)


def indexar_raw(raw: Path) -> dict[str, Path]:
    """stem original -> ruta de la foto en dataset/raw/<clase>/."""
    return {f.stem: f for f in raw.glob("*/*.jpg")}


def integrar_carpeta(export: Path, nuestras: list[str], raw_idx: dict, sobrescribir: bool, stats: Counter):
    yamls = list(export.rglob("data.yaml"))
    if not yamls:
        raise SystemExit(f"{export}: no tiene data.yaml (¿se exportó en formato YOLO?)")
    suyas = nombres_de(yamls[0])
    faltan = [n for n in suyas if n not in nuestras]
    if faltan:
        raise SystemExit(f"Clases de Roboflow que no existen en el proyecto: {faltan}")
    traduccion = {i: nuestras.index(n) for i, n in enumerate(suyas)}

    for lab in export.rglob("labels/*.txt"):
        m = PATRON.search(lab.stem)
        foto = raw_idx.get(m.group(1)) if m else None
        if foto is None:
            stats["sin_foto_original"] += 1
            continue
        lineas = []
        for ln in lab.read_text(encoding="utf-8").split("\n"):
            v = ln.split()
            if len(v) == 5:
                lineas.append(" ".join([str(traduccion[int(v[0])])] + v[1:]))
            elif len(v) >= 7 and len(v) % 2 == 1:
                # Polígono (Smart Polygon de Roboflow): "cls x1 y1 x2 y2 ..." -> caja que lo encierra
                xs, ys = [float(a) for a in v[1::2]], [float(a) for a in v[2::2]]
                cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
                w, h = max(xs) - min(xs), max(ys) - min(ys)
                lineas.append(f"{traduccion[int(v[0])]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
                stats["poligonos_convertidos"] += 1
            elif v:
                stats["linea_no_valida"] += 1
        if not lineas and foto.parent.name != "fondo":
            stats["vacia_omitida"] += 1
            continue
        destino = foto.with_suffix(".txt")
        if destino.exists() and not sobrescribir:
            stats["ya_existia"] += 1
            continue
        destino.write_text("\n".join(lineas) + ("\n" if lineas else ""), encoding="utf-8")
        stats[f"escritas_{foto.parent.name}"] += 1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("exportaciones", nargs="+", help="zips o carpetas exportadas de Roboflow")
    p.add_argument("--raw", default=str(RAIZ / "dataset" / "raw"))
    p.add_argument("--sobrescribir", action="store_true")
    args = p.parse_args()

    nuestras = nombres_de(RAIZ / "dataset" / "data.yaml")
    raw_idx = indexar_raw(Path(args.raw))
    stats = Counter()
    for ruta in map(Path, args.exportaciones):
        if ruta.suffix.lower() == ".zip":
            with tempfile.TemporaryDirectory() as tmp:
                zipfile.ZipFile(ruta).extractall(tmp)
                integrar_carpeta(Path(tmp), nuestras, raw_idx, args.sobrescribir, stats)
        else:
            integrar_carpeta(ruta, nuestras, raw_idx, args.sobrescribir, stats)
    for k, v in sorted(stats.items()):
        print(f"{k:28s} {v}")


if __name__ == "__main__":
    main()
