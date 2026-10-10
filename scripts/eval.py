"""Evalúa un modelo sobre el split test y escribe metricas.md, metricas.csv y la matriz de confusión.

- mAP, precision y recall con el protocolo estándar (conf 0.001, curva completa).
- Matriz de confusión y precision/recall al umbral de OPERACIÓN (el que se usa en vivo).

Ej.: python scripts/eval.py --model models/best.pt
"""
import argparse
import shutil
from datetime import date
from pathlib import Path

import pandas as pd
import yaml
from ultralytics import YOLO

RAIZ = Path(__file__).resolve().parent.parent


def tabla_md(df: pd.DataFrame) -> str:
    """DataFrame -> tabla Markdown (sin depender de tabulate)."""
    filas = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    filas += ["| " + " | ".join(str(v) for v in fila) + " |" for fila in df.itertuples(index=False)]
    return "\n".join(filas)


def por_clase(res) -> pd.DataFrame:
    filas = [{"clase": "todas", "precision": res.box.mp, "recall": res.box.mr,
              "mAP50": res.box.map50, "mAP50-95": res.box.map}]
    for i, c in enumerate(res.ap_class_index):
        pr, rc, ap50, ap = res.box.class_result(i)
        filas.append({"clase": res.names[int(c)], "precision": pr, "recall": rc, "mAP50": ap50, "mAP50-95": ap})
    return pd.DataFrame(filas).round(4)


def matriz_md(res) -> str:
    """Matriz de confusión numérica (filas = predicción, columnas = real), como la imagen de ultralytics."""
    nombres = [res.names[i] for i in sorted(res.names)] + ["fondo"]
    m = res.confusion_matrix.matrix.astype(int)
    df = pd.DataFrame(m, columns=nombres)
    df.insert(0, "pred \\ real", nombres)
    return tabla_md(df)


def preparar_data_yaml(ruta: Path) -> str:
    """El repo solo incluye dataset/test. Ultralytics exige que existan train y val aunque se evalúe
    test, así que si faltan se apuntan a test en un yaml temporal (no afecta las métricas de test)."""
    cfg = yaml.safe_load(ruta.read_text(encoding="utf-8"))
    base = ruta.parent
    faltan = [k for k in ("train", "val") if not (base / cfg[k]).exists()]
    if not faltan:
        return str(ruta)
    cfg["path"] = str(base.resolve())
    for k in faltan:
        cfg[k] = cfg["test"]
    tmp = RAIZ / "runs" / "data_eval.yaml"
    tmp.parent.mkdir(exist_ok=True)
    tmp.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    print(f"[eval] Sin {faltan} en el dataset: se usa test en su lugar solo para cargar ({tmp}).")
    return str(tmp)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="models/best.pt")
    p.add_argument("--data", default=str(RAIZ / "dataset" / "data.yaml"))
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf-operacion", type=float, default=None,
                   help="umbral global de operación; por defecto el de config.yaml (modelo.conf)")
    args = p.parse_args()
    if args.conf_operacion is None:

        args.conf_operacion = yaml.safe_load((RAIZ / "config.yaml").read_text(encoding="utf-8"))["modelo"]["conf"]

    modelo = YOLO(args.model)
    datos = preparar_data_yaml(Path(args.data))
    comun = dict(data=datos, split="test", imgsz=args.imgsz, device="cpu", project=str(RAIZ / "runs"))
    res = modelo.val(conf=0.001, name="eval", exist_ok=True, **comun)
    res_op = modelo.val(conf=args.conf_operacion, name="eval_operacion", exist_ok=True, **comun)

    df = por_clase(res)
    df.to_csv(RAIZ / "metricas.csv", index=False)
    df_op = por_clase(res_op)[["clase", "precision", "recall"]]

    for origen, destino in (("confusion_matrix.png", "confusion_matrix.png"),
                            ("confusion_matrix_normalized.png", "confusion_matrix_normalized.png")):
        src = Path(res_op.save_dir) / origen
        if src.exists():
            (RAIZ / "docs").mkdir(exist_ok=True)
            shutil.copy2(src, RAIZ / "docs" / destino)

    (RAIZ / "metricas.md").write_text(
        f"# Métricas\n\nModelo: `{args.model}` · split: **test** (separado de train/val) · fecha: {date.today()}\n\n"
        f"Reproducir: `python scripts/eval.py --model {args.model}`\n\n"
        f"## Resultados por clase (protocolo estándar, conf 0.001)\n\n{tabla_md(df)}\n\n"
        f"## Precision y recall al umbral de operación (conf {args.conf_operacion})\n\n{tabla_md(df_op)}\n\n"
        f"## Matriz de confusión (conf {args.conf_operacion})\n\n"
        f"Filas = predicción, columnas = clase real; \"fondo\" = sin objeto.\n\n{matriz_md(res_op)}\n\n"
        f"![Matriz de confusión](docs/confusion_matrix.png)\n",
        encoding="utf-8")
    print(df.to_string(index=False))
    print(f"\nAl umbral de operación {args.conf_operacion}:\n{df_op.to_string(index=False)}")


if __name__ == "__main__":
    main()
