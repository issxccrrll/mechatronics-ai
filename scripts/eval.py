"""Evalúa un modelo sobre el split test y escribe metricas.md y metricas.csv.

Ej.: python scripts/eval.py --model models/best.pt
"""
import argparse
import shutil
from datetime import date
from pathlib import Path

import pandas as pd
from ultralytics import YOLO

RAIZ = Path(__file__).resolve().parent.parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="models/best.pt")
    p.add_argument("--data", default=str(RAIZ / "dataset" / "data.yaml"))
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.001)
    args = p.parse_args()

    res = YOLO(args.model).val(data=args.data, split="test", imgsz=args.imgsz, conf=args.conf,
                               device="cpu", project=str(RAIZ / "runs"), name="eval", exist_ok=True)

    filas = [{"clase": "todas", "precision": res.box.mp, "recall": res.box.mr,
              "mAP50": res.box.map50, "mAP50-95": res.box.map}]
    for i, c in enumerate(res.ap_class_index):
        pr, rc, ap50, ap = res.box.class_result(i)
        filas.append({"clase": res.names[int(c)], "precision": pr, "recall": rc, "mAP50": ap50, "mAP50-95": ap})
    df = pd.DataFrame(filas).round(4)
    df.to_csv(RAIZ / "metricas.csv", index=False)

    matriz_src = Path(res.save_dir) / "confusion_matrix.png"
    matriz_dst = RAIZ / "docs" / "confusion_matrix.png"
    if matriz_src.exists():
        matriz_dst.parent.mkdir(exist_ok=True)
        shutil.copy2(matriz_src, matriz_dst)

    tabla = df.to_markdown(index=False) if _tiene_tabulate() else df.to_string(index=False)
    (RAIZ / "metricas.md").write_text(
        f"# Métricas\n\nModelo: `{args.model}` · split: test · fecha: {date.today()}\n\n"
        f"## Resultados por clase\n\n{tabla}\n\n"
        f"## Matriz de confusión\n\n![Matriz de confusión](docs/confusion_matrix.png)\n",
        encoding="utf-8")
    print(df.to_string(index=False))


def _tiene_tabulate() -> bool:
    try:
        import tabulate  # noqa: F401
        return True
    except ImportError:
        return False


if __name__ == "__main__":
    main()
