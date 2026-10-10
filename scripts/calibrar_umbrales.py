"""Calibra el umbral de confianza por clase con el split de validación.

Para cada clase elige el umbral MÁS BAJO cuyo F1 queda a menos de --tolerancia del F1
máximo (la curva suele ser plana cerca del máximo; un umbral bajo conserva recall en casos
difíciles que val casi no tiene, como objetos en la mano), sin bajar de --minimo.
Imprime la tabla y, con --aplicar, la escribe en config.yaml
(filtros.class_threshold.umbrales).

Ej.: python scripts/calibrar_umbrales.py --model models/best.pt
     python scripts/calibrar_umbrales.py --model models/best.pt --aplicar
"""
import argparse
import re
from pathlib import Path

import numpy as np
from ultralytics import YOLO

RAIZ = Path(__file__).resolve().parent.parent


def mejor_umbral(f1: np.ndarray, px: np.ndarray, minimo: float, tolerancia: float) -> tuple[float, float]:
    """(umbral, F1): el umbral más bajo >= minimo con F1 >= F1_max - tolerancia."""
    validos = px >= minimo
    f1_max = float(np.max(np.where(validos, f1, -1)))
    i = int(np.argmax(validos & (f1 >= f1_max - tolerancia)))
    return float(px[i]), float(f1[i])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="models/best.pt")
    p.add_argument("--data", default=str(RAIZ / "dataset" / "data.yaml"))
    p.add_argument("--minimo", type=float, default=0.20, help="umbral más bajo permitido")
    p.add_argument("--tolerancia", type=float, default=0.02, help="F1 que se acepta perder por bajar el umbral")
    p.add_argument("--aplicar", action="store_true", help="escribe los umbrales en config.yaml")
    args = p.parse_args()

    res = YOLO(args.model).val(data=args.data, split="val", conf=0.001, device="cpu",
                               project=str(RAIZ / "runs"), name="calibracion", exist_ok=True, plots=False)
    px, f1, p_curve, r_curve = res.box.px, res.box.f1_curve, res.box.p_curve, res.box.r_curve
    umbrales = {}
    print(f"{'clase':15s} {'umbral':>7s} {'F1':>6s} {'precision':>10s} {'recall':>7s}")
    for fila, c in enumerate(res.ap_class_index):
        nombre = res.names[int(c)]
        u, f = mejor_umbral(f1[fila], px, args.minimo, args.tolerancia)
        k = int(np.argmin(np.abs(px - u)))
        umbrales[nombre] = round(u, 2)
        print(f"{nombre:15s} {u:7.2f} {f:6.3f} {p_curve[fila][k]:10.3f} {r_curve[fila][k]:7.3f}")

    if args.aplicar:
        ruta = RAIZ / "config.yaml"
        texto = ruta.read_text(encoding="utf-8")
        for nombre, u in umbrales.items():
            # Solo dentro del bloque umbrales: líneas "      <clase>: <número>"
            texto, n = re.subn(rf"(\n {{6}}{nombre}: )[0-9.]+", rf"\g<1>{u:.2f}", texto)
            if n != 1:
                raise SystemExit(f"No se encontró el umbral de '{nombre}' en config.yaml")
        ruta.write_text(texto, encoding="utf-8")
        print("\nUmbrales escritos en config.yaml (filtros.class_threshold.umbrales).")


if __name__ == "__main__":
    main()
