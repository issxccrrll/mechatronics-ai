"""Captura de fotos por clase para el dataset.

Uso: python scripts/capture.py --clase taza [--source 0]
Teclas: espacio = guardar foto, q = salir.
Cada ejecución es una "sesión"; el nombre del archivo la incluye para dividir por sesión.
"""
import argparse
import sys
from datetime import datetime
from pathlib import Path

import cv2

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
from src.config import parse_source  # noqa: E402

CLASES = ["taza", "botella", "tijeras", "destornillador", "celular"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--clase", required=True, choices=CLASES + ["fondo"])
    p.add_argument("--source", default="0")
    p.add_argument("--salida", default=str(RAIZ / "dataset" / "raw"))
    args = p.parse_args()

    carpeta = Path(args.salida) / args.clase
    carpeta.mkdir(parents=True, exist_ok=True)
    sesion = datetime.now().strftime("s%Y%m%d_%H%M%S")
    cap = cv2.VideoCapture(parse_source(args.source))
    if not cap.isOpened():
        sys.exit(f"No se pudo abrir la cámara {args.source}")

    n = 0
    print(f"Sesión {sesion} -> {carpeta}. Espacio guarda, q sale.")
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Se perdió la señal de la cámara.")
            break
        vista = frame.copy()
        cv2.putText(vista, f"{args.clase} | {sesion} | fotos: {n}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow("Captura", vista)
        tecla = cv2.waitKey(1) & 0xFF
        if tecla == ord(" "):
            cv2.imwrite(str(carpeta / f"{sesion}_{n:04d}.jpg"), frame)
            n += 1
        elif tecla == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()
    print(f"Guardadas {n} fotos.")


if __name__ == "__main__":
    main()
