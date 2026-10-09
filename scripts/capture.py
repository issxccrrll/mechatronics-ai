"""Captura de fotos por clase para el dataset.

Uso: python scripts/capture.py --clase taza [--source 0]
     --clase mixto  -> escenas con VARIAS clases del reto a la vez
     --clase fondo  -> escenas SIN ninguna clase del reto (negativos)
     --ejemplar tijeras-verdes -> marca el objeto físico en el nombre del archivo
       (s20261009_101500_e-tijeras-verdes_0000.jpg) para poder reservarlo como test no visto
Teclas: espacio = guardar foto, q = salir.
Cada ejecución es una "sesión"; el nombre del archivo la incluye para dividir por sesión.
"""
import argparse
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
from src.config import parse_source  # noqa: E402

CLASES = ["taza", "botella", "tijeras", "destornillador", "celular"]
EXTRA = ["mixto", "fondo"]


def abrir_camara(source, ancho, alto):
    if isinstance(source, int) and platform.system() == "Windows":
        cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, ancho)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, alto)
    return cap


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--clase", required=True, choices=CLASES + EXTRA)
    p.add_argument("--ejemplar", default="",
                   help="nombre del objeto físico (ej. tijeras-verdes); permite reservarlo para test")
    p.add_argument("--source", default="0")
    p.add_argument("--ancho", type=int, default=640, help="misma resolución que el detector")
    p.add_argument("--alto", type=int, default=480)
    p.add_argument("--salida", default=str(RAIZ / "dataset" / "raw"))
    args = p.parse_args()

    carpeta = Path(args.salida) / args.clase
    carpeta.mkdir(parents=True, exist_ok=True)
    sesion = datetime.now().strftime("s%Y%m%d_%H%M%S")
    if args.ejemplar:
        sesion += "_e-" + args.ejemplar.strip().lower().replace(" ", "-").replace("_", "-")
    cap = abrir_camara(parse_source(args.source), args.ancho, args.alto)
    if not cap.isOpened():
        sys.exit(f"No se pudo abrir la cámara {args.source}")

    previas = len(list(carpeta.glob("*.jpg")))
    n, flash = 0, 0.0
    print(f"Sesión {sesion} -> {carpeta} ({previas} fotos previas). Espacio guarda, q sale.")
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Se perdió la señal de la cámara.")
            break
        if frame.shape[1] != args.ancho or frame.shape[0] != args.alto:
            frame = cv2.resize(frame, (args.ancho, args.alto))
        vista = frame.copy()
        cv2.rectangle(vista, (0, 0), (vista.shape[1], 30), (0, 0, 0), -1)
        cv2.putText(vista, f"{args.clase} | sesion: {n} fotos | total clase: {previas + n} | espacio=foto q=salir",
                    (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        if time.time() - flash < 0.15:  # destello visual al guardar
            cv2.rectangle(vista, (0, 0), (vista.shape[1] - 1, vista.shape[0] - 1), (255, 255, 255), 8)
        cv2.imshow("Captura", vista)
        tecla = cv2.waitKey(1) & 0xFF
        if tecla == ord(" "):
            cv2.imwrite(str(carpeta / f"{sesion}_{n:04d}.jpg"), frame)  # se guarda sin el texto
            n, flash = n + 1, time.time()
        elif tecla == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()
    print(f"Guardadas {n} fotos en esta sesión ({previas + n} en total para '{args.clase}').")


if __name__ == "__main__":
    main()
