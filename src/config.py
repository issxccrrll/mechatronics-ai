"""Carga config.yaml y aplica overrides de línea de comandos."""
import argparse
from pathlib import Path

import yaml

RAIZ = Path(__file__).resolve().parent.parent


def parse_source(valor):
    """'0' -> 0 (webcam); cualquier otra cosa se deja como URL/ruta."""
    if isinstance(valor, int):
        return valor
    texto = str(valor).strip()
    return int(texto) if texto.isdigit() else texto


def cargar_config(argv=None) -> dict:
    parser = argparse.ArgumentParser(description="Detector en vivo Mechatronics AI")
    parser.add_argument("--config", default=str(RAIZ / "config.yaml"))
    parser.add_argument("--source", help="índice de webcam o URL de cámara")
    parser.add_argument("--conf", type=float, help="umbral global de confianza")
    parser.add_argument("--model", help="ruta a .pt o carpeta OpenVINO")
    parser.add_argument("--imgsz", type=int, help="tamaño de entrada del modelo")
    args = parser.parse_args(argv)

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if args.source is not None:
        cfg["camara"]["source"] = args.source
    if args.conf is not None:
        cfg["modelo"]["conf"] = args.conf
    if args.model is not None:
        cfg["modelo"]["ruta"] = args.model
    if args.imgsz is not None:
        cfg["modelo"]["imgsz"] = args.imgsz

    cfg["camara"]["source"] = parse_source(cfg["camara"]["source"])
    return cfg
