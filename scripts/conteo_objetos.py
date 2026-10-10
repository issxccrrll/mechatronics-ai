"""Cuenta cuántos objetos DISTINTOS pasaron frente a la cámara, a partir del CSV del logger.

main.py ya genera este resumen al cerrar (logs/<sesión>_conteo.csv). Este script sirve para
recalcularlo sobre cualquier sesión o con otro mínimo de cuadros.

Ej.: python scripts/conteo_objetos.py                      # último CSV de logs/
     python scripts/conteo_objetos.py logs/detecciones_20261010_103000.csv --min-cuadros 10
"""
import argparse
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
from src.conteo import contar, resumen_sesion  # noqa: E402,F401  (contar se reexporta para las pruebas)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("csv", nargs="?", help="CSV del logger (por defecto el más reciente de logs/)")
    p.add_argument("--min-cuadros", type=int, default=25, help="cuadros mínimos para contar un objeto")
    args = p.parse_args()

    if args.csv:
        ruta = Path(args.csv)
    else:
        candidatos = sorted(f for f in (RAIZ / "logs").glob("detecciones_*.csv") if not f.stem.endswith("_conteo"))
        if not candidatos:
            raise SystemExit("No hay CSV en logs/. Ejecuta main.py primero.")
        ruta = candidatos[-1]
    print(f"Archivo: {ruta.name}")
    resumen_sesion(ruta, args.min_cuadros)


if __name__ == "__main__":
    main()
