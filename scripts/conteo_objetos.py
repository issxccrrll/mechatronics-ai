"""Cuenta cuántos objetos DISTINTOS pasaron frente a la cámara, a partir del CSV del logger.

Pensado para una línea de producción: cada fila del CSV es una detección en un cuadro, así que
no se cuentan filas sino IDs de seguimiento (ByteTrack) únicos. Un ID visto en menos de
--min-cuadros cuadros se descarta (destellos o cambios de ID por oclusión).

Ej.: python scripts/conteo_objetos.py                      # último CSV de logs/
     python scripts/conteo_objetos.py logs/detecciones_20261010_103000.csv --min-cuadros 10
"""
import argparse
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent


def contar(df: pd.DataFrame, min_cuadros: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(conteo por clase, detalle por objeto) usando solo detecciones aceptadas con track_id."""
    ok = df[(df["estado"] == "aceptada") & df["track_id"].notna()].copy()
    ok["track_id"] = ok["track_id"].astype(int)
    objetos = ok.groupby("track_id").agg(
        clase=("clase", lambda s: s.mode().iat[0]),   # clase más frecuente del objeto
        cuadros=("clase", "size"),
        confianza_media=("confianza", "mean"),
        primera_vez=("timestamp", "min"),
        ultima_vez=("timestamp", "max"),
    ).reset_index()
    objetos = objetos[objetos["cuadros"] >= min_cuadros].sort_values("primera_vez")
    conteo = objetos.groupby("clase").size().rename("objetos").reset_index()
    return conteo, objetos


def main():
    p = argparse.ArgumentParser()
    p.add_argument("csv", nargs="?", help="CSV del logger (por defecto el más reciente de logs/)")
    p.add_argument("--min-cuadros", type=int, default=5, help="cuadros mínimos para contar un objeto")
    p.add_argument("--guardar", action="store_true", help="escribe el resumen junto al CSV")
    args = p.parse_args()

    if args.csv:
        ruta = Path(args.csv)
    else:
        candidatos = sorted((RAIZ / "logs").glob("detecciones_*.csv"))
        if not candidatos:
            raise SystemExit("No hay CSV en logs/. Ejecuta main.py primero.")
        ruta = candidatos[-1]

    conteo, objetos = contar(pd.read_csv(ruta), args.min_cuadros)
    print(f"Archivo: {ruta.name}  (objetos vistos en al menos {args.min_cuadros} cuadros)\n")
    print(conteo.to_string(index=False) if len(conteo) else "Ningún objeto aceptado.")
    print(f"\nTotal: {int(conteo['objetos'].sum()) if len(conteo) else 0} objetos")
    if args.guardar:
        destino = ruta.with_name(ruta.stem + "_conteo.csv")
        objetos.to_csv(destino, index=False, encoding="utf-8-sig")
        print(f"Detalle por objeto guardado en {destino}")


if __name__ == "__main__":
    main()
