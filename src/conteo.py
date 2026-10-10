"""Conteo de objetos distintos a partir del CSV del logger (IDs de seguimiento únicos).

Cada fila del CSV es una detección en un cuadro, así que no se cuentan filas sino track_id
únicos de detecciones aceptadas. Un ID visto en menos de `min_cuadros` cuadros se descarta
(destellos o cambios de ID por oclusión).
"""
from pathlib import Path

import pandas as pd


def contar(df: pd.DataFrame, min_cuadros: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(conteo por clase, detalle por objeto)."""
    ok = df[(df["estado"] == "aceptada") & df["track_id"].notna()].copy()
    if ok.empty:
        return pd.DataFrame(columns=["clase", "objetos"]), pd.DataFrame()
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


def resumen_sesion(ruta_csv: Path, min_cuadros: int = 25) -> Path | None:
    """Cuenta los objetos de una sesión, guarda <csv>_conteo.csv e imprime el resumen."""
    conteo, objetos = contar(pd.read_csv(ruta_csv), min_cuadros)
    total = int(conteo["objetos"].sum()) if len(conteo) else 0
    print(f"\n[conteo] Objetos distintos en la sesión (vistos al menos {min_cuadros} cuadros): {total}")
    for fila in conteo.itertuples(index=False):
        print(f"  {fila.clase:15s} {fila.objetos}")
    destino = ruta_csv.with_name(ruta_csv.stem + "_conteo.csv")
    objetos.to_csv(destino, index=False, encoding="utf-8-sig")
    print(f"[conteo] Detalle guardado en {destino}")
    return destino
