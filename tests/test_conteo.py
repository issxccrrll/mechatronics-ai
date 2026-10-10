"""Pruebas del conteo de objetos únicos a partir del CSV del logger."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from conteo_objetos import contar  # noqa: E402


def filas(track_id, clase, n, estado="aceptada"):
    return [{"timestamp": f"t{i:03d}", "clase": clase, "confianza": 0.8, "estado": estado,
             "track_id": track_id} for i in range(n)]


def test_cuenta_ids_unicos_no_filas():
    df = pd.DataFrame(filas(1, "taza", 75) + filas(2, "taza", 40) + filas(3, "botella", 30))
    conteo, _ = contar(df, min_cuadros=5)
    assert dict(zip(conteo["clase"], conteo["objetos"])) == {"botella": 1, "taza": 2}


def test_descarta_ids_con_pocos_cuadros():
    df = pd.DataFrame(filas(1, "celular", 20) + filas(9, "celular", 2))
    conteo, _ = contar(df, min_cuadros=5)
    assert dict(zip(conteo["clase"], conteo["objetos"])) == {"celular": 1}


def test_ignora_rechazadas_y_sin_id():
    df = pd.DataFrame(filas(1, "tijeras", 10, estado="rechazada") + filas(None, "tijeras", 10))
    conteo, _ = contar(df, min_cuadros=5)
    assert conteo.empty


def test_clase_mayoritaria_por_objeto():
    df = pd.DataFrame(filas(4, "destornillador", 12) + filas(4, "tijeras", 3))
    _, objetos = contar(df, min_cuadros=5)
    assert objetos.iloc[0]["clase"] == "destornillador"
