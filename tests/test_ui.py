"""Pruebas unitarias de las funciones puras de interfaz de usuario (UI)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.types import Detection, Estado
from src.ui import contar_por_clase, dibujar_contador, texto_etiqueta

CLASES = ["taza", "botella", "tijeras", "destornillador", "celular"]


def test_contar_por_clase_incluye_todas_en_cero():
    conteo = contar_por_clase([], CLASES)
    assert conteo == {c: 0 for c in CLASES}
    assert list(conteo.keys()) == CLASES


def test_contar_por_clase_ignora_rechazadas_y_desconocidas():
    dets = [
        Detection("taza", 0.9, (10, 10, 50, 50), estado=Estado.ACEPTADA),
        Detection("botella", 0.8, (10, 10, 50, 50), estado=Estado.RECHAZADA, rechazada_por="Filtro"),
        Detection("desconocido", 0.5, (10, 10, 50, 50), estado=Estado.DESCONOCIDO),
    ]
    conteo = contar_por_clase(dets, CLASES)
    assert conteo["taza"] == 1
    assert conteo["botella"] == 0
    assert "desconocido" not in conteo


def test_contar_por_clase_cuenta_multiples():
    dets = [
        Detection("taza", 0.9, (10, 10, 50, 50), estado=Estado.ACEPTADA),
        Detection("taza", 0.85, (60, 60, 100, 100), estado=Estado.ACEPTADA),
        Detection("celular", 0.95, (110, 110, 150, 150), estado=Estado.ACEPTADA),
    ]
    conteo = contar_por_clase(dets, CLASES)
    assert conteo["taza"] == 2
    assert conteo["celular"] == 1
    assert conteo["botella"] == 0


def test_texto_etiqueta_con_track_id():
    d = Detection("taza", 0.87, (10, 10, 50, 50), track_id=3)
    txt = texto_etiqueta(d, mostrar_id=True)
    assert "#3" in txt
    assert txt.startswith("taza #3")


def test_texto_etiqueta_sin_track_id():
    d = Detection("taza", 0.87, (10, 10, 50, 50), track_id=None)
    txt = texto_etiqueta(d, mostrar_id=True)
    assert "#" not in txt
    assert txt.startswith("taza 0.87")


def test_texto_etiqueta_mostrar_id_false():
    d = Detection("taza", 0.87, (10, 10, 50, 50), track_id=7)
    txt = texto_etiqueta(d, mostrar_id=False)
    assert "#" not in txt
    assert txt.startswith("taza 0.87")


def test_texto_etiqueta_rechazada_termina_con_filtro():
    d = Detection("tijeras", 0.40, (10, 10, 50, 50), track_id=2,
                  estado=Estado.RECHAZADA, rechazada_por="AreaAspectFilter")
    txt = texto_etiqueta(d, mostrar_id=True)
    assert txt.endswith("[AreaAspectFilter]")
    assert "#2" in txt


def test_dibujar_contador_zona_correcta_sin_tocar_barra_superior():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    conteo = {c: 1 for c in CLASES}
    dibujar_contador(frame, conteo)

    # La barra superior (filas 0 a 23) no debe ser alterada
    assert np.all(frame[0:24, :] == 0)

    # La esquina inferior izquierda debe contener píxeles dibujados
    assert np.any(frame[300:480, 0:200] != 0)
