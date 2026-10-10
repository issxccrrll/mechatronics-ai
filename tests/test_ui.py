"""Pruebas unitarias de las funciones puras de interfaz de usuario (UI)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.types import Detection, Estado
from src.ui import (ALTO_BARRA, color_clase, color_de_caja, color_insignia_fps,
                    contar_por_clase, dibujar_contador, geom_etiqueta, texto_etiqueta)

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


def test_color_clase_paleta_distinta():
    colores = [color_clase(c) for c in CLASES]
    assert len(set(colores)) == 5
    assert color_clase("desconocido") != color_clase("taza")


def test_color_de_caja_estado():
    ok = Detection("taza", 0.9, (10, 10, 50, 50), estado=Estado.ACEPTADA)
    rech = Detection("taza", 0.3, (10, 10, 50, 50), estado=Estado.RECHAZADA, rechazada_por="Filtro")
    desc = Detection("desconocido", 0.8, (10, 10, 50, 50), estado=Estado.DESCONOCIDO)
    assert color_de_caja(ok) == color_clase("taza")
    assert color_de_caja(rech) == (50, 50, 230)
    assert color_de_caja(desc) == (155, 155, 155)


def test_color_insignia_fps():
    assert color_insignia_fps(27.4) == (60, 200, 70)
    assert color_insignia_fps(20.0) == (60, 200, 70)
    assert color_insignia_fps(17.0) == (0, 165, 255)
    assert color_insignia_fps(15.0) == (0, 165, 255)
    assert color_insignia_fps(14.9) == (50, 50, 230)


def test_geom_etiqueta_caja_en_borde_superior_queda_dentro():
    bbox = (40, 10, 180, 170)
    shape = (480, 640, 3)
    x1, y1, x2, y2 = geom_etiqueta(bbox, "taza #3 0.91", shape)
    assert y1 >= 0
    assert y2 <= 480
    assert x1 >= 0
    assert x2 <= 640
    # No cabe arriba (10 px); la pildora empieza en el borde de la caja o dentro.
    assert y1 >= bbox[1]


def test_geom_etiqueta_con_espacio_arriba_va_encima():
    bbox = (220, 90, 300, 330)
    x1, y1, x2, y2 = geom_etiqueta(bbox, "botella #7 0.77", (480, 640, 3))
    assert y2 <= bbox[1]
    assert y1 >= 0


def test_geom_etiqueta_no_queda_bajo_la_barra_superior():
    # Caja que empieza dentro de la barra (y=15 < ALTO_BARRA): la etiqueta debe quedar visible debajo de la barra
    bbox = (100, 15, 260, 200)
    _, y1, _, y2 = geom_etiqueta(bbox, "celular #12 0.88", (480, 640, 3))
    assert y1 >= ALTO_BARRA
    assert y2 <= 480
