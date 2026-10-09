"""Pruebas unitarias de los filtros con detecciones simuladas."""
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.filters import (AreaAspectFilter, ClassThresholdFilter, PersistenceFilter,  # noqa: E402
                         UnknownRejectFilter, aplicar_filtros, construir_filtros, iou)
from src.types import Detection, Estado  # noqa: E402

CLASES = ["taza", "botella", "tijeras", "destornillador", "celular"]
SHAPE = (480, 640, 3)


def det(clase="taza", conf=0.9, bbox=(100, 100, 200, 220), tid=None, estado=Estado.ACEPTADA):
    return Detection(clase=clase, confianza=conf, bbox=bbox, track_id=tid, estado=estado)


def test_unknown_nunca_recibe_clase_oficial():
    f = UnknownRejectFilter(CLASES)
    out = f.aplicar([det("perro"), det("taza")], SHAPE)
    assert out[0].clase == "desconocido" and out[0].estado == Estado.DESCONOCIDO
    assert out[1].estado == Estado.ACEPTADA


def test_unknown_puede_ocultarse():
    f = UnknownRejectFilter(CLASES, mostrar_desconocidos=False)
    assert len(f.aplicar([det("perro"), det("taza")], SHAPE)) == 1


def test_umbral_por_clase_y_global():
    f = ClassThresholdFilter({"tijeras": 0.6}, umbral_global=0.4)
    out = f.aplicar([det("tijeras", 0.55), det("taza", 0.45), det("taza", 0.3)], SHAPE)
    assert [d.estado for d in out] == [Estado.RECHAZADA, Estado.ACEPTADA, Estado.RECHAZADA]
    assert out[0].rechazada_por == "ClassThresholdFilter"


def test_umbral_global_vivo_domina():
    f = ClassThresholdFilter({"taza": 0.3}, umbral_global=0.3)
    f.umbral_global = 0.8  # como si se moviera el trackbar
    assert f.aplicar([det("taza", 0.7)], SHAPE)[0].estado == Estado.RECHAZADA


def test_area_aspecto():
    f = AreaAspectFilter(area_min=0.01, area_max=0.9, aspecto_min=0.2, aspecto_max=5)
    diminuta = det(bbox=(0, 0, 5, 5))
    gigante = det(bbox=(0, 0, 640, 480))
    alargada = det(bbox=(0, 0, 600, 20))
    normal = det(bbox=(100, 100, 200, 220))
    out = f.aplicar([diminuta, gigante, alargada, normal], SHAPE)
    assert [d.estado for d in out] == [Estado.RECHAZADA] * 3 + [Estado.ACEPTADA]


def test_persistencia_con_track_id():
    f = PersistenceFilter(n=3, m=5)
    estados = [f.aplicar([det(tid=7)], SHAPE)[0].estado for _ in range(4)]
    assert estados == [Estado.RECHAZADA, Estado.RECHAZADA, Estado.ACEPTADA, Estado.ACEPTADA]


def test_persistencia_por_iou_sin_track_id():
    f = PersistenceFilter(n=2, m=3, iou_match=0.4)
    f.aplicar([det(bbox=(100, 100, 200, 200))], SHAPE)
    d = f.aplicar([det(bbox=(105, 102, 205, 203))], SHAPE)[0]
    assert d.estado == Estado.ACEPTADA


def test_persistencia_parpadeo_se_rechaza():
    f = PersistenceFilter(n=3, m=5)
    f.aplicar([det(tid=1)], SHAPE)
    for _ in range(4):
        f.aplicar([], SHAPE)
    assert f.aplicar([det(tid=1)], SHAPE)[0].estado == Estado.RECHAZADA


def test_varios_objetos_simultaneos():
    f = PersistenceFilter(n=2, m=3)
    escena = lambda: [det("taza", tid=1), det("botella", tid=2, bbox=(300, 100, 360, 300))]  # noqa: E731
    f.aplicar(escena(), SHAPE)
    out = f.aplicar(escena(), SHAPE)
    assert all(d.estado == Estado.ACEPTADA for d in out)


def test_rechazadas_no_se_reprocesan():
    d = det("taza", 0.1)
    filtros = [ClassThresholdFilter({}, 0.5), AreaAspectFilter(area_min=0.5)]
    aplicar_filtros(filtros, [d], SHAPE)
    assert d.rechazada_por == "ClassThresholdFilter"


def test_iou():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_construir_filtros_respeta_config():
    with open(Path(__file__).resolve().parent.parent / "config.yaml", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    assert len(construir_filtros(cfg)) == 4
    cfg["filtros"]["persistence"]["activo"] = False
    cfg["filtros"]["area_aspect"]["activo"] = False
    nombres = [f.nombre for f in construir_filtros(cfg)]
    assert nombres == ["UnknownRejectFilter", "ClassThresholdFilter"]
