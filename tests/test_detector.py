"""Pruebas del mapeo de clases del detector (sin cargar ningún modelo)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.detector import YoloDetector  # noqa: E402

CLASES = ["taza", "botella", "tijeras", "destornillador", "celular"]
MAPEO_COCO = {"cup": "taza", "bottle": "botella", "scissors": "tijeras", "cell phone": "celular"}


def detector_sin_modelo(mapeo):
    d = YoloDetector.__new__(YoloDetector)  # evita __init__ (no carga YOLO)
    d.mapeo, d.oficiales = mapeo, set(CLASES)
    return d


def test_modelo_coco_se_traduce():
    d = detector_sin_modelo(MAPEO_COCO)
    assert d._mapear("cup") == "taza"
    assert d._mapear("cell phone") == "celular"


def test_clase_coco_no_mapeada_es_desconocido():
    d = detector_sin_modelo(MAPEO_COCO)
    assert d._mapear("person") == "desconocido"
    assert d._mapear("knife") == "desconocido"


def test_modelo_propio_pasa_directo_aunque_haya_mapeo_coco():
    d = detector_sin_modelo(MAPEO_COCO)
    assert d._mapear("destornillador") == "destornillador"
    assert d._mapear("taza") == "taza"


def test_mapeo_hacia_clase_no_oficial_es_desconocido():
    d = detector_sin_modelo({"dog": "perro"})
    assert d._mapear("dog") == "desconocido"
