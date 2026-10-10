"""Pruebas para el script de ensayo_escenas.py."""
import sys
from pathlib import Path
import pytest

RAIZ = Path(__file__).resolve().parent.parent
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from scripts.ensayo_escenas import conteo_aceptadas, moda_conteos, es_acierto, resumen, ESCENAS
from src.types import Detection, Estado

def test_conteo_aceptadas():
    clases = ["taza", "botella"]
    detecciones = [
        Detection(clase="taza", confianza=0.9, bbox=[0,0,10,10], estado=Estado.ACEPTADA),
        Detection(clase="taza", confianza=0.8, bbox=[0,0,10,10], estado=Estado.ACEPTADA),
        Detection(clase="botella", confianza=0.7, bbox=[0,0,10,10], estado=Estado.RECHAZADA),
        Detection(clase="botella", confianza=0.6, bbox=[0,0,10,10], estado=Estado.DESCONOCIDO),
        Detection(clase="tijeras", confianza=0.9, bbox=[0,0,10,10], estado=Estado.ACEPTADA) # no oficial en esta llamada
    ]
    conteo = conteo_aceptadas(detecciones, clases)
    assert conteo == {"taza": 2}

def test_moda_conteos():
    conteos = [
        {"taza": 1, "botella": 0},
        {"taza": 1},
        {"botella": 1},
        {"taza": 1},
        {}
    ]
    # taza:1 aparece 3 veces (la de botella:0 se ignora el 0 internamente)
    assert moda_conteos(conteos) == {"taza": 1}
    
    # Mayoría sin nada
    conteos_vacios = [{}, {}, {"taza": 1}]
    assert moda_conteos(conteos_vacios) == {}

def test_es_acierto():
    assert es_acierto({"taza": 1}, {"taza": 1}) == True
    assert es_acierto({"taza": 1}, {"taza": 1, "botella": 1}) == False
    assert es_acierto({"taza": 1}, {"taza": 2}) == False
    assert es_acierto({}, {}) == True
    assert es_acierto({}, {"taza": 1}) == False
    assert es_acierto({"taza": 1}, {}) == False
    
    # Ignorar ceros
    assert es_acierto({"taza": 1, "botella": 0}, {"taza": 1}) == True
    assert es_acierto({"taza": 1}, {"taza": 1, "botella": 0}) == True

def test_resumen():
    assert resumen([]) == (0, 0, 0.0)
    
    resultados = [{"acierto": True}] * 17 + [{"acierto": False}] * 3
    aciertos, evaluadas, pct = resumen(resultados)
    assert aciertos == 17
    assert evaluadas == 20
    assert pct == 85.0

def test_escenas_validas():
    assert len(ESCENAS) == 20
    clases_oficiales = {"taza", "botella", "tijeras", "destornillador", "celular"}
    for esc in ESCENAS:
        for clase in esc["esperado"]:
            assert clase in clases_oficiales, f"Clase no oficial en escena: {clase}"
