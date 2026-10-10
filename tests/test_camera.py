"""Pruebas unitarias de sincronización y ciclo de vida de Camara."""
import sys
import threading
import time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.camera import Camara


def test_camara_leer_timeout_sin_frame():
    # Instanciamos cámara con source inválido para no abrir hardware
    cam = Camara(source=-999, espera_reintento_s=10.0)
    t0 = time.perf_counter()
    frame = cam.leer(timeout=0.05)
    duracion = time.perf_counter() - t0
    cam.cerrar()

    assert frame is None
    # No debe hacer busy-waiting instantáneo ni tardar de más
    assert duracion < 0.2


def test_camara_entrega_frame_sincronizado():
    cam = Camara(source=-999, espera_reintento_s=10.0)
    # Simulamos estado ok y la llegada de un frame desde el hilo
    mock_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    def productor():
        time.sleep(0.02)
        with cam._lock:
            cam.estado = "ok"
            cam._frame = mock_frame
            cam._nuevo = True
            cam._cond.notify_all()

    h = threading.Thread(target=productor)
    h.start()

    # leer debe esperar a que el frame llegue mediante Condition
    frame_obtenido = cam.leer(timeout=0.2)
    h.join()

    assert frame_obtenido is not None
    assert np.array_equal(frame_obtenido, mock_frame)

    # El siguiente leer inmediato debe devolver None (solo se entrega cada frame una vez)
    frame_segundo = cam.leer(timeout=0.01)
    assert frame_segundo is None

    cam.cerrar()


def test_camara_cerrar_despierta_hilos_en_espera():
    cam = Camara(source=-999, espera_reintento_s=10.0)
    with cam._lock:
        cam.estado = "ok"

    t0 = time.perf_counter()

    def cerrador():
        time.sleep(0.02)
        cam.cerrar()

    h = threading.Thread(target=cerrador)
    h.start()

    # leer está esperando con timeout de 1s, pero cam.cerrar() debe despertarlo en ~0.02s
    frame = cam.leer(timeout=1.0)
    duracion = time.perf_counter() - t0
    h.join()

    assert frame is None
    assert duracion < 0.2


def test_candidatos_webcam_prueba_otros_indices():
    cam = Camara.__new__(Camara)  # sin iniciar el hilo ni abrir hardware
    cam.source = 0
    assert cam.candidatos() == [0, 1, 2, 3]
    cam.source = 2
    assert cam.candidatos() == [2, 0, 1, 3]


def test_candidatos_url_o_fuera_de_rango_no_cambia():
    cam = Camara.__new__(Camara)
    cam.source = "http://192.168.1.50:8080/video"
    assert cam.candidatos() == ["http://192.168.1.50:8080/video"]
    cam.source = -999
    assert cam.candidatos() == [-999]
