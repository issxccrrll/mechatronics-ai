"""Punto de entrada: Cámara -> Detector -> Verificador -> Filtros -> UI + Logger."""
import time

from src.camera import Camara
from src.config import cargar_config
from src.detector import YoloDetector
from src.filters import ClassThresholdFilter, PersistenceFilter, aplicar_filtros, construir_filtros
from src.logger import DetectionLogger
from src.ui import UI, MedidorFPS
from src.verifier import construir_verificador

MENSAJES = {
    "conectando": "Conectando a la camara...",
    "reconectando": "Sin senal. Reintentando conexion ({intentos})...",
    "sin_camara": "No se encontro la camara. Revisa --source",
}


def actualizar_desde_trackbars(ui: UI, filtros):
    valores = ui.leer_trackbars()
    if valores is None:
        return
    conf, n = valores
    for f in filtros:
        if isinstance(f, ClassThresholdFilter):
            f.umbral_global = conf
        elif isinstance(f, PersistenceFilter):
            f.n = n


def main():
    cfg = cargar_config()
    cam_cfg = cfg["camara"]
    print(f"[main] fuente={cam_cfg['source']!r} modelo={cfg['modelo']['ruta']}")

    detector = YoloDetector(cfg["modelo"], cfg["mapeo_clases"], cfg["clases"])
    verificador = construir_verificador(cfg["verificador"])
    filtros = construir_filtros(cfg)
    logger = DetectionLogger(cfg["logger"]["carpeta"], cfg["logger"]["incluir_rechazadas"], cfg["logger"]["activo"])
    ui = UI(cfg)
    fps = MedidorFPS()
    camara = Camara(cam_cfg["source"], cam_cfg["ancho"], cam_cfg["alto"],
                    cam_cfg["reintentos_max"], cam_cfg["espera_reintento_s"])
    frames, t_inicio = 0, None
    try:
        while True:
            frame = camara.leer()
            if frame is None:
                if camara.estado != "ok":
                    msg = MENSAJES[camara.estado].format(intentos=camara.intentos)
                    if not ui.mostrar(ui.pantalla_estado(msg, cam_cfg["ancho"], cam_cfg["alto"])):
                        break
                elif not ui.procesar_eventos():
                    break
                continue
            t_inicio = t_inicio or time.perf_counter()
            frames += 1
            actualizar_desde_trackbars(ui, filtros)
            detecciones = detector.detectar(frame)
            detecciones = verificador.verificar(frame, detecciones)
            detecciones = aplicar_filtros(filtros, detecciones, frame.shape)
            logger.registrar(detecciones)
            if not ui.mostrar(ui.dibujar(frame, detecciones, fps.tick())):
                break
    except KeyboardInterrupt:
        pass
    finally:
        camara.cerrar()
        logger.cerrar()
        ui.cerrar()
        if t_inicio and frames > 1:
            print(f"[main] {frames} frames, FPS promedio {frames / (time.perf_counter() - t_inicio):.1f}")
        print("[main] cerrado limpiamente")


if __name__ == "__main__":
    main()
