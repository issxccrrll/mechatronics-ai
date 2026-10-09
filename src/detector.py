"""Detectores de objetos. YoloDetector acepta .pt o carpeta OpenVINO."""
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from .types import Detection, Estado

DESCONOCIDO = "desconocido"


class Detector(ABC):
    @abstractmethod
    def detectar(self, frame) -> list[Detection]:
        ...


class YoloDetector(Detector):
    def __init__(self, cfg_modelo: dict, mapeo_clases: dict, clases_oficiales: list[str]):
        from ultralytics import YOLO

        self.cfg = cfg_modelo
        self.mapeo = mapeo_clases or {}
        self.oficiales = set(clases_oficiales)
        ruta = Path(cfg_modelo["ruta"])
        if ruta.name.endswith("_openvino_model") and not ruta.exists():
            # Respaldo: si aún no se exportó a OpenVINO, usar el .pt equivalente
            alterno = ruta.with_name(ruta.name.removesuffix("_openvino_model") + ".pt")
            print(f"[detector] No existe {ruta}; usando {alterno}. Exporta con scripts/export.py")
            ruta = alterno
        self.modelo = YOLO(str(ruta), task="detect")
        # Umbral bajo en el modelo: el filtrado real lo hacen los filtros con el umbral vivo
        self.conf_modelo = min(0.25, cfg_modelo["conf"])
        self._calentar()

    def _calentar(self):
        """La primera inferencia (compilación OpenVINO) tarda segundos: hacerla antes de abrir la UI."""
        print("[detector] Calentando modelo...")
        # Mismo camino que detectar(): track() crea su propio predictor, predict() no lo calentaría
        self.detectar(np.zeros((480, 640, 3), dtype=np.uint8))

    def _mapear(self, nombre: str) -> str:
        clase = self.mapeo.get(nombre, DESCONOCIDO)
        return clase if clase in self.oficiales else DESCONOCIDO

    def detectar(self, frame) -> list[Detection]:
        kwargs = dict(imgsz=self.cfg["imgsz"], conf=self.conf_modelo, iou=self.cfg["iou"],
                      device=self.cfg["device"], verbose=False)
        if self.cfg.get("tracking", True):
            res = self.modelo.track(frame, persist=True, tracker=self.cfg["tracker"], **kwargs)[0]
        else:
            res = self.modelo.predict(frame, **kwargs)[0]

        cajas = res.boxes
        if cajas is None or len(cajas) == 0:
            return []
        ids = cajas.id.int().tolist() if cajas.id is not None else [None] * len(cajas)
        salida = []
        for xyxy, conf, cls, tid in zip(cajas.xyxy.tolist(), cajas.conf.tolist(),
                                        cajas.cls.int().tolist(), ids):
            original = res.names[cls]
            clase = self._mapear(original)
            salida.append(Detection(
                clase=clase, confianza=float(conf), bbox=tuple(int(v) for v in xyxy),
                track_id=tid, clase_original=original,
                estado=Estado.DESCONOCIDO if clase == DESCONOCIDO else Estado.ACEPTADA,
            ))
        return salida
