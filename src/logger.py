"""Registro CSV de detecciones."""
import csv
from datetime import datetime
from pathlib import Path

from .types import Detection, Estado

COLUMNAS = ["timestamp", "clase", "confianza", "estado", "rechazada_por", "track_id", "clase_original"]


class DetectionLogger:
    def __init__(self, carpeta="logs", incluir_rechazadas=True, activo=True):
        self.activo = activo
        self.incluir_rechazadas = incluir_rechazadas
        self._f = None
        if not activo:
            return
        Path(carpeta).mkdir(parents=True, exist_ok=True)
        ruta = Path(carpeta) / f"detecciones_{datetime.now():%Y%m%d_%H%M%S}.csv"
        self._f = open(ruta, "w", newline="", encoding="utf-8")
        self._w = csv.writer(self._f)
        self._w.writerow(COLUMNAS)
        self.ruta = ruta

    def registrar(self, detecciones: list[Detection]):
        if not self.activo:
            return
        ts = datetime.now().isoformat(timespec="milliseconds")
        for d in detecciones:
            if d.estado == Estado.RECHAZADA and not self.incluir_rechazadas:
                continue
            self._w.writerow([ts, d.clase, f"{d.confianza:.3f}", d.estado.value,
                              d.rechazada_por or "", d.track_id if d.track_id is not None else "",
                              d.clase_original])

    def cerrar(self):
        if self._f:
            self._f.close()
