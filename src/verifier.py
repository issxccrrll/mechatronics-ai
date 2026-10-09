"""Verificador opcional: segunda opinión de un clasificador CNN propio (bonus).

Desactivado por defecto. Recibe el recorte de cada detección aceptada y puede
rechazarla si el clasificador no coincide con la clase del detector.
"""
from abc import ABC, abstractmethod

from .types import Detection, Estado


class Verifier(ABC):
    @abstractmethod
    def verificar(self, frame, detecciones: list[Detection]) -> list[Detection]:
        ...


class NullVerifier(Verifier):
    """No hace nada; se usa cuando el verificador está desactivado."""

    def verificar(self, frame, detecciones):
        return detecciones


class CnnVerifier(Verifier):
    """TODO (bonus): CNN pequeña entrenada desde cero sobre recortes de dataset/raw/<clase>/."""

    def __init__(self, ruta_modelo: str, umbral: float = 0.5):
        self.ruta = ruta_modelo
        self.umbral = umbral
        self.modelo = None  # TODO: cargar torch.load(ruta) y modo eval

    def clasificar(self, recorte) -> tuple[str, float]:
        raise NotImplementedError("TODO: preprocesar recorte (ej. 96x96) y ejecutar la CNN")

    def verificar(self, frame, detecciones):
        for d in detecciones:
            if d.estado != Estado.ACEPTADA:
                continue
            x1, y1, x2, y2 = d.bbox
            clase, prob = self.clasificar(frame[y1:y2, x1:x2])
            if clase != d.clase or prob < self.umbral:
                d.rechazar("CnnVerifier")
        return detecciones


def construir_verificador(cfg: dict) -> Verifier:
    if not cfg.get("activo"):
        return NullVerifier()
    return CnnVerifier(cfg["ruta"], cfg.get("umbral", 0.5))
