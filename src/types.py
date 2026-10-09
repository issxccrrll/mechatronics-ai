"""Tipos compartidos por todo el pipeline."""
from dataclasses import dataclass
from enum import Enum


class Estado(str, Enum):
    ACEPTADA = "aceptada"
    RECHAZADA = "rechazada"
    DESCONOCIDO = "desconocido"


@dataclass
class Detection:
    clase: str                     # clase oficial o "desconocido"
    confianza: float
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2 en píxeles
    track_id: int | None = None
    estado: Estado = Estado.ACEPTADA
    rechazada_por: str | None = None   # nombre del filtro que la rechazó
    clase_original: str = ""           # nombre que dio el modelo (útil para depurar)

    @property
    def area(self) -> int:
        x1, y1, x2, y2 = self.bbox
        return max(0, x2 - x1) * max(0, y2 - y1)

    def rechazar(self, filtro: str) -> None:
        self.estado = Estado.RECHAZADA
        self.rechazada_por = filtro
