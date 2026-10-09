"""Filtros anti falsos positivos (patrón Strategy).

Cada filtro recibe la lista completa de detecciones y marca como RECHAZADA las
que no pasan. Solo evalúan detecciones ACEPTADAS (no re-procesan rechazadas).
"""
from abc import ABC, abstractmethod
from collections import deque

from .types import Detection, Estado


class Filtro(ABC):
    nombre = "Filtro"

    @abstractmethod
    def aplicar(self, detecciones: list[Detection], frame_shape) -> list[Detection]:
        ...


class UnknownRejectFilter(Filtro):
    """Asegura que nada fuera de las clases oficiales se reporte como oficial."""
    nombre = "UnknownRejectFilter"

    def __init__(self, clases_oficiales, mostrar_desconocidos=True):
        self.oficiales = set(clases_oficiales)
        self.mostrar = mostrar_desconocidos

    def aplicar(self, detecciones, frame_shape):
        salida = []
        for d in detecciones:
            if d.clase not in self.oficiales:
                d.clase, d.estado = "desconocido", Estado.DESCONOCIDO
                if not self.mostrar:
                    continue
            salida.append(d)
        return salida


class ClassThresholdFilter(Filtro):
    """Umbral por clase; nunca por debajo del umbral global (ajustable en vivo)."""
    nombre = "ClassThresholdFilter"

    def __init__(self, umbrales: dict, umbral_global: float):
        self.umbrales = umbrales or {}
        self.umbral_global = umbral_global

    def aplicar(self, detecciones, frame_shape):
        for d in detecciones:
            if d.estado != Estado.ACEPTADA:
                continue
            umbral = max(self.umbral_global, self.umbrales.get(d.clase, 0.0))
            if d.confianza < umbral:
                d.rechazar(self.nombre)
        return detecciones


class AreaAspectFilter(Filtro):
    """Rechaza cajas diminutas, gigantes o con proporciones imposibles."""
    nombre = "AreaAspectFilter"

    def __init__(self, area_min=0.002, area_max=0.9, aspecto_min=0.15, aspecto_max=6.0):
        self.area_min, self.area_max = area_min, area_max
        self.aspecto_min, self.aspecto_max = aspecto_min, aspecto_max

    def aplicar(self, detecciones, frame_shape):
        area_frame = frame_shape[0] * frame_shape[1]
        for d in detecciones:
            if d.estado != Estado.ACEPTADA:
                continue
            x1, y1, x2, y2 = d.bbox
            ancho, alto = x2 - x1, y2 - y1
            frac = d.area / area_frame if area_frame else 0
            aspecto = ancho / alto if alto > 0 else 0
            if not (self.area_min <= frac <= self.area_max
                    and self.aspecto_min <= aspecto <= self.aspecto_max):
                d.rechazar(self.nombre)
        return detecciones


def iou(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


class PersistenceFilter(Filtro):
    """Acepta un objeto solo si aparece en al menos N de los últimos M frames.

    Usa track_id si existe; si no, empareja con las pistas previas por IoU y clase.
    """
    nombre = "PersistenceFilter"

    def __init__(self, n=3, m=5, iou_match=0.4):
        self.n, self.m, self.iou_match = n, m, iou_match
        self._pistas: dict = {}   # clave -> {"hist": deque, "bbox": bbox, "clase": str}
        self._siguiente = 0

    def _clave(self, d: Detection, usadas: set):
        if d.track_id is not None:
            return ("t", d.track_id)
        mejor, mejor_iou = None, self.iou_match
        for clave, p in self._pistas.items():
            if clave in usadas or clave[0] != "i" or p["clase"] != d.clase:
                continue
            v = iou(p["bbox"], d.bbox)
            if v >= mejor_iou:
                mejor, mejor_iou = clave, v
        if mejor is None:
            mejor = ("i", self._siguiente)
            self._siguiente += 1
        return mejor

    def aplicar(self, detecciones, frame_shape):
        vistas = set()
        for d in detecciones:
            if d.estado != Estado.ACEPTADA:
                continue
            clave = self._clave(d, vistas)
            vistas.add(clave)
            p = self._pistas.setdefault(clave, {"hist": deque(maxlen=self.m)})
            p["hist"].append(1)
            p["bbox"], p["clase"] = d.bbox, d.clase
            if sum(p["hist"]) < self.n:
                d.rechazar(self.nombre)
        # Pistas no vistas en este frame suman un 0; se eliminan si quedan vacías
        for clave in list(self._pistas):
            if clave not in vistas:
                hist = self._pistas[clave]["hist"]
                hist.append(0)
                if sum(hist) == 0:
                    del self._pistas[clave]
        return detecciones


def construir_filtros(cfg: dict) -> list[Filtro]:
    """Crea la lista de filtros activos según config.yaml (orden fijo)."""
    f = cfg["filtros"]
    filtros: list[Filtro] = []
    if f["unknown_reject"]["activo"]:
        filtros.append(UnknownRejectFilter(cfg["clases"], f["unknown_reject"].get("mostrar_desconocidos", True)))
    if f["class_threshold"]["activo"]:
        filtros.append(ClassThresholdFilter(f["class_threshold"]["umbrales"], cfg["modelo"]["conf"]))
    if f["area_aspect"]["activo"]:
        p = f["area_aspect"]
        filtros.append(AreaAspectFilter(p["area_min"], p["area_max"], p["aspecto_min"], p["aspecto_max"]))
    if f["persistence"]["activo"]:
        p = f["persistence"]
        filtros.append(PersistenceFilter(p["n"], p["m"], p["iou_match"]))
    return filtros


def aplicar_filtros(filtros: list[Filtro], detecciones, frame_shape):
    for filtro in filtros:
        detecciones = filtro.aplicar(detecciones, frame_shape)
    return detecciones
