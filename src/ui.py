"""Dibujo en pantalla y trackbars de OpenCV."""
import time
from collections import deque

import cv2
import numpy as np

from .types import Detection, Estado

COLORES = {  # BGR
    Estado.ACEPTADA: (60, 200, 60),
    Estado.RECHAZADA: (60, 60, 220),
    Estado.DESCONOCIDO: (160, 160, 160),
}
FUENTE = cv2.FONT_HERSHEY_SIMPLEX


def contar_por_clase(detecciones: list[Detection], clases: list[str]) -> dict[str, int]:
    """Cuenta detecciones ACEPTADAS para cada clase oficial en el orden dado."""
    conteo = {c: 0 for c in clases}
    for d in detecciones:
        if d.estado == Estado.ACEPTADA and d.clase in conteo:
            conteo[d.clase] += 1
    return conteo


def texto_etiqueta(d: Detection, mostrar_id: bool = True) -> str:
    """Genera el texto de etiqueta para una detección: clase [#id] conf [filtro]."""
    id_part = f" #{d.track_id}" if (mostrar_id and d.track_id is not None) else ""
    etiqueta = f"{d.clase}{id_part} {d.confianza:.2f}"
    if d.estado == Estado.RECHAZADA and d.rechazada_por:
        etiqueta += f" [{d.rechazada_por}]"
    return etiqueta


def dibujar_contador(frame: np.ndarray, conteo: dict[str, int]) -> None:
    """Dibuja un panel semitransparente con el conteo de clases en la esquina inferior izquierda.

    No invade la barra superior (filas 0-23).
    """
    if not conteo:
        return
    alto_frame, ancho_frame = frame.shape[:2]
    linea_h = 18
    pad_x, pad_y = 10, 8
    ancho_panel = 175
    alto_panel = len(conteo) * linea_h + pad_y * 2

    x1 = 10
    x2 = min(x1 + ancho_panel, ancho_frame - 1)
    y2 = alto_frame - 10
    y1 = max(24, y2 - alto_panel)

    if y2 <= y1 or x2 <= x1:
        return

    # Fondo semitransparente oscuro (alpha blending localizado para máximo rendimiento)
    sub = frame[y1:y2, x1:x2]
    negro = np.zeros_like(sub)
    cv2.addWeighted(sub, 0.35, negro, 0.65, 0, sub)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (70, 70, 70), 1)

    # Texto alineado por clase
    for i, (clase, cant) in enumerate(conteo.items()):
        y_text = y1 + pad_y + (i + 1) * linea_h - 4
        color = (255, 255, 255) if cant > 0 else (140, 140, 140)
        cv2.putText(frame, clase, (x1 + pad_x, y_text), FUENTE, 0.45, color, 1, cv2.LINE_AA)
        cv2.putText(frame, str(cant), (x2 - 20, y_text), FUENTE, 0.45, color, 1, cv2.LINE_AA)


class MedidorFPS:
    def __init__(self, ventana=30):
        self._t = deque(maxlen=ventana)

    def tick(self) -> float:
        self._t.append(time.perf_counter())
        if len(self._t) < 2:
            return 0.0
        return (len(self._t) - 1) / (self._t[-1] - self._t[0])


class UI:
    def __init__(self, cfg: dict):
        self.nombre = cfg["ui"]["ventana"]
        self.mostrar_rechazadas = cfg["ui"]["mostrar_rechazadas"]
        self.mostrar_contador_clases = cfg.get("ui", {}).get("mostrar_contador_clases", True)
        self.mostrar_id = cfg.get("ui", {}).get("mostrar_id", True)
        self.clases = cfg.get("clases", ["taza", "botella", "tijeras", "destornillador", "celular"])
        cv2.namedWindow(self.nombre, cv2.WINDOW_NORMAL)
        self.trackbars = cfg["ui"]["trackbars"]
        if self.trackbars:
            pers = cfg["filtros"]["persistence"]
            cv2.createTrackbar("Umbral %", self.nombre, int(cfg["modelo"]["conf"] * 100), 100, lambda _: None)
            cv2.createTrackbar("Persistencia N", self.nombre, pers["n"], pers["m"], lambda _: None)

    def leer_trackbars(self) -> tuple[float, int] | None:
        if not self.trackbars:
            return None
        conf = cv2.getTrackbarPos("Umbral %", self.nombre) / 100
        n = max(1, cv2.getTrackbarPos("Persistencia N", self.nombre))
        return conf, n

    def dibujar(self, frame, detecciones: list[Detection], fps: float):
        conteo = {e: 0 for e in Estado}
        for d in detecciones:
            conteo[d.estado] += 1
            if d.estado == Estado.RECHAZADA and not self.mostrar_rechazadas:
                continue
            color = COLORES[d.estado]
            x1, y1, x2, y2 = d.bbox
            grosor = 2 if d.estado == Estado.ACEPTADA else 1
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, grosor)
            etiqueta = texto_etiqueta(d, self.mostrar_id)
            (tw, th), _ = cv2.getTextSize(etiqueta, FUENTE, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(frame, etiqueta, (x1 + 2, y1 - 4), FUENTE, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        if self.mostrar_contador_clases:
            conteo_clases = contar_por_clase(detecciones, self.clases)
            dibujar_contador(frame, conteo_clases)

        texto = (f"FPS {fps:5.1f} | aceptadas {conteo[Estado.ACEPTADA]} | "
                 f"rechazadas {conteo[Estado.RECHAZADA]} | desconocidas {conteo[Estado.DESCONOCIDO]}")
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 24), (0, 0, 0), -1)
        cv2.putText(frame, texto, (6, 17), FUENTE, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        return frame

    def pantalla_estado(self, mensaje: str, ancho=640, alto=480):
        """Frame negro con mensaje (sin cámara / reconectando)."""
        frame = np.zeros((alto, ancho, 3), dtype=np.uint8)
        cv2.putText(frame, mensaje, (20, alto // 2), FUENTE, 0.7, (0, 200, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "Presiona q para salir", (20, alto // 2 + 35), FUENTE, 0.5, (200, 200, 200), 1)
        return frame

    def mostrar(self, frame) -> bool:
        """Muestra el frame. Devuelve False si el usuario pidió salir (q o cerrar ventana)."""
        cv2.imshow(self.nombre, frame)
        return self.procesar_eventos()

    def procesar_eventos(self) -> bool:
        """Atiende teclado/ventana sin dibujar (mientras llega el siguiente frame)."""
        tecla = cv2.waitKey(1) & 0xFF
        if tecla == ord("q"):
            return False
        return cv2.getWindowProperty(self.nombre, cv2.WND_PROP_VISIBLE) >= 1

    def cerrar(self):
        cv2.destroyAllWindows()
