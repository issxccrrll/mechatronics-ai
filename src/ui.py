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
            etiqueta = f"{d.clase} {d.confianza:.2f}"
            if d.estado == Estado.RECHAZADA:
                etiqueta += f" [{d.rechazada_por}]"
            (tw, th), _ = cv2.getTextSize(etiqueta, FUENTE, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(frame, etiqueta, (x1 + 2, y1 - 4), FUENTE, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

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
