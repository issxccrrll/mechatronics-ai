"""Dibujo en pantalla y trackbars de OpenCV."""
import time
from collections import deque

import cv2
import numpy as np

from .types import Detection, Estado

FUENTE = cv2.FONT_HERSHEY_SIMPLEX
ALTO_BARRA = 30

# Paleta BGR por clase (caja, etiqueta y punto del contador).
COLORES_CLASE = {
    "taza": (0, 165, 255),          # ambar
    "botella": (230, 130, 40),      # azul
    "tijeras": (210, 40, 210),      # magenta
    "destornillador": (50, 190, 70),  # verde
    "celular": (220, 200, 40),      # cian
}
ROJO_RECHAZADA = (50, 50, 230)
GRIS_DESCONOCIDO = (155, 155, 155)
COLORES = {  # BGR por estado (compatibilidad)
    Estado.ACEPTADA: (60, 200, 60),
    Estado.RECHAZADA: ROJO_RECHAZADA,
    Estado.DESCONOCIDO: GRIS_DESCONOCIDO,
}
FPS_VERDE = (60, 200, 70)
FPS_AMBAR = (0, 165, 255)
FPS_ROJO = (50, 50, 230)


def color_clase(nombre: str) -> tuple[int, int, int]:
    return COLORES_CLASE.get(nombre, GRIS_DESCONOCIDO)


def color_de_caja(d: Detection) -> tuple[int, int, int]:
    if d.estado == Estado.RECHAZADA:
        return ROJO_RECHAZADA
    if d.estado == Estado.DESCONOCIDO:
        return GRIS_DESCONOCIDO
    return color_clase(d.clase)


def color_insignia_fps(fps: float) -> tuple[int, int, int]:
    if fps >= 20:
        return FPS_VERDE
    if fps >= 15:
        return FPS_AMBAR
    return FPS_ROJO


def _luma(bgr) -> float:
    b, g, r = bgr
    return 0.114 * b + 0.587 * g + 0.299 * r


def color_texto(fondo_bgr) -> tuple[int, int, int]:
    return (20, 20, 20) if _luma(fondo_bgr) > 150 else (255, 255, 255)


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


def _fusionar(frame, y1, y2, x1, x2, alpha_fondo=0.35):
    """Oscurece solo la region [y1:y2, x1:x2]."""
    h, w = frame.shape[:2]
    y1, y2 = max(0, y1), min(h, y2)
    x1, x2 = max(0, x1), min(w, x2)
    if y2 <= y1 or x2 <= x1:
        return
    sub = frame[y1:y2, x1:x2]
    cv2.addWeighted(sub, alpha_fondo, np.zeros_like(sub), 1.0 - alpha_fondo, 0, sub)


def geom_etiqueta(bbox, texto: str, frame_shape, escala=0.5, grosor=1,
                  alto_barra=ALTO_BARRA) -> tuple[int, int, int, int]:
    """Rectangulo de la pildora (x1, y1, x2, y2). Si no cabe arriba del frame, va dentro de la caja."""
    x1, y1, x2, y2 = (int(v) for v in bbox)
    fh, fw = frame_shape[:2]
    (tw, th), _ = cv2.getTextSize(texto, FUENTE, escala, grosor)
    pad, barra = 4, 3
    ancho = tw + pad * 2
    alto = th + pad * 2 + barra
    rx1 = max(0, x1)
    if rx1 + ancho > fw:
        rx1 = max(0, fw - ancho)
    rx2 = min(fw, rx1 + ancho)
    # Preferir encima de la caja; si choca con la barra superior, dentro de la caja.
    ry1 = y1 - alto
    if ry1 < alto_barra:
        ry1 = max(y1, alto_barra)
    ry2 = ry1 + alto
    if ry2 > fh:
        ry2 = fh
        ry1 = max(0, fh - alto)
    return rx1, ry1, rx2, ry2


def _esquinas(frame, x1, y1, x2, y2, color, largo=11, grosor=2):
    l = max(6, min(largo, (x2 - x1) // 3, (y2 - y1) // 3))
    pares = (
        ((x1, y1), (x1 + l, y1), (x1, y1 + l)),
        ((x2, y1), (x2 - l, y1), (x2, y1 + l)),
        ((x1, y2), (x1 + l, y2), (x1, y2 - l)),
        ((x2, y2), (x2 - l, y2), (x2, y2 - l)),
    )
    for origen, a, b in pares:
        cv2.line(frame, origen, a, color, grosor, cv2.LINE_AA)
        cv2.line(frame, origen, b, color, grosor, cv2.LINE_AA)


def _pildora(frame, d: Detection, texto: str, color, aceptada: bool):
    escala = 0.5 if aceptada else 0.4
    grosor_txt = 1
    x1, y1, x2, y2 = d.bbox
    rx1, ry1, rx2, ry2 = geom_etiqueta(d.bbox, texto, frame.shape, escala, grosor_txt)
    cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), color, -1)
    barra_h = 3
    fill_w = int((rx2 - rx1) * max(0.0, min(1.0, d.confianza)))
    yb = ry2 - barra_h
    oscuro = tuple(int(c * 0.45) for c in color)
    cv2.rectangle(frame, (rx1, yb), (rx2, ry2), oscuro, -1)
    if fill_w > 0:
        claro = color_texto(color)
        cv2.rectangle(frame, (rx1, yb), (rx1 + fill_w, ry2), claro, -1)
    (tw, th), _ = cv2.getTextSize(texto, FUENTE, escala, grosor_txt)
    tx = rx1 + 4
    ty = ry1 + 4 + th
    cv2.putText(frame, texto, (tx, ty), FUENTE, escala, color_texto(color), grosor_txt, cv2.LINE_AA)


def dibujar_contador(frame: np.ndarray, conteo: dict[str, int]) -> None:
    """Dibuja un panel semitransparente con el conteo de clases en la esquina inferior izquierda.

    No invade la barra superior (filas 0-29).
    """
    if not conteo:
        return
    alto_frame, ancho_frame = frame.shape[:2]
    linea_h = 18
    pad_x, pad_y = 12, 8
    ancho_panel = 178
    alto_panel = len(conteo) * linea_h + pad_y * 2

    x1 = 8
    x2 = min(x1 + ancho_panel, ancho_frame - 1)
    y2 = alto_frame - 8
    y1 = max(ALTO_BARRA, y2 - alto_panel)

    if y2 <= y1 or x2 <= x1:
        return

    _fusionar(frame, y1, y2, x1, x2, 0.32)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (90, 90, 90), 1)

    for i, (clase, cant) in enumerate(conteo.items()):
        y_text = y1 + pad_y + (i + 1) * linea_h - 4
        punto = (x1 + 12, y_text - 4)
        cv2.circle(frame, punto, 4, color_clase(clase), -1, cv2.LINE_AA)
        color = (255, 255, 255) if cant > 0 else (130, 130, 130)
        cv2.putText(frame, clase, (x1 + pad_x + 10, y_text), FUENTE, 0.45, color, 1, cv2.LINE_AA)
        cv2.putText(frame, str(cant), (x2 - 22, y_text), FUENTE, 0.45, color, 1, cv2.LINE_AA)


def _barra_superior(frame, fps: float, conteo_estado: dict):
    h, w = frame.shape[:2]
    _fusionar(frame, 0, ALTO_BARRA, 0, w, 0.30)
    cv2.line(frame, (0, ALTO_BARRA - 1), (w, ALTO_BARRA - 1), (70, 70, 70), 1)

    cv2.putText(frame, "MECHATRONICS AI", (8, 20), FUENTE, 0.48, (240, 240, 240), 1, cv2.LINE_AA)

    centro = (f"aceptadas {conteo_estado[Estado.ACEPTADA]}  |  "
              f"rechazadas {conteo_estado[Estado.RECHAZADA]}  |  "
              f"desconocidas {conteo_estado[Estado.DESCONOCIDO]}")
    (tw, th), _ = cv2.getTextSize(centro, FUENTE, 0.42, 1)
    cv2.putText(frame, centro, ((w - tw) // 2, 20), FUENTE, 0.42, (210, 210, 210), 1, cv2.LINE_AA)

    badge = f"FPS {fps:4.1f}"
    (bw, bh), _ = cv2.getTextSize(badge, FUENTE, 0.48, 1)
    bx2, by2 = w - 8, 25
    bx1, by1 = bx2 - bw - 10, 5
    color_fps = color_insignia_fps(fps)
    cv2.rectangle(frame, (bx1, by1), (bx2, by2), color_fps, -1)
    cv2.putText(frame, badge, (bx1 + 5, 20), FUENTE, 0.48, color_texto(color_fps), 1, cv2.LINE_AA)


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
            color = color_de_caja(d)
            x1, y1, x2, y2 = d.bbox
            aceptada = d.estado == Estado.ACEPTADA
            grosor = 2 if aceptada else 1
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, grosor)
            if aceptada:
                _esquinas(frame, x1, y1, x2, y2, color)
            etiqueta = texto_etiqueta(d, self.mostrar_id)
            _pildora(frame, d, etiqueta, color, aceptada)

        if self.mostrar_contador_clases:
            conteo_clases = contar_por_clase(detecciones, self.clases)
            dibujar_contador(frame, conteo_clases)

        _barra_superior(frame, fps, conteo)
        return frame

    def pantalla_estado(self, mensaje: str, ancho=640, alto=480):
        """Frame negro con mensaje (sin camara / reconectando)."""
        frame = np.zeros((alto, ancho, 3), dtype=np.uint8)
        cx, cy = ancho // 2, alto // 2 - 40
        color = (0, 200, 255)
        cv2.circle(frame, (cx, cy), 32, color, 2, cv2.LINE_AA)
        cv2.line(frame, (cx - 22, cy - 22), (cx + 22, cy + 22), color, 2, cv2.LINE_AA)
        (tw, th), _ = cv2.getTextSize(mensaje, FUENTE, 0.65, 1)
        cv2.putText(frame, mensaje, ((ancho - tw) // 2, cy + 70), FUENTE, 0.65, color, 1, cv2.LINE_AA)
        ayuda = "Presiona q para salir"
        (aw, _), _ = cv2.getTextSize(ayuda, FUENTE, 0.5, 1)
        cv2.putText(frame, ayuda, ((ancho - aw) // 2, cy + 100), FUENTE, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        return frame

    def mostrar(self, frame) -> bool:
        """Muestra el frame. Devuelve False si el usuario pidio salir (q o cerrar ventana)."""
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
