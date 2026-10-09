"""Lectura de cámara en hilo separado con reconexión automática."""
import platform
import threading
import time

import cv2


class Camara:
    """Mantiene siempre el frame más reciente; nunca lanza excepción por pérdida de señal."""

    def __init__(self, source, ancho=640, alto=480, reintentos_max=0, espera_reintento_s=2.0):
        self.source = source
        self.ancho, self.alto = ancho, alto
        self.reintentos_max = reintentos_max
        self.espera = espera_reintento_s
        self.estado = "conectando"     # conectando | ok | reconectando | sin_camara
        self.intentos = 0
        self._frame = None
        self._nuevo = False
        self._lock = threading.Lock()
        self._activo = True
        self._cap = None
        self._hilo = threading.Thread(target=self._bucle, daemon=True)
        self._hilo.start()

    def _abrir(self) -> bool:
        if isinstance(self.source, int) and platform.system() == "Windows":
            cap = cv2.VideoCapture(self.source, cv2.CAP_DSHOW)
        elif isinstance(self.source, str) and "://" in self.source:
            # Timeouts para que una cámara IP caída no bloquee el hilo indefinidamente
            cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG,
                                   [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000, cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000])
        else:
            cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            cap.release()
            return False
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.ancho)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.alto)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._cap = cap
        return True

    def _bucle(self):
        while self._activo:
            if self._cap is None:
                if self._abrir():
                    self.estado, self.intentos = "ok", 0
                else:
                    self.intentos += 1
                    agotado = self.reintentos_max and self.intentos >= self.reintentos_max
                    self.estado = "sin_camara" if agotado else "reconectando"
                    if agotado:
                        return
                    time.sleep(self.espera)
                    continue
            ok, frame = self._cap.read()
            if not ok or frame is None:
                # Se perdió la conexión: liberar y reintentar en la siguiente vuelta
                self._cap.release()
                self._cap = None
                self.estado = "reconectando"
                continue
            with self._lock:
                self._frame, self._nuevo = frame, True

    def leer(self):
        """Devuelve (frame_nuevo_o_None). Solo entrega cada frame una vez."""
        with self._lock:
            if not self._nuevo:
                return None
            self._nuevo = False
            return self._frame

    def cerrar(self):
        self._activo = False
        self._hilo.join(timeout=2)
        if self._cap is not None:
            self._cap.release()
