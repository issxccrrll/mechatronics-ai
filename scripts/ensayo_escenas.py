"""Script de ensayo de las 20 escenas del jurado."""
import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

# Añadir raíz a sys.path para permitir importaciones de src
RAIZ = Path(__file__).resolve().parent.parent
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from src.types import Estado


ESCENAS = [
    {"descripcion": "Taza de frente, distancia media", "esperado": {"taza": 1}},
    {"descripcion": "Botella de plástico de pie", "esperado": {"botella": 1}},
    {"descripcion": "Tijeras abiertas sobre la mesa", "esperado": {"tijeras": 1}},
    {"descripcion": "Destornillador acostado sobre la mesa", "esperado": {"destornillador": 1}},
    {"descripcion": "Celular boca arriba sobre la mesa", "esperado": {"celular": 1}},
    {"descripcion": "Taza lejos (~1.5 m)", "esperado": {"taza": 1}},
    {"descripcion": "Celular muy cerca de la cámara", "esperado": {"celular": 1}},
    {"descripcion": "Botella acostada", "esperado": {"botella": 1}},
    {"descripcion": "Destornillador en la mano, inclinado", "esperado": {"destornillador": 1}},
    {"descripcion": "Tijeras cerradas, de lado", "esperado": {"tijeras": 1}},
    {"descripcion": "Taza y celular juntos", "esperado": {"taza": 1, "celular": 1}},
    {"descripcion": "Botella, tijeras y destornillador juntos", "esperado": {"botella": 1, "tijeras": 1, "destornillador": 1}},
    {"descripcion": "Las 5 clases juntas", "esperado": {"taza": 1, "botella": 1, "tijeras": 1, "destornillador": 1, "celular": 1}},
    {"descripcion": "Taza parcialmente tapada con la mano", "esperado": {"taza": 1}},
    {"descripcion": "Destornillador parcialmente oculto detrás de la botella", "esperado": {"botella": 1, "destornillador": 1}},
    {"descripcion": "Luz tenue: celular y tijeras", "esperado": {"celular": 1, "tijeras": 1}},
    {"descripcion": "Contraluz o luz fuerte: botella", "esperado": {"botella": 1}},
    {"descripcion": "Objeto ajeno: pluma o lápiz", "esperado": {}},
    {"descripcion": "Objetos ajenos: vaso, control remoto, cuchara", "esperado": {}},
    {"descripcion": "Fondo vacío (mesa sin objetos)", "esperado": {}},
]


def conteo_aceptadas(detecciones, clases) -> dict[str, int]:
    """Cuenta detecciones ACEPTADAS de clases oficiales."""
    conteo = {}
    oficiales = set(clases)
    for d in detecciones:
        if d.estado == Estado.ACEPTADA and d.clase in oficiales:
            conteo[d.clase] = conteo.get(d.clase, 0) + 1
    return conteo


def moda_conteos(conteos: list[dict]) -> dict:
    """Retorna el conteo mayoritario en la lista. En empate, el primero."""
    if not conteos:
        return {}
    freqs = {}
    primera_aparicion = {}
    for i, c in enumerate(conteos):
        fs = frozenset({k: v for k, v in c.items() if v > 0}.items())
        freqs[fs] = freqs.get(fs, 0) + 1
        if fs not in primera_aparicion:
            primera_aparicion[fs] = i
    
    mejor_fs = None
    mejor_freq = -1
    mejor_primera = float('inf')
    for fs, freq in freqs.items():
        if freq > mejor_freq or (freq == mejor_freq and primera_aparicion[fs] < mejor_primera):
            mejor_freq = freq
            mejor_primera = primera_aparicion[fs]
            mejor_fs = fs
    
    return dict(mejor_fs) if mejor_fs else {}


def es_acierto(esperado: dict, obtenido: dict) -> bool:
    """Verifica si obtenido coincide exactamente con esperado (ignorando ceros)."""
    e = {k: v for k, v in esperado.items() if v > 0}
    o = {k: v for k, v in obtenido.items() if v > 0}
    return e == o


def resumen(resultados: list[dict]) -> tuple[int, int, float]:
    """Calcula el resumen: (aciertos, evaluadas, porcentaje)."""
    evaluadas = len(resultados)
    if evaluadas == 0:
        return 0, 0, 0.0
    aciertos = sum(1 for r in resultados if r["acierto"])
    return aciertos, evaluadas, (aciertos / evaluadas) * 100


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--listar", action="store_true", help="Muestra las escenas sin abrir cámara")
    parser.add_argument("--help", "-h", action="store_true", help="Muestra esta ayuda")
    args, resto = parser.parse_known_args()

    if args.help:
        parser.print_help()
        sys.exit(0)

    if args.listar:
        for i, esc in enumerate(ESCENAS, 1):
            print(f"{i:2d}. {esc['descripcion']} -> {esc['esperado']}")
        sys.exit(0)

    # Importaciones pesadas solo si se va a abrir el pipeline
    import cv2

    from src.camera import Camara
    from src.config import cargar_config
    from src.detector import YoloDetector
    from src.filters import aplicar_filtros, construir_filtros
    from src.ui import UI, MedidorFPS
    from src.verifier import construir_verificador

    cfg = cargar_config(resto)
    cam_cfg = cfg["camara"]
    clases_oficiales = cfg["clases"]

    detector = YoloDetector(cfg["modelo"], cfg["mapeo_clases"], clases_oficiales)
    verificador = construir_verificador(cfg["verificador"])
    filtros = construir_filtros(cfg)
    ui = UI(cfg)
    fps_meter = MedidorFPS()
    camara = Camara(cam_cfg["source"], cam_cfg["ancho"], cam_cfg["alto"],
                    cam_cfg["reintentos_max"], cam_cfg["espera_reintento_s"])

    fuente = cv2.FONT_HERSHEY_SIMPLEX

    def procesar_frame(frame):
        """Pipeline completo: detectar → verificar → filtrar → dibujar."""
        dets = detector.detectar(frame)
        dets = verificador.verificar(frame, dets)
        dets = aplicar_filtros(filtros, dets, frame.shape)
        frame = ui.dibujar(frame, dets, fps_meter.tick())
        return frame, dets

    def leer_o_estado(timeout=0.05):
        """Lee un frame; si no hay, muestra pantalla de estado si la cámara está desconectada."""
        frame = camara.leer(timeout=timeout)
        if frame is not None:
            return frame, True
        if camara.estado != "ok":
            estado_frame = ui.pantalla_estado("Sin senal. Reintentando conexion...")  # la fuente de OpenCV no dibuja acentos
            cv2.imshow(ui.nombre, estado_frame)
        return None, False

    resultados = []

    try:
        for i, esc in enumerate(ESCENAS, 1):
            terminar = False

            while True:
                frame, hay_frame = leer_o_estado()
                if not hay_frame:
                    if (cv2.waitKey(30) & 0xFF) == ord('q'):
                        terminar = True
                        break
                    continue

                frame, _dets = procesar_frame(frame)

                # Rótulo de la escena
                texto_arriba = f"Escena {i}/{len(ESCENAS)}: {esc['descripcion']}"
                texto_abajo = "ESPACIO = evaluar | s = saltar | q = terminar"
                cv2.rectangle(frame, (0, 24), (frame.shape[1], 75), (0, 0, 0), -1)
                cv2.putText(frame, texto_arriba, (10, 45), fuente, 0.6, (0, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(frame, texto_abajo, (10, 65), fuente, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
                cv2.imshow(ui.nombre, frame)
                tecla = cv2.waitKey(1) & 0xFF

                if tecla == ord('s'):
                    break
                elif tecla == ord('q'):
                    terminar = True
                    break
                elif tecla == ord(' '):
                    # Evaluación: capturar 30 frames por el pipeline
                    capturas = []
                    while len(capturas) < 30:
                        f2, hay = leer_o_estado()
                        if not hay:
                            if (cv2.waitKey(10) & 0xFF) == ord('q'):
                                terminar = True
                                break
                            continue
                        f2, d2 = procesar_frame(f2)
                        capturas.append(conteo_aceptadas(d2, clases_oficiales))
                        cv2.rectangle(f2, (0, 24), (f2.shape[1], 75), (0, 0, 0), -1)
                        cv2.putText(f2, f"Evaluando... {len(capturas)}/30", (10, 55), fuente, 0.7, (0, 200, 255), 2)
                        cv2.imshow(ui.nombre, f2)
                        if (cv2.waitKey(1) & 0xFF) == ord('q'):
                            terminar = True
                            break

                    if terminar:
                        break

                    moda = moda_conteos(capturas)
                    acierto = es_acierto(esc['esperado'], moda)

                    # Pantalla de resultado (~2 s)
                    color_res = (0, 255, 0) if acierto else (0, 0, 255)
                    texto_res = "ACIERTO" if acierto else "FALLO"
                    y_c = frame.shape[0] // 2
                    f_res = frame.copy()
                    cv2.rectangle(f_res, (0, y_c - 40), (f_res.shape[1], y_c + 40), (0, 0, 0), -1)
                    cv2.putText(f_res, f"{texto_res} (Obtenido: {moda} | Esperado: {esc['esperado']})",
                                (10, y_c + 10), fuente, 0.55, color_res, 2, cv2.LINE_AA)
                    cv2.imshow(ui.nombre, f_res)
                    cv2.waitKey(2000)

                    resultados.append({
                        "escena": i,
                        "descripcion": esc["descripcion"],
                        "esperado": str(esc["esperado"]),
                        "obtenido": str(moda),
                        "acierto": acierto,
                        "frames": len(capturas),
                    })
                    break

            if terminar:
                break
    finally:
        camara.cerrar()
        cv2.destroyAllWindows()

        # Reporte en consola y CSV
        if resultados:
            aciertos, evaluadas, pct = resumen(resultados)
            print("\n--- RESUMEN DEL ENSAYO ---")
            for r in resultados:
                marca = "OK" if r["acierto"] else "X "
                print(f"Escena {r['escena']:2d} [{marca}]: Obtenido {r['obtenido']} vs {r['esperado']}")
            print(f"\nAciertos: {aciertos}/{evaluadas} = {pct:.1f}% (meta: 85%)")

            logs_dir = RAIZ / "logs"
            logs_dir.mkdir(exist_ok=True)
            t_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_path = logs_dir / f"ensayo_{t_str}.csv"
            with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
                w = csv.DictWriter(f, fieldnames=["escena", "descripcion", "esperado", "obtenido", "acierto", "frames"])
                w.writeheader()
                w.writerows(resultados)
            print(f"Reporte guardado en {csv_path.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
