# CLAUDE.md: contexto del proyecto

Detector de objetos en vivo para el reto "Mechatronics AI" (Hack Knights IA Challenge 2026).

## Requisitos no negociables
- Python 3.11 y CPU con **15 FPS o más** en el pipeline completo.
- Las 5 clases oficiales son `taza, botella, tijeras, destornillador, celular`. Cualquier otra cosa es `desconocido` o se ignora; **nunca** se le asigna una clase oficial.
- La fuente es configurable (índice de webcam o URL). Se sale con `q`, y la pérdida de cámara se reintenta con un mensaje en pantalla, sin crashear.

## Arquitectura
`main.py` solo orquesta: Camara → YoloDetector → Verifier → filtros → UI + DetectionLogger.
- La configuración vive en `config.yaml`. `src/config.py` aplica los overrides `--source --conf --model --imgsz`.
- `src/types.py`: `Detection` y `Estado` (ACEPTADA, RECHAZADA, DESCONOCIDO). Un filtro rechaza con `d.rechazar(nombre)`.
- Filtros (Strategy) en `src/filters.py`:
  - Cada uno implementa `aplicar(detecciones, frame_shape)` y solo evalúa detecciones ACEPTADAS.
  - Para agregar uno: crear la clase, registrarla en `construir_filtros` y agregar su bloque on/off en `config.yaml`.
- El mapeo de clases del modelo a las oficiales está en `config.yaml > mapeo_clases`. Con `best.pt` propio, usar el mapeo identidad.
- El detector usa un umbral bajo (0.25); el umbral real lo aplica `ClassThresholdFilter.umbral_global`, que mueve el trackbar.

## Convenciones
- Comentarios, nombres de clases del reto y mensajes en español. Módulos cortos.
- Ejecutar siempre desde la raíz del repo: `python main.py`, `python scripts/x.py`.
  - `src/types.py` sombrea al módulo estándar `types` si se ejecuta `src/` directamente.
- Entorno: `venv\Scripts\python.exe` (Windows). Torch se instala desde el índice CPU.
- Pruebas: `venv\Scripts\python.exe -m pytest`. Toda lógica nueva de filtros lleva su prueba con detecciones simuladas.

## Rendimiento medido (laptop de desarrollo)
- Inferencia a imgsz 640: yolo11n.pt unos 22.5 FPS; OpenVINO unos 62 FPS. Por eso OpenVINO es el modelo por defecto.
- La primera inferencia con OpenVINO tarda unos 10 s; `YoloDetector._calentar()` la hace antes de abrir la cámara.

## TODO principales
- Dataset propio y `best.pt`. COCO no tiene destornillador.
- Verificador CNN (bonus) en `src/verifier.py`.
- Completar respuestas.md y metricas.md con resultados reales.
