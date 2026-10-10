# Revisión de README.md y respuestas.md

Revisión **sin editar** esos archivos. Orden: primero lo que puede hacer que **el código no corra siguiendo el README**.

Comprobado contra: `main.py`, `src/config.py`, `src/detector.py`, `src/ui.py`, `src/camera.py`, `src/filters.py`, `scripts/*.py`, `config.yaml`, `metricas.md`, `requirements.txt`, `.gitignore`.

---

## 1. Riesgo de que el código no corra siguiendo el README

### 1.1 No dice desde qué carpeta ejecutar (alto)

El README nunca indica `cd` a la raíz del repo ni clonar el proyecto. `pytest` usa `testpaths = tests` relativo al directorio actual. `modelo.ruta` en `config.yaml` es relativa (`models/best_openvino_model`). Si el jurado crea el venv o corre `python main.py` fuera de la raíz, falla.

**Sugerencia:** al inicio de Instalación y de Ejecución: clonar, `cd` al repo, y «todos los comandos se ejecutan desde la raíz».

### 1.2 `python scripts/eval.py --model models/best.pt` falla sin dataset (alto)

README líneas 97–102 y 118. Las fotos están en `.gitignore` (`dataset/*` salvo `data.yaml`). En un clone limpio `eval.py` no tiene split de test. Eso no impide `main.py`, pero **sí** rompe la sección Evaluación tal cual está escrita.

**Sugerencia:** aclarar que las métricas ya están en `metricas.md` / `metricas.csv`, y que `eval.py` solo se reproduce con el dataset (no incluido).

### 1.3 `python scripts/autolabel.py --vista` usa un peso que no está en el repo (medio)

README línea 177. El default de `autolabel.py` es `models/yolo11n.pt`, ignorado por git. El bloque de descarga de COCO está más arriba y parece opcional. Quien siga el flujo de dataset en orden se encuentra con archivo faltante.

`--vista` sí existe.

### 1.4 `pytest` a secas (bajo–medio)

README línea 100: `pytest`. Con el venv activado funciona (`pytest` está en `requirements.txt`). Sin activar, no. Más robusto: `python -m pytest`. Hay que estar en la raíz (punto 1.1).

### 1.5 Lo que sí coincide y debería correr para la demo

| Comando / ruta | ¿Existe? |
|---|---|
| `python main.py --source 0` | Sí. `--source` en `src/config.py`. |
| `python main.py --source http://... --conf 0.5` | Sí. `--conf`. |
| `python main.py --model models/best_openvino_model --imgsz 480` | Sí. Carpeta OpenVINO en el repo (`!models/best_openvino_model/` en `.gitignore`). `--model`, `--imgsz`. |
| `models/best.pt`, `models/best.onnx` | Sí, excepciones del gitignore. |
| `python scripts/ensayo_escenas.py --listar` | Sí. `--listar` no abre cámara. |
| `python scripts/ensayo_escenas.py` | Sí. ESPACIO / `s` / `q` como dice el README. |
| `python scripts/benchmark_fps.py --model models/best_openvino_model` | Sí. `--model` existe. |
| `python scripts/capture.py --clase taza\|mixto\|fondo` | Sí. `--clase` con esas opciones. |
| `python scripts/capture.py --clase tijeras --ejemplar ...` | Sí. `--ejemplar`. |
| `python scripts/split_dataset.py --ejemplares-test ...` | Sí. |
| `python scripts/export.py --model ... --formatos openvino` | Sí. `--formatos`. |
| `python scripts/autolabel_zeroshot.py --carpeta destornillador --texto ... --vista` | Sí. `--carpeta`, `--texto`, `--vista`. |
| `python scripts/train.py --epochs 80 --device 0` | Sí. `--epochs`, `--device`. |
| `python scripts/autolabel.py --model models/best.pt --carpetas destornillador` | Sí. `--model`, `--carpetas`. |
| `python scripts/autolabel.py --sobrescribir` | Sí. |
| Trackbars, contador, `mostrar_id` | Sí, en `src/ui.py` y `config.yaml > ui`. |

Instalación: `py -3.11`, venv, torch CPU, `pip install -r requirements.txt` es coherente con `requirements.txt` (incluye `opencv-python==5.0.0.93`). No hace falta entrenar para `main.py`; eso el README lo dice bien.

---

## 2. Comandos o textos del README que no coinciden con el código

| Archivo:línea | Texto actual | Qué está mal |
|---|---|---|
| README:64 | La ventana muestra *«Reintentando conexión…»* | En `main.py` el mensaje es `"Sin senal. Reintentando conexion ({intentos})..."`. En `ensayo_escenas.py` es `"Sin señal. Reintentando conexión..."`. Ninguno es exactamente el del README. |
| README:75 | Trackbar «umbral global (%)» | El control se llama **`Umbral %`** (`src/ui.py`). |
| README:78 | Ejemplo `taza #3 0.87` | El código pone un espacio antes del `#`: `taza #3 0.87`. Coincide. (Nada que corregir.) |
| README:137 | Umbrales taza 0.45, botella 0.45, tijeras 0.50, destornillador 0.55, celular 0.50; global **0.40** | `config.yaml`: global `modelo.conf: 0.30`; por clase 0.30 / 0.30 / 0.40 / 0.32 / 0.30. El trackbar arranca en 30, no en 40. |
| README:223–225 | Con modelo propio, «reemplaza `mapeo_clases` por el mapeo identidad» | El default ya es el modelo propio. `YoloDetector._mapear` acepta nombres oficiales aunque el YAML siga con cup/bottle/… El mapeo identidad **no es necesario**; dejar el de COCO no rompe el modelo propio. El párrafo confunde. |
| README:179 | `python scripts/split_dataset.py` | Sin `--bloques`. El v0 en `respuestas.md` **sí** usó `--bloques`. Si ya existen `dataset/train|val|test`, el script sale con error y pide `--reemplazar` (no documentado). |
| README:100 | `pytest` | No es un mismatch de argparse; ver 1.4. |
| README:171 | Reinstalar `opencv-python==5.0.0.93` | Coincide con `requirements.txt`. `train.py` al faltar Albumentations no fija esa versión. |

Opciones de argparse citadas en el README: todas existen. No hay flags inventados.

Scripts que existen y el README no menciona (no es error): `scripts/calibrar_umbrales.py`, `scripts/integrar_roboflow.py`.

---

## 3. Contradicciones entre README, respuestas.md, metricas.md y config.yaml

### 3.1 Umbrales de operación (grave para la historia del filtro)

| Fuente | Global | taza | botella | tijeras | destornillador | celular |
|---|---|---|---|---|---|---|
| README y `respuestas.md` | 0.40 | 0.45 | 0.45 | 0.50 | 0.55 | 0.50 |
| `config.yaml` (lo que corre en vivo) | **0.30** | 0.30 | 0.30 | 0.40 | 0.32 | 0.30 |
| `metricas.md` / `eval.py` | val a **conf 0.4** | — | — | — | — | — |

`eval.py` documenta `--conf-operacion` como «umbral usado en vivo (config.yaml)», default 0.40, pero el vivo es 0.30 más umbrales por clase distintos.

Además, README/`respuestas.md` dicen que la precision 0.856 → 0.874 es el `ClassThresholdFilter`. En realidad `eval.py` compara `modelo.val(conf=0.001)` vs `val(conf=0.40)` de Ultralytics, **sin** pasar por `src/filters.py`.

### 3.2 FPS OpenVINO de inferencia

| Fuente | Inferencia OpenVINO | Pipeline en vivo | PyTorch |
|---|---|---|---|
| README (tablas y decisiones) | **~32–35** (modelo propio) | 17.8–29.8 | ~22.5 |
| `respuestas.md` §1 y §7 | **~62** y «triplica … ~62 vs ~22.5» | 17.8–29.8 | ~22.5 |
| `metricas.md` | no hay FPS | no hay FPS | — |
| `config.yaml` comentario | «~2.5x más rápido» (encaja más con 62/22.5 que con 32–35) | — | — |

Los 17.8–29.8 coinciden. El **62 vs 32–35** no. El 62 encaja con OpenVINO de `yolo11n` COCO (notas internas del repo), no con el modelo propio de las tablas del README.

### 3.3 `batch` del entrenamiento

`respuestas.md` §4: `batch: -1 (auto)`. `scripts/train.py`: default **`16`**. El comando del README (`--epochs 80 --device 0`) no pasa `--batch`, así que entrena a 16, no a −1.

### 3.4 «112 s por época»

README: «con miles de fotos (una época tarda ~112 s)». `respuestas.md`: «una época con **427 imágenes** tarda 112 segundos». El split v0 tiene **1 432** de train. El 427 parece de un dataset anterior.

### 3.5 Split: sesión vs bloques

README decisiones: «Split por sesión de captura». Limitaciones del mismo README y `respuestas.md`: el v0 usó **`--bloques`** (70/15/15 dentro de cada sesión). Las dos frases pueden convivir si se distinguen «script por defecto» vs «lo que se usó en v0»; ahora se leen como si el v0 hubiera sido por sesión completa.

### 3.6 Métricas v0

Las tablas de README, `respuestas.md` y `metricas.md` **coinciden** (precision/recall/mAP y matriz a conf 0.40). Bien.

### 3.7 Dataset

`respuestas.md` da conteos (2 288 jpg, 2 036 txt, sin carpeta `botella/`). El README habla de ~2 000 y de botellas solo en mixto / captura en curso. Alineado en lo cualitativo; los números exactos solo están en `respuestas.md`.

---

## 4. Ortografía y redacción

Formato: archivo → actual → sugerido.

| Archivo | Texto actual | Texto sugerido |
|---|---|---|
| README:64 | *"Reintentando conexión…"* | Usar el texto real de `main.py`, o unificar código y docs (el README está mejor acentuado que el código). |
| README:149 | «sin filtro de umbral» / «con el ClassThresholdFilter activo» | «sin umbral de NMS estricto (conf 0.001)» / «con `val` a conf 0.40» — no es un filtro del pipeline. |
| README:257 vs 236 | «Test por bloques» vs «Split por sesión» | Dejar explícito: el script *puede* partir por sesión; el v0 usó `--bloques`. |
| `respuestas.md` §1 | OpenVINO «~62» FPS de inferencia | Alinear con README (~32–35 del modelo propio) o aclarar que 62 es COCO `yolo11n` OpenVINO. |
| `respuestas.md` §7 | «triplica los FPS respecto a PyTorch (~62 vs ~22.5)» | Con 32–35 vs 22.5 no triplica. Corregir el verbo y el número. |
| `respuestas.md` §4 | «427 imágenes» | Poner el tamaño real de train (1 432) o decir de qué corrida salió el 427. |
| README / `respuestas.md` | «precision», «recall» | Aceptable como nombres de métrica; si se españoliza: «precisión». No es error. |

No hay faltas graves de ortografía (tildes en clases, «detección», «etiquetado», «confusión» van bien). El problema es **cifras y nombres de umbral**, no la prosa.

---

## 5. Huecos respecto a la ficha del reto

La ficha pide README con: instalación, cámara, ejecución, evaluación, decisiones, fuentes, limitaciones y mejoras. **−10 si el código no corre siguiendo el README.**

| Pedido | ¿Está? |
|---|---|
| Instalación | Sí (Windows 3.11 + nota Linux/macOS). Falta clone + directorio de trabajo (1.1). |
| Configurar la cámara | Sí (webcam, IP Webcam, DroidCam, `--source`). |
| Ejecución | Sí (`main.py`, `q`, trackbars, colores). |
| Evaluación | Comandos sí; en clone limpio `eval.py` no reproduce (1.2). El ensayo de 20 escenas sí está. |
| Decisiones técnicas | Sí (README y `respuestas.md`). |
| Fuentes | Sí. |
| Limitaciones y mejoras | Sí. |
| Video ≤ 2 min | Entregable aparte; el README no tiene que incluirlo. Guion en `docs/guion_video.md`. |
| El jurado prueba poses, distancias, varios objetos, oclusión, luz, ajenos, fondo vacío | Cubierto en `scripts/ensayo_escenas.py` y en el README. No hace falta más texto para cumplir la ficha, sí para ensayar. |

Nada crítico de la ficha falta como *sección*, salvo dejar el camino de instalación a prueba de un clone en otra máquina (carpeta de trabajo, y no mandar a correr `eval.py` sin datos).

---

## 6. Prioridad para quien edite el README (no esta revisión)

1. Decir **raíz del repo** y comando canónico `python main.py --source 0` (demo).
2. Separar «correr el detector» de «reproducir métricas» (dataset no va en git).
3. Igualar umbrales README/`respuestas.md` ↔ `config.yaml` (o viceversa).
4. Un solo número de FPS OpenVINO (32–35 del propio vs 62 de COCO).
5. Mensaje de cámara igual en README y en `main.py`.
