# Mechatronics AI: detector de objetos en vivo

Proyecto para el reto **Mechatronics AI** del Hack Knights IA Challenge 2026.
Detecta en vivo, desde una webcam o la cámara de un celular, cinco clases: **taza, botella, tijeras, destornillador y celular**.
Cualquier otro objeto se marca como `desconocido` y nunca recibe una clase oficial.

## Arquitectura

```
Cámara (hilo) → Detector (YOLO11n) → Verificador (opcional) → Filtros → UI + Logger CSV
```

| Módulo | Responsabilidad |
|---|---|
| `src/camera.py` | Lee en un hilo separado y reconecta automáticamente. |
| `src/detector.py` | `YoloDetector` con `.pt` u OpenVINO; mapeo de clases desde `config.yaml`. |
| `src/verifier.py` | Stub del clasificador CNN propio (bonus). Desactivado por defecto. |
| `src/filters.py` | Filtros anti falsos positivos activables: desconocidos, umbral por clase, área y aspecto, persistencia N de M. |
| `src/ui.py` | Cajas, etiquetas, FPS, contador por clase, ID de seguimiento y trackbars. |
| `src/logger.py` | Escribe un CSV por sesión en `logs/`. |

## Instalación (Windows, Python 3.11)

**Todos los comandos de este README se ejecutan desde la raíz del repositorio** (`mechatronics-ai`),
con el entorno virtual activado.

```powershell
git clone https://github.com/issxccrrll/mechatronics-ai.git
cd mechatronics-ai
# Si no tienes Python 3.11: uv python install 3.11  (convive con otras versiones)
py -3.11 -m venv venv            # o: & (uv python find 3.11) -m venv venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass   # permite activar el venv en esta terminal
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

En Linux o macOS: `python3.11 -m venv venv && source venv/bin/activate` y los mismos `pip install`.

Los pesos del modelo propio ya vienen incluidos en `models/`:
- `models/best_openvino_model/` — modelo OpenVINO (por defecto en `config.yaml`).
- `models/best.pt` — pesos PyTorch.
- `models/best.onnx` — exportación ONNX (bonus).

Si la carpeta OpenVINO no existe, el detector usa automáticamente `best.pt` (más lento).
**No es necesario descargar ni entrenar nada para ejecutar el detector.**

> **Nota:** Para usar el modelo COCO de respaldo (sin destornillador), descarga los pesos base y
> expórtalos a OpenVINO. Esto solo hace falta para el pre-etiquetado (`autolabel.py`) o para
> comparar con el modelo COCO:
> ```powershell
> cd models; python -c "from ultralytics import YOLO; YOLO('yolo11n.pt')"; cd ..
> python scripts/export.py --model models/yolo11n.pt --formatos openvino
> python main.py --model models/yolo11n_openvino_model
> ```

## Conexión de cámara

- **Webcam de la laptop:** `--source 0` (o 1 o 2 si hay varias).
- **Celular con IP Webcam (Android):**
  1. Abre la app y pulsa *Iniciar servidor*.
  2. Usa la URL que muestra: `--source http://IP:8080/video`.
  3. El celular y la PC deben estar en la misma red Wi-Fi.
- **Celular con DroidCam:**
  - Por URL: `--source http://IP:4747/video`.
  - O instala el cliente de Windows, que crea una webcam virtual, y usa `--source 1`.

Si la cámara no está disponible o se pierde la conexión, la ventana muestra *"Sin senal. Reintentando conexion (N)..."* (sin acentos: la fuente de OpenCV no los dibuja) y reintenta sin cerrarse.

## Ejecución

```powershell
python main.py --source 0
python main.py --source http://192.168.1.50:8080/video --conf 0.5
python main.py --model models/best_openvino_model --imgsz 480
```

- Tecla **q**: salir.
- **Trackbars** `Umbral %` y `Persistencia N`: umbral global de confianza y persistencia, ajustables en vivo.
- **Colores**: verde = aceptada, rojo = rechazada (con el filtro que la rechazó), gris = desconocido.
- **Contador por clase**: panel en la esquina inferior izquierda con el conteo de objetos aceptados por clase. Configurable en `config.yaml > ui > mostrar_contador_clases`.
- **ID de seguimiento**: cada detección muestra su ID de ByteTrack (`taza #3 0.87`). Configurable en `config.yaml > ui > mostrar_id`.
- Toda la configuración está en `config.yaml`.

### Ensayo de las 20 escenas

El script `scripts/ensayo_escenas.py` permite ensayar las 20 escenas que evaluará el jurado, usando el mismo pipeline del detector, y calcula el porcentaje de aciertos (meta: 85%).

```powershell
python scripts/ensayo_escenas.py --listar              # muestra las 20 escenas sin abrir cámara
python scripts/ensayo_escenas.py                       # abre la cámara con detección en vivo
```

Durante el ensayo:
- **ESPACIO**: evalúa la escena actual (captura 30 frames y calcula la moda).
- **s**: salta la escena (no cuenta en el total).
- **q**: termina antes.

Al finalizar imprime el resumen en consola y guarda un CSV en `logs/ensayo_AAAAMMDD_HHMMSS.csv`.

## Evaluación

```powershell
python -m pytest                                        # pruebas unitarias
python scripts/eval.py --model models/best.pt           # usa dataset/test (incluido); escribe metricas.md y metricas.csv
python scripts/benchmark_fps.py --model models/best_openvino_model
```

### Resultados del modelo final (v2) en test (471 imágenes, incluido en `dataset/test/`)

| clase          | precision | recall | mAP50  | mAP50-95 | mAP50 del v0 |
|----------------|----------:|-------:|-------:|---------:|-------------:|
| **todas**      |    0.7799 | 0.6951 | **0.7404** | 0.5362 | 0.626 |
| taza           |    0.6485 | 0.7566 | 0.7467 |   0.6113 | 0.676 |
| botella        |    0.7543 | 0.4380 | 0.5032 |   0.2051 | 0.325 |
| tijeras        |    0.7041 | 0.7083 | 0.7415 |   0.4667 | 0.588 |
| destornillador |    0.8815 | 0.6760 | 0.7839 |   0.5454 | 0.614 |
| celular        |    0.9109 | 0.8966 | 0.9267 |   0.8526 | 0.926 |

El test incluye una **botella que el modelo nunca vio** al entrenar. Matriz de confusión y detalle en
[`metricas.md`](metricas.md).

### Rendimiento en CPU (Intel Core i7-13620H)

| Medición                                   | FPS |
|--------------------------------------------|-----|
| Inferencia OpenVINO del modelo final       | ~47.6 (benchmark) |
| Inferencia PyTorch (`.pt`)                 | ~22.5 |
| Pipeline completo en vivo (cámara + todo)  | 16–30 según la carga de la laptop; 25.9 con la interfaz final y la laptop libre |
| Requisito mínimo del reto                  | 15 |

## Justificación de la técnica contra falsos positivos

El pipeline aplica 4 filtros en cascada (patrón Strategy, cada uno activable en `config.yaml`, definidos en `src/filters.py`):

### 1. UnknownRejectFilter
Cualquier clase que no sea una de las 5 oficiales pasa a estado `desconocido`. Garantiza la regla del reto: nunca se asigna una clase oficial a un objeto ajeno.

### 2. ClassThresholdFilter
Umbral de confianza individual por clase (taza 0.30, botella 0.33, tijeras 0.50, destornillador 0.45, celular 0.30), nunca inferior al umbral global (0.30, ajustable en vivo con el trackbar). **Calibrados con datos** con `scripts/calibrar_umbrales.py`: el umbral más bajo cuyo F1 en validación queda a menos de 0.02 del máximo (tijeras limitado a 0.50 para no perder tijeras en la mano; destornillador subido a 0.45 tras la prueba en vivo: precisión 0.87 y recall 0.71 en validación, frente a 0.78 y 0.78 con 0.30).

### 3. AreaAspectFilter
Rechaza cajas con área menor al 0.2% o mayor al 90% del frame, o con relación de aspecto fuera de 0.15–6.0. Elimina cajas físicamente imposibles.

### 4. PersistenceFilter
Un objeto solo se acepta si aparece en al menos **4 de los últimos 5 frames**, usando el `track_id` de ByteTrack. Sin ID, empareja por IoU ≥ 0.4. Elimina los falsos positivos más comunes en video: detecciones de 1–2 frames por ruido o reflejos.

### Evidencia con datos

- **Calibración:** en una prueba en vivo, el registro CSV (`logs/`) mostró que los objetos en la mano se detectaban con confianza 0.34–0.40 y los umbrales puestos a mano (0.45–0.55) los rechazaban (más de 2 000 rechazos de destornillador en una sesión). Se recalibraron con el F1 de validación.
- **Matriz de confusión a conf 0.30** (`metricas.md`): pocas confusiones entre clases oficiales; la principal es botella→taza (16), por la botella nunca vista del test.
- **Fotos de fondo:** 313 fotos de objetos ajenos y mesa vacía en el entrenamiento, para que el modelo no asigne clases oficiales a lo que no lo es.
- **Persistencia:** elimina destellos de 1–2 cuadros; el efecto individual de cada filtro se mide con el ensayo de escenas (`scripts/ensayo_escenas.py`).

### Riesgo y mitigación

Un modelo de 5 clases tiende a asignar la clase más parecida a objetos ajenos (por ejemplo, una pluma como destornillador). Mitigación: fotos de fondo (negativos) en el dataset, umbrales por clase estrictos, filtro de persistencia temporal y calibración de umbrales con la curva precision-recall.

## Dataset y entrenamiento

Las fotos se toman con **la misma webcam de la demo**, a 640×480, en 3 o más sesiones por clase (luz, fondo, distancia y ejemplar distintos).
- `--clase mixto`: escenas con varios objetos del reto juntos.
- `--clase fondo`: escenas sin ninguno (negativos, reducen falsos positivos).

`train.py` aplica por defecto augmentation para que el modelo **generalice** a ejemplares que no vio (por ejemplo, tijeras de otro color) y aprenda la forma en vez de atajos:
- **Color:** tono muy variable (`hsv_h=0.4`), canales intercambiados y escala de grises.
- **Iluminación:** sombras artificiales, brillo y contraste.
- **Geometría:** rotación, perspectiva y volteos.
- **Cámara barata:** desenfoque, ruido, JPEG y baja resolución.

Usa Albumentations.
- Instálalo con `pip install albumentations`.
- En local, después ejecuta `pip uninstall -y opencv-python-headless` y `pip install --force-reinstall --no-deps opencv-python==5.0.0.93`. Si no, se rompen las ventanas de OpenCV.

```powershell
python scripts/capture.py --clase taza            # espacio guarda, q sale; una sesión por ejecución
python scripts/capture.py --clase mixto           # varias clases en la misma foto
python scripts/capture.py --clase fondo           # sin objetos del reto
python scripts/autolabel.py --model models/best.pt --vista   # pre-etiqueta; vistas en dataset/revision/
# etiquetar (Roboflow / LabelImg / CVAT) en formato YOLO, .txt junto a cada imagen
python scripts/split_dataset.py                   # train/val/test por sesión
python scripts/train.py --epochs 80 --device 0   # en la nube: notebooks/train_colab.ipynb (GPU)
python scripts/export.py --model runs/mechatronics/weights/best.pt
```

### Test con ejemplares no vistos

Para demostrar que el modelo reconoce objetos que nunca vio:
1. Captura uno o dos objetos físicos por clase marcándolos con `--ejemplar`.
   - Ejemplo: `python scripts/capture.py --clase tijeras --ejemplar tijeras-verdes`
2. Al dividir, resérvalos solo para test.
   - Ejemplo: `python scripts/split_dataset.py --ejemplares-test tijeras-verdes taza-azul`
3. Esos objetos van completos a test y ninguna foto suya entra a train ni a val.

### Entrenamiento en la nube

Entrenar en CPU es inviable con miles de fotos (una época tarda ~112 s en CPU, unas 11–12 horas para el entrenamiento completo). Se entrena en **GPU Tesla T4** con `notebooks/train_colab.ipynb`:
- El notebook clona el repo, copia `dataset.zip` desde Drive y ejecuta `scripts/train.py --device 0`.
- La **inferencia** sigue en CPU, como exige el reto, gracias a la exportación a OpenVINO.

### Etiquetado

1. **Pre-etiquetar:** `python scripts/autolabel.py --model models/best.pt --vista` (sin `--model` usa `models/yolo11n.pt`, que hay que descargar; ver la nota de Instalación).
   - Con `yolo11n.pt` crea los `.txt` de taza, botella, tijeras y celular.
   - En una carpeta de una sola clase, toda caja recibe la clase de la carpeta.
   - Si el modelo no detecta nada en una foto, no crea `.txt` y la lista en `dataset/raw/revisar.txt`.
   - Nunca sobrescribe un `.txt` existente, salvo con `--sobrescribir`.
   - Para clases que COCO no conoce (destornillador) o donde falla (celular visto por detrás), usar el pre-etiquetado *zero-shot* con OWLv2:
     - `python scripts/autolabel_zeroshot.py --carpeta destornillador --texto "a photo of a screwdriver" --vista`
     - Requiere `pip install transformers` y tarda ~7 s por imagen en CPU.
     - Escribe una caja por imagen y marca como "REVISAR" las de confianza baja.
2. **Corregir en Roboflow** (gratis; en el plan gratuito los proyectos son públicos):
   - Crea un proyecto *Object Detection* con las 5 clases.
   - Arrastra cada carpeta de `dataset/raw/<clase>/` con sus `.jpg`, sus `.txt` y `dataset/raw/classes.txt`. Roboflow reconoce el formato YOLO.
   - Corrige las cajas malas y dibuja las faltantes: destornillador y las fotos de `revisar.txt`.
   - Exporta en formato **YOLOv11** y descomprime el zip en `dataset/etiquetado/`.
3. **Dividir:** `python scripts/split_dataset.py --raw dataset/etiquetado`.
   - La sesión se reconoce aunque Roboflow renombre los archivos.
   - Se omiten las fotos sin etiqueta, salvo las de `fondo`.

Atajo para el destornillador:
- Etiqueta a mano unas 100 fotos y entrena un primer modelo.
- Luego pre-etiqueta el resto con `python scripts/autolabel.py --model models/best.pt --carpetas destornillador`.

Para cambiar de modelo basta con `modelo.ruta` en `config.yaml` (o `--model` al ejecutar). Un modelo
propio devuelve los nombres oficiales y se usa tal cual; `mapeo_clases` solo traduce los nombres de modelos COCO.

## Decisiones técnicas

- **YOLO11n con transfer learning:** 4 de 5 clases ya están en COCO; partir de pesos preentrenados reduce el tiempo de entrenamiento y permite converger con un dataset pequeño (~2 000 imágenes).
- **Modelo propio (`best.pt`):** resuelve la falta de destornillador en COCO y devuelve directamente los nombres oficiales sin mapeo.
- **OpenVINO por defecto:** ~47.6 FPS de inferencia del modelo final frente a ~22.5 FPS con PyTorch. Pipeline completo en vivo: 16–30 FPS.
- **Entrenamiento en la nube, inferencia en CPU:** entrenar en GPU Tesla T4 (Colab) tarda minutos; en CPU tardaría 11–12 horas. La inferencia sí corre en CPU, como exige el reto.
- **Cámara en un hilo:** siempre se procesa el frame más reciente, sin acumular retraso. Reconexión automática con mensaje en pantalla.
- **Umbral bajo en el modelo (0.25) y umbral real en los filtros:** así el trackbar puede bajar o subir el umbral en vivo.
- **Persistencia N de M con ByteTrack:** elimina falsos positivos de un solo frame. Si no hay `track_id`, empareja por IoU.
- **Split por sesión de captura:** evita fuga de datos entre train y test.
- **Exportación a ONNX (bonus):** `models/best.onnx` incluido en el repo.

## Fuentes

- [Ultralytics YOLO11](https://docs.ultralytics.com/models/yolo11/)
- [Exportación a OpenVINO](https://docs.ultralytics.com/integrations/openvino/)
- [ONNX](https://onnx.ai/) — formato de exportación interoperable.
- [Tracking (ByteTrack)](https://docs.ultralytics.com/modes/track/)
- Dataset [COCO](https://cocodataset.org/) (pesos preentrenados).
- [OWLv2](https://huggingface.co/google/owlv2-base-patch16-ensemble) (Minderer et al., 2023): pre-etiquetado zero-shot.
- [Albumentations](https://albumentations.ai/): aumento de datos.
- [Roboflow](https://roboflow.com/): revisión y corrección manual de etiquetas.
- [Google Colab](https://colab.research.google.com/): entrenamiento en la nube con GPU.

Se usaron asistentes de IA (Claude y Gemini) como apoyo en el desarrollo y la documentación; el código fue revisado y probado por el equipo.

## Limitaciones conocidas

- **Tijeras con recall bajo:** mAP50 de 0.66 y recall de 0.54. Se necesitan más fotos con variedad de tijeras (distintos colores, tamaños y posiciones).
- **Tazas sobre fondo:** 38 predicciones de "taza" sobre fondo en la matriz de confusión. Probablemente corresponden en buena parte a tazas reales no etiquetadas por el pre-etiquetado automático. Se corregirá con revisión manual.
- **Test por bloques:** salvo la botella (ejemplar nunca visto), el test se divide por bloques dentro de cada sesión (70/15/15), no por sesión completa; las métricas pueden ser algo optimistas para lugares totalmente nuevos.
- **Faltan fotos de fondo y botella:** la captura de fotos de fondo (objetos ajenos y mesa vacía), botellas propias y destornilladores sobre la mesa está en curso.
- **Riesgo con objetos ajenos:** un modelo de 5 clases puede asignar la clase más parecida a objetos ajenos (por ejemplo, una pluma como destornillador). La mitigación está en los filtros (persistencia, umbral por clase) y en las fotos de fondo del modelo v1.
- La primera inferencia con OpenVINO tarda unos 10 s (compilación). Se hace al iniciar, antes de abrir la ventana.
- Los FPS dependen de la CPU y de la resolución de la cámara.

## Mejoras propuestas

- **Modelo v1** con fotos de fondo (negativos), botellas propias y destornilladores sobre la mesa.
- **Revisión manual de etiquetas** en Roboflow, especialmente las marcadas como "REVISAR".
- **División por sesión completa** (en vez de por bloques) para métricas más realistas.
- **Calibración de umbrales** por clase con la curva precision-recall del conjunto de validación.
- **Cuantización INT8** con OpenVINO (NNCF) para ganar FPS.
- Activar el verificador CNN (bonus) para las clases con más confusión.
