# Respuestas del reto Mechatronics AI

## 1. Modelo elegido

**YOLO11n** (Ultralytics), la variante más ligera de la familia YOLO11, con **transfer learning**
desde pesos preentrenados en COCO (`yolo11n.pt`).

**¿Por qué YOLO11n?**

- Cumple el requisito de **15 FPS o más en CPU** sin necesidad de GPU dedicada.
- Detecta **varios objetos por frame** en una sola pasada, ideal para las escenas del jurado con
  múltiples clases simultáneas.
- El transfer learning permite entrenar con un dataset pequeño (unas 2 000 imágenes propias) y
  converger rápido en GPU gratuita (Google Colab, Tesla T4).

**¿Por qué no entrenar desde cero?** COCO ya cubre 4 de las 5 clases (cup, bottle, scissors,
cell phone). Solo falta destornillador, que COCO no incluye. Partir de pesos preentrenados
aprovecha ese conocimiento y reduce drásticamente el tiempo de entrenamiento.

Antes de entrenar un modelo propio, el sistema funcionaba con el YOLO11n de COCO y un mapeo de
clases en `config.yaml` (cup→taza, bottle→botella, scissors→tijeras, cell phone→celular). Esto
permitió desarrollar y probar todo el pipeline mientras se preparaba el dataset.

**Inferencia con OpenVINO.** El modelo `.pt` se exporta a formato OpenVINO para acelerar la
inferencia en CPU. Comparación medida en la laptop del equipo:

| Motor      | FPS de inferencia (imgsz 640) | FPS en pipeline completo (en vivo) |
|------------|------------------------------:|-----------------------------------:|
| PyTorch    |                         ~22.5 |                                 — |
| OpenVINO   |                         ~47.6 | 16–30 (25.9 con la laptop libre) |

El pipeline completo incluye captura de cámara, detección, verificación, filtros y dibujo en
pantalla, por lo que el FPS real es menor que el de inferencia pura. Aun así supera los 15 FPS
requeridos con holgura.

---

## 2. Dataset y etiquetado

### Captura de imágenes

Todas las fotos fueron tomadas con la webcam del equipo a 640×480 usando el script
`scripts/capture.py`. Se varió intencionalmente la iluminación, el fondo, la distancia y la
posición de cada objeto. Los archivos se nombran con la fecha y hora de la sesión de captura
(`sAAAAMMDD_HHMMSS_NNNN.jpg`), lo que permite agrupar por sesión al dividir el dataset.

### Fotos por carpeta

| Carpeta        | Fotos (jpg) | Etiquetadas (txt) |
|----------------|------------:|------------------:|
| botella        |         276 |               276 |
| celular        |         623 |               579 |
| destornillador |       1 013 |               940 |
| fondo          |         313 |      313 (vacías) |
| mixto          |         299 |               203 |
| taza           |         584 |               581 |
| tijeras        |         328 |               325 |
| **Total**      |   **3 436** |         **3 217** |

Son 25 sesiones de captura. Las fotos de **fondo** (objetos ajenos y mesa vacía) llevan una etiqueta
vacía a propósito: le enseñan al modelo qué no es un objeto. Las imágenes sin archivo `.txt` se excluyen del entrenamiento: un `.txt` vacío le enseñaría al
modelo que no hay ningún objeto en la imagen.

### División del dataset (modelo final v2)

`scripts/split_dataset.py --bloques --ejemplares-test botella-test`:
- Cada sesión se divide en bloques consecutivos 70/15/15 (train/val/test); las fotos seguidas
  de una sesión quedan en el mismo bloque, para reducir la fuga entre train y test.
- **Excepción, para medir generalización:** una botella física (`botella-test`) se fotografió
  aparte y va **solo a test**; el modelo nunca la vio al entrenar.
- Total: **2 229** imágenes de train, **517** de val y **471** de test (incluido en el repo, en `dataset/test/`).

| clase          | train | val | test |
|----------------|------:|----:|-----:|
| botella        |   150 |  73 |   53 |
| celular        |   410 |  86 |   83 |
| destornillador |   669 | 144 |  127 |
| fondo          |   221 |  47 |   45 |
| mixto          |   141 |  30 |   32 |
| taza           |   407 |  88 |   86 |
| tijeras        |   231 |  49 |   45 |

### Etiquetado semiautomático (formato YOLO)

El etiquetado combina detección automática con revisión humana en tres etapas:

1. **`scripts/autolabel.py`** — Pre-etiquetado con YOLO11n COCO en varias pasadas:
   - Normal (conf 0.30) → umbral bajo (0.10, mejor caja) → TTA (multiescala y volteo) → imagen
     rotada 180° (objetos boca abajo).
   - En una carpeta de una sola clase, si el modelo detecta otra clase se reasigna a la de la
     carpeta solo cuando la clase correcta no fue detectada (por ejemplo, una taza boca abajo
     detectada como "botella").
   - Descarta cajas diminutas en las pasadas de umbral bajo.

2. **`scripts/autolabel_zeroshot.py`** — Para clases que COCO no conoce (destornillador) o donde
   falla (celular visto de espaldas), se usa un detector de vocabulario abierto **OWLv2**
   (`google/owlv2-base-patch16-ensemble`) con texto como prompt ("a photo of a screwdriver").
   - Se evaluaron previamente **YOLOE** (0 de 40 aciertos) y **YOLO-World** (7 de 40). Ninguno
     sirvió para destornilladores; OWLv2 fue el único que dio resultados aceptables.

3. **Revisión humana** — Las propuestas automáticas se revisaron foto por foto: de 474 cajas
   propuestas por el modelo v0 en las fotos nuevas, se aprobaron 141 y se descartaron 333 (cajas
   sobre objetos equivocados). Las 694 fotos restantes se etiquetaron a mano en Roboflow
   (`scripts/integrar_roboflow.py` las trae al dataset traduciendo clases y nombres).

---

## 3. Aumento de datos

El aumento busca que el modelo **generalice** a ejemplares no vistos (por ejemplo, tijeras de
otro color) y aprenda la **forma** del objeto en vez de atajos como el color o el fondo.

### Aumentos nativos de Ultralytics

| Parámetro     | Valor  | Objetivo                                              |
|---------------|-------:|-------------------------------------------------------|
| `hsv_h`       |   0.40 | Varía mucho el tono (default 0.015); fuerza invarianza al color |
| `hsv_s`       |   0.70 | Variación de saturación                                |
| `hsv_v`       |   0.50 | Variación de brillo                                    |
| `degrees`     |     15 | Rotación moderada                                      |
| `perspective` | 0.0005 | Distorsión de perspectiva leve                         |
| `flipud`      |   0.20 | Volteo vertical (objetos boca abajo)                   |
| `fliplr`      |   0.50 | Volteo horizontal                                      |
| `mosaic`      |   1.00 | Mosaico de 4 imágenes durante el entrenamiento         |
| `close_mosaic` |    10 | Desactiva mosaico las últimas 10 épocas para afinar   |

### Aumentos con Albumentations

| Técnica                    | Probabilidad | Objetivo                                |
|----------------------------|-------------:|-----------------------------------------|
| `ChannelShuffle`           |         0.15 | Invarianza al orden de canales de color |
| `ToGray`                   |         0.15 | Funcionar en escala de grises           |
| `RandomShadow`             |         0.25 | Sombras parciales                       |
| `RandomBrightnessContrast` |         0.30 | Cambios de iluminación                  |
| Motion/Gaussian/Defocus blur |       0.30 | Simular cámara en movimiento o desenfocada |
| Gauss/ISO noise            |         0.30 | Ruido de sensor                         |
| `ImageCompression` (q 30–80) |       0.30 | Simular compresión JPEG de baja calidad |
| `Downscale`                |         0.20 | Simular baja resolución                 |

---

## 4. Hiperparámetros y entrenamiento

### Hiperparámetros principales (modelo final v2)

| Parámetro   | Valor          | Notas                                    |
|-------------|----------------|------------------------------------------|
| Pesos       | `yolo11n.pt`   | Transfer learning desde COCO             |
| `epochs`    | 80             | Con parada temprana (`patience=20`)       |
| `imgsz`     | 640            | Resolución de entrada                    |
| `batch`     | -1 (auto)      | Se ajusta según la GPU disponible        |
| `lr0`       | 0.01           | Tasa de aprendizaje inicial              |
| `patience`  | 20             | Épocas sin mejora para detener           |

### Dónde y cómo se entrena

El entrenamiento se realiza en **GPU Tesla T4 en Google Colab** usando el notebook
`notebooks/train_colab.ipynb`, que ejecuta `scripts/train.py`.

En una prueba en la laptop del equipo, una época con 427 imágenes tardó 112 segundos en CPU; con
las 2 229 imágenes del v2 serían horas. En la T4, las 80 épocas tardaron alrededor de 1 hora. Por eso se entrena en la nube. La **inferencia** sí corre en
CPU, como exige el reto, gracias a la exportación a OpenVINO.

---

## 5. Técnica contra falsos positivos

El pipeline aplica 4 filtros en cascada (patrón Strategy, cada uno activable en `config.yaml`)
definidos en `src/filters.py`:

### 1. UnknownRejectFilter

Cualquier clase que no sea una de las 5 oficiales pasa a estado `desconocido`. Nunca recibe una
clase oficial. Esto garantiza la regla del reto de no asignar clases oficiales a objetos ajenos.

### 2. ClassThresholdFilter

Umbral de confianza individual por clase, nunca inferior al umbral global (0.30, ajustable en
vivo con un trackbar):

| Clase          | Umbral |
|----------------|-------:|
| taza           |   0.30 |
| botella        |   0.33 |
| tijeras        |   0.50 |
| destornillador |   0.45 |
| celular        |   0.30 |

Los umbrales se **calibraron con datos** (`scripts/calibrar_umbrales.py`): para cada clase, el
umbral más bajo cuyo F1 en validación queda a menos de 0.02 del máximo. El de tijeras salió 0.65 y
se limitó a 0.50, porque validación casi no tiene tijeras en la mano. El de destornillador se subió a 0.45 tras la prueba en vivo, donde aparecían
detecciones falsas en el fondo: en validación su precisión pasa de 0.78 a 0.87 y su recall de 0.78 a 0.71. El motivo de calibrar fue una
prueba en vivo: el registro CSV mostró que los objetos en la mano se detectaban con confianza
0.34–0.40 y los umbrales anteriores (0.45–0.55, puestos a mano) los descartaban.

### 3. AreaAspectFilter

Rechaza cajas con:
- Área menor al 0.2% o mayor al 90% del frame.
- Relación de aspecto (ancho/alto) fuera del rango 0.15–6.0.

Esto elimina cajas físicamente imposibles: demasiado pequeñas para ser un objeto real, o con
proporciones que no corresponden a ninguna de las 5 clases.

### 4. PersistenceFilter

Un objeto solo se acepta si aparece en al menos **4 de los últimos 5 frames**. Usa el
`track_id` de **ByteTrack** (tracking integrado de Ultralytics). Si no hay ID, empareja por IoU
de 0.4 o más. Elimina los falsos positivos más comunes en video: detecciones de 1–2 frames por
ruido, movimiento o reflejos.

### Configuración del detector

El detector usa un umbral interno bajo (0.25) para que el umbral real lo decidan los filtros y
pueda ajustarse en vivo con el trackbar.

### Riesgo identificado y mitigación

Un modelo entrenado solo con 5 clases tiende a asignar la clase más parecida a objetos ajenos
(por ejemplo, una pluma como destornillador). La mitigación consiste en:

- Fotos de fondo (negativos) en el dataset.
- Umbrales por clase estrictos.
- Filtro de persistencia temporal.
- Umbrales calibrados con el F1 del conjunto de validación.

---

## 6. Resultados

Modelo final: `models/best.pt` (v2; YOLO11n, transfer learning desde COCO, 80 épocas en Tesla T4).
Evaluado sobre el split de **test** (471 imágenes separadas de train/val, con una botella que el
modelo nunca vio). La última columna es el modelo anterior (v0) en **el mismo test**.

Reproducir: `python scripts/eval.py --model models/best.pt`

| clase          | precision | recall | mAP50  | mAP50-95 | mAP50 del v0 |
|----------------|----------:|-------:|-------:|---------:|-------------:|
| **todas**      |    0.7799 | 0.6951 | **0.7404** | 0.5362 | 0.626 |
| taza           |    0.6485 | 0.7566 | 0.7467 |   0.6113 | 0.676 |
| botella        |    0.7543 | 0.4380 | 0.5032 |   0.2051 | 0.325 |
| tijeras        |    0.7041 | 0.7083 | 0.7415 |   0.4667 | 0.588 |
| destornillador |    0.8815 | 0.6760 | 0.7839 |   0.5454 | 0.614 |
| celular        |    0.9109 | 0.8966 | 0.9267 |   0.8526 | 0.926 |

**El v2 mejora en todas las clases: mAP50 de 0.626 a 0.740 (+11.4 puntos, +18%).** Las mayores
subidas son botella (+17.8), destornillador (+17.0) y tijeras (+15.3), justo donde se agregaron
fotos en la mano, sobre la mesa y de fondo.

### Matriz de confusión (conf 0.30, el umbral de operación)

Filas = predicción, columnas = clase real; "fondo" = sin objeto.

| pred \ real    | taza | botella | tijeras | destornillador | celular | fondo |
|----------------|-----:|--------:|--------:|---------------:|--------:|------:|
| taza           |   79 |      16 |       0 |              0 |       1 |    28 |
| botella        |    0 |      61 |       0 |              0 |       1 |    26 |
| tijeras        |    0 |       0 |      35 |              0 |       0 |    16 |
| destornillador |    0 |       0 |       3 |            108 |       0 |    12 |
| celular        |    0 |       0 |       0 |              0 |      78 |    13 |
| fondo          |   21 |      44 |      10 |             35 |       7 |     0 |

### Lectura de resultados

- **Clases fuertes:** celular (mAP50 0.93), destornillador (0.78), taza (0.75) y tijeras (0.74).
- **Clase más difícil: botella (0.50).** Su test es la botella que el modelo **nunca vio**; 44 no
  se detectan y 16 se confunden con taza. Muestra el límite de generalizar a un ejemplar nuevo
  con pocas botellas distintas en entrenamiento; la mejora es fotografiar más tipos de botella.
- **Pocas confusiones entre clases** salvo botella→taza (16) y tijeras→destornillador (3).
- **Detecciones sobre fondo:** 28 de taza y 26 de botella son las más altas; parte corresponde a
  objetos reales sin etiquetar en escenas con varios objetos.
- **Límite del test:** salvo la botella, el test comparte sesiones con train (bloques), así que
  puede ser algo optimista para lugares totalmente nuevos.

### Rendimiento en CPU (Intel Core i7-13620H)

| Medición                                   | FPS |
|--------------------------------------------|-----|
| Inferencia OpenVINO del modelo final       | ~47.6 (benchmark) |
| Inferencia PyTorch (`.pt`)                 | ~22.5 |
| Pipeline completo en vivo (cámara + todo)  | 16–30 según la carga de la laptop; 25.9 con la interfaz final y la laptop libre |
| Requisito mínimo del reto                  | 15 |

Comando: `python scripts/benchmark_fps.py --model models/best_openvino_model`

---

## 7. Decisiones principales del equipo

- **YOLO11n con transfer learning** en lugar de entrenar desde cero: 4 de 5 clases ya están en
  COCO, y el modelo converge rápido con pocas imágenes propias.
- **OpenVINO para la inferencia en CPU:** duplica los FPS de inferencia respecto a PyTorch
  (~47.6 frente a ~22.5 con el modelo final), con margen sobre los 15 FPS requeridos.
- **Cámara en un hilo separado** que siempre entrega el frame más reciente, con reconexión
  automática y mensaje en pantalla cuando se pierde la señal.
- **División por sesión o bloques consecutivos** en vez de al azar, para evitar que fotos casi
  idénticas de la misma sesión caigan en train y test, inflando las métricas artificialmente.
- **Entrenamiento en la nube (Colab)** e inferencia en CPU: el entrenamiento tomaría 11–12 horas
  en CPU; en una T4 tarda minutos.
- **Etiquetado semiautomático con revisión humana:** tres etapas (YOLO COCO, OWLv2 zero-shot,
  revisión manual) para cubrir las 5 clases con calidad.
- **Aumento de datos orientado a generalizar:** cambios agresivos de color, sombras y simulación
  de cámara de baja calidad, para que el modelo no dependa de un fondo o iluminación específica.
- **Funciones extra en pantalla:** contador de objetos detectados por clase e ID de seguimiento
  visible, útiles para depuración y para la demo en vivo.

---

## 8. Fuentes y herramientas

- **Ultralytics YOLO11** — Detección de objetos (modelo y entrenamiento).
- **ByteTrack** — Tracking multi-objeto integrado en Ultralytics.
- **OpenVINO** (Intel) — Optimización de inferencia en CPU.
- **COCO** — Dataset de preentrenamiento.
- **OWLv2** (Minderer et al., 2023) — Detección zero-shot para el etiquetado de destornilladores.
- **Albumentations** — Aumento de datos avanzado.
- **Roboflow** — Revisión y corrección manual de etiquetas.
- **Google Colab** (Tesla T4) — Entrenamiento en la nube.

Se usaron asistentes de IA (Claude y Gemini) como apoyo en el desarrollo y la documentación;
el código fue revisado y probado por el equipo.
