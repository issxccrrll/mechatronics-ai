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
| OpenVINO   |                          ~62  |                        17.8–29.8 |

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
| taza           |         505 |               475 |
| tijeras        |         200 |               195 |
| destornillador |         834 |               618 |
| celular        |         549 |               548 |
| mixto          |         200 |               200 |
| **Total**      |   **2 288** |         **2 036** |

Las imágenes sin archivo `.txt` se excluyen del entrenamiento: un `.txt` vacío le enseñaría al
modelo que no hay ningún objeto en la imagen.

### División del dataset (modelo v0)

El split se realiza con `scripts/split_dataset.py --bloques`: cada sesión de captura se divide en
bloques consecutivos 70/15/15 (train/val/test). Las fotos seguidas de una misma sesión quedan en el
mismo bloque, para reducir la fuga de información entre train y test. Total: **1 432** imágenes de
train, **317** de val y **287** de test.

| clase          | train | val | test |
|----------------|------:|----:|-----:|
| celular        |   385 |  83 |   80 |
| destornillador |   442 |  98 |   78 |
| mixto          |   140 |  30 |   30 |
| taza           |   327 |  76 |   72 |
| tijeras        |   138 |  30 |   27 |

Para el modelo final se planea dividir por sesión completa (más estricto, sin compartir fotos de
una misma sesión entre splits).

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

3. **Revisión humana** — Las cajas marcadas como "REVISAR" se verifican manualmente en Roboflow.
   Esta etapa está en curso para el modelo final.

### Trabajo pendiente de captura

- Fotos propias de botella de plástico (actualmente solo aparece en las fotos de mixto).
- Fotos de fondo: objetos ajenos y mesa vacía, sin etiquetas, para enseñar al modelo a no
  detectar clases oficiales donde no las hay.
- Destornilladores apoyados sobre la mesa en posición horizontal.

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

### Hiperparámetros principales (modelo v0)

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

En la laptop del equipo, una época con 427 imágenes tarda 112 segundos en CPU (unas 11–12 horas
para el entrenamiento completo). Por eso se entrena en la nube. La **inferencia** sí corre en
CPU, como exige el reto, gracias a la exportación a OpenVINO.

---

## 5. Técnica contra falsos positivos

El pipeline aplica 4 filtros en cascada (patrón Strategy, cada uno activable en `config.yaml`)
definidos en `src/filters.py`:

### 1. UnknownRejectFilter

Cualquier clase que no sea una de las 5 oficiales pasa a estado `desconocido`. Nunca recibe una
clase oficial. Esto garantiza la regla del reto de no asignar clases oficiales a objetos ajenos.

### 2. ClassThresholdFilter

Umbral de confianza individual por clase, nunca inferior al umbral global (0.40, ajustable en
vivo con un trackbar):

| Clase          | Umbral |
|----------------|-------:|
| taza           |   0.45 |
| botella        |   0.45 |
| tijeras        |   0.50 |
| destornillador |   0.55 |
| celular        |   0.50 |

Las clases que más se confunden (destornillador) llevan un umbral más estricto.

### 3. AreaAspectFilter

Rechaza cajas con:
- Área menor al 0.2% o mayor al 90% del frame.
- Relación de aspecto (ancho/alto) fuera del rango 0.15–6.0.

Esto elimina cajas físicamente imposibles: demasiado pequeñas para ser un objeto real, o con
proporciones que no corresponden a ninguna de las 5 clases.

### 4. PersistenceFilter

Un objeto solo se acepta si aparece en al menos **3 de los últimos 5 frames**. Usa el
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
- Los umbrales se calibrarán con la curva precision-recall del conjunto de validación.

---

## 6. Resultados

Modelo: `models/best.pt` (YOLO11n, transfer learning desde COCO, 80 épocas en Tesla T4).
Evaluado sobre el split de **test** (287 imágenes separadas de train/val).

Reproducir: `python scripts/eval.py --model models/best.pt`

### Resultados por clase (protocolo estándar, conf 0.001)

| clase          | precision | recall | mAP50  | mAP50-95 |
|----------------|----------:|-------:|-------:|---------:|
| todas          |    0.8562 | 0.8064 | 0.8417 |   0.6619 |
| taza           |    0.6489 | 0.8306 | 0.7588 |   0.6505 |
| botella        |    0.9655 | 0.9286 | 0.9907 |   0.6628 |
| tijeras        |    0.8392 | 0.5402 | 0.6608 |   0.5261 |
| destornillador |    0.8908 | 0.7692 | 0.8487 |   0.5966 |
| celular        |    0.9365 | 0.9634 | 0.9493 |   0.8735 |

### Precision y recall al umbral de operación (conf 0.40)

| clase          | precision | recall |
|----------------|----------:|-------:|
| todas          |    0.8743 | 0.7807 |
| taza           |    0.6422 | 0.7865 |
| botella        |    1.0000 | 0.8929 |
| tijeras        |    0.8824 | 0.5172 |
| destornillador |    0.9062 | 0.7436 |
| celular        |    0.9405 | 0.9634 |

### Matriz de confusión (conf 0.40)

Filas = predicción, columnas = clase real; "fondo" = sin objeto.

| pred \ real    | taza | botella | tijeras | destornillador | celular | fondo |
|----------------|-----:|--------:|--------:|---------------:|--------:|------:|
| taza           |   70 |       1 |       0 |              0 |       0 |    38 |
| botella        |    0 |      24 |       0 |              0 |       0 |     1 |
| tijeras        |    0 |       0 |      15 |              0 |       0 |     2 |
| destornillador |    0 |       0 |       0 |             59 |       0 |     5 |
| celular        |    0 |       0 |       1 |              0 |      79 |     4 |
| fondo          |   19 |       3 |      13 |             19 |       3 |     0 |

### Lectura de resultados

**Clases fuertes:** botella (mAP50 0.99, precision 1.0 a conf 0.40) y celular (mAP50 0.95,
recall 0.96) son las más confiables. Destornillador (mAP50 0.85) también tiene buen rendimiento.

**Clases a mejorar:** tijeras tiene el recall más bajo (0.54 estándar, 0.52 a conf 0.40) y
mAP50 de 0.66; se necesitan más fotos con variedad de tijeras. Taza tiene precision baja
(0.64): las 38 predicciones de "taza sobre fondo" probablemente corresponden en buena parte a
tazas reales que el pre-etiquetado automático no marcó en escenas con varias tazas; esto se
corregirá con la revisión manual de etiquetas.

Casi no hay confusiones **entre clases oficiales** (solo 1 botella predicha como taza y 1
tijeras predicha como celular). Los errores principales son predicciones sobre fondo y objetos
no detectados.

**Nota:** el test del modelo v0 se dividió por bloques dentro de cada sesión (70/15/15). Las
métricas pueden ser algo optimistas respecto a un escenario real. El modelo final se evaluará
con división por sesión completa.

### Rendimiento en CPU (Intel Core i7-13620H)

| Medición                              | FPS           |
|---------------------------------------|---------------|
| Inferencia OpenVINO (modelo propio)   | ~32–35        |
| Pipeline completo en vivo             | 17.8–29.8     |
| Requisito mínimo del reto             | 15            |

Comando: `python scripts/benchmark_fps.py --model models/best_openvino_model`

---

## 7. Decisiones principales del equipo

- **YOLO11n con transfer learning** en lugar de entrenar desde cero: 4 de 5 clases ya están en
  COCO, y el modelo converge rápido con pocas imágenes propias.
- **OpenVINO para la inferencia en CPU:** triplica los FPS respecto a PyTorch (~62 vs ~22.5),
  permitiendo superar los 15 FPS requeridos con holgura.
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
