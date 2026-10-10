# Contexto para Gemini: redactar respuestas.md y pulir el script de ensayo

Son **dos tareas de documentación y limpieza**, en archivos que nadie más está tocando. Sigue las
instrucciones al pie de la letra. **No inventes datos:** usa solo los hechos de este documento y
lo que leas en el código. Lo que todavía no existe (las métricas del modelo) se deja marcado como
pendiente. Si algo no coincide, **DETENTE y repórtalo**.

---

## 0. Requisito previo y dónde trabajar

```powershell
git -C C:\Users\romer\MatchVision-gemini branch --show-current
git -C C:\Users\romer\MatchVision-gemini status --short
```
- La rama debe ser `gemini` y el `status` debe salir vacío. Si no, DETENTE.
- **Trabaja SOLO en `C:\Users\romer\MatchVision-gemini`.** Nunca en `C:\Users\romer\MatchVision`, que es de Claude.
- **Python:** `C:\Users\romer\MatchVision\venv\Scripts\python.exe`, con ruta completa.

---

## TAREA 1: Redactar `respuestas.md` (vale 8 pts de la rúbrica)

### 1.1 Qué pide la ficha del reto
> **respuestas.md:** explicación del modelo elegido, uso de modelos preentrenados o entrenamiento
> propio, técnica contra falsos positivos y decisiones principales del equipo.

Y la rúbrica (bloque C, 8 pts):
> Describir el modelo, el etiquetado, el aumento de datos y los hiperparámetros principales.

**Reemplaza por completo** el `respuestas.md` actual (una plantilla con TODO) siguiendo la
estructura de la sección 1.3. Es un documento para el **jurado**: claro, en español, en
párrafos cortos y tablas. Sin relleno, sin exagerar y sin emojis. Unas 150–250 líneas.

### 1.2 Hechos del proyecto (fuente de verdad: no contradigas nada de esto)

**Reto:** detectar en vivo 5 clases (taza, botella, tijeras, destornillador, celular) con CPU a
15 FPS o más. Los objetos ajenos se marcan como `desconocido` o se ignoran, nunca con una clase oficial.

**Modelo**
- **YOLO11n** (Ultralytics), el más ligero de la familia, con **transfer learning** desde pesos preentrenados en COCO.
- **Por qué YOLO11n:** cumple 15 FPS o más en CPU, entrena rápido en GPU gratuita (Colab) y detecta varios objetos por frame.
- **Inferencia con OpenVINO** (exportado desde el `.pt`): ~62 FPS de inferencia en la laptop del equipo, frente a ~22.5 FPS con PyTorch (imgsz 640). Con el pipeline completo en vivo se midieron entre **17.8 y 29.8 FPS**.
- **Antes de entrenar**, el sistema usaba yolo11n de COCO con un mapeo de clases en `config.yaml` (cup→taza, bottle→botella, scissors→tijeras, cell phone→celular). **COCO no tiene destornillador**, por eso se entrena un modelo propio.

**Dataset (fotos propias con la webcam del equipo)**
- **Captura:** `scripts/capture.py`, a 640×480, organizado por sesiones (el nombre del archivo incluye fecha y hora de la sesión), con variación de luz, fondo, distancia y posición.
- **Fotos por carpeta:** taza 505, tijeras 200, destornillador 834, celular 549, mixto 200 (varios objetos juntos). Total **2 288**, de las cuales **2 036** tienen etiquetas.
- **División del modelo v0:** train 1 432, val 317, test 287 imágenes.
  - Método: bloques consecutivos 70/15/15 dentro de cada sesión (`split_dataset.py --bloques`).
  - Las fotos seguidas de una sesión quedan en el mismo bloque, para reducir fuga entre train y test.
  - Para el modelo final se planea dividir **por sesión completa** (más estricto).

| clase | train | val | test |
|---|---|---|---|
| celular | 385 | 83 | 80 |
| destornillador | 442 | 98 | 78 |
| mixto | 140 | 30 | 30 |
| taza | 327 | 76 | 72 |
| tijeras | 138 | 30 | 27 |

- **Pendiente de captura** (indícalo como trabajo en curso): fotos propias de botella de plástico, fotos de `fondo` (objetos ajenos y mesa vacía, sin etiquetas, para reducir falsos positivos) y destornilladores apoyados en la mesa.

**Etiquetado (formato YOLO), semiautomático**
1. `scripts/autolabel.py`: pre-etiqueta con yolo11n COCO en varias pasadas.
   - Pasadas: normal (conf 0.30) → umbral bajo (0.10, mejor caja) → TTA (multiescala y volteo) → imagen rotada 180° (objetos boca abajo).
   - En una carpeta de una sola clase, una caja de otra clase se reasigna a la de la carpeta solo si el modelo no vio la clase correcta (por ejemplo, una taza boca abajo detectada como "botella").
   - Descarta cajas diminutas en las pasadas de umbral bajo.
2. `scripts/autolabel_zeroshot.py`: para clases que COCO no conoce (destornillador) o donde falla (celular visto por detrás), detector de vocabulario abierto **OWLv2** (google/owlv2-base-patch16-ensemble) con texto ("a photo of a screwdriver").
   - Antes se probaron **YOLOE** (0 de 40 aciertos) y **YOLO-World** (7 de 40), que no sirvieron para destornilladores.
3. **Revisión humana** de las cajas dudosas (marcadas "REVISAR") en Roboflow. Indícala como paso en curso para el modelo final.

Resultado del pre-etiquetado:

| clase | fotos | etiquetadas |
|---|---|---|
| taza | 505 | 475 |
| tijeras | 200 | 195 |
| destornillador | 834 | 618 |
| celular | 549 | 548 |
| mixto | 200 | 200 |

Las imágenes sin etiqueta se **excluyen** del entrenamiento: un `.txt` vacío le enseñaría al modelo que no hay objeto.

**Aumento de datos** (`scripts/train.py`): busca que el modelo **generalice** a ejemplares no vistos (por ejemplo, tijeras de otro color) y aprenda la **forma** en vez de atajos.
- **Nativo de Ultralytics:**
  - Color: `hsv_h=0.4` (el default es 0.015; cambia mucho el tono), `hsv_s=0.7`, `hsv_v=0.5`.
  - Geometría: `degrees=15`, `perspective=0.0005`, `flipud=0.2`, `fliplr=0.5`.
  - Mosaico: `mosaic=1.0`, `close_mosaic=10`.
- **Albumentations:**
  - Color: `ChannelShuffle` (p 0.15) y `ToGray` (p 0.15).
  - Sombras e iluminación: `RandomShadow` (p 0.25) y `RandomBrightnessContrast` (p 0.3).
  - Cámara de baja calidad: Motion/Gaussian blur o Defocus (p 0.3), Gauss o ISO noise (p 0.3), `ImageCompression` calidad 30–80 (p 0.3) y `Downscale` (p 0.2).

**Hiperparámetros principales (entrenamiento v0)**
- Pesos iniciales `yolo11n.pt`, `epochs=80`, `imgsz=640`, `batch=-1` (automático según la GPU), `lr0=0.01`, `patience=20` (parada temprana).
- GPU **Tesla T4 en Google Colab** (`notebooks/train_colab.ipynb`, que ejecuta `scripts/train.py`).
- En la laptop, 1 época con 427 imágenes tardó 112 s en CPU (unas 11–12 h el entrenamiento completo). Por eso se entrena en la nube y la **inferencia** sí corre en CPU, como exige el reto.

**Técnica contra falsos positivos** (`src/filters.py`, patrón Strategy, cada filtro activable en `config.yaml`, en este orden):
1. **UnknownRejectFilter:** cualquier clase que no sea oficial pasa a `desconocido`; nunca recibe clase oficial.
2. **ClassThresholdFilter:** umbral por clase (taza 0.45, botella 0.45, tijeras 0.50, destornillador 0.55, celular 0.50), nunca por debajo del umbral global (0.40, ajustable en vivo con un trackbar).
3. **AreaAspectFilter:** rechaza cajas con área menor a 0.2% o mayor a 90% del frame, o con relación de aspecto (ancho/alto) fuera de 0.15–6.0.
4. **PersistenceFilter:** un objeto solo se acepta si aparece en al menos **3 de los últimos 5 frames**. Usa el `track_id` de **ByteTrack**; sin ID, empareja por IoU de 0.4 o más. Elimina destellos de un solo frame.
- El detector usa un umbral interno bajo (0.25) para que el umbral real lo decidan los filtros y se pueda ajustar en vivo.
- **Justificación:**
  - La persistencia ataca el falso positivo más común en video: detecciones de 1–2 frames por ruido, movimiento o reflejos.
  - El umbral por clase permite ser más estricto con las clases que más se confunden (destornillador).
  - El filtro de área y aspecto descarta cajas físicamente imposibles.
  - El rechazo de desconocidos garantiza la regla del reto.
- **Riesgo identificado y mitigación:** un modelo entrenado solo con 5 clases tiende a asignar la clase más parecida a objetos ajenos (por ejemplo, una pluma como destornillador). La mitigación son las fotos de `fondo` (negativos), los umbrales por clase y la persistencia. Los umbrales se calibrarán con la curva precision-recall del conjunto de validación.

**Decisiones principales del equipo** (redáctalas como lista breve con su porqué):
- YOLO11n con transfer learning en lugar de entrenar desde cero.
- OpenVINO para la inferencia en CPU.
- Cámara en un hilo separado que siempre entrega el frame más reciente, con reconexión automática y mensaje en pantalla.
- División por sesión o bloques en vez de al azar.
- Entrenamiento en la nube (Colab) e inferencia en CPU.
- Etiquetado semiautomático con revisión humana.
- Aumento de datos orientado a generalizar (color, sombras, cámara de baja calidad).
- Funciones extra: contador de objetos por clase y ID de seguimiento en pantalla.

**Métricas: NO LAS INVENTES.** El modelo v0 se está entrenando ahora. En la sección de resultados
escribe exactamente: `PENDIENTE: se completa con scripts/eval.py tras entrenar el modelo v0 (ver metricas.md).`

### 1.3 Estructura obligatoria de `respuestas.md`
```
# Respuestas del reto Mechatronics AI
## 1. Modelo elegido
   (qué modelo, preentrenado + transfer learning, por qué; tabla de rendimiento CPU: PyTorch vs OpenVINO)
## 2. Dataset y etiquetado
   (captura, tablas de fotos/etiquetas y split v0, proceso semiautomático, revisión humana, pendientes)
## 3. Aumento de datos
   (tabla técnica → objetivo)
## 4. Hiperparámetros y entrenamiento
   (tabla; dónde y cómo se entrena; por qué en la nube)
## 5. Técnica contra falsos positivos
   (los 4 filtros, valores, justificación, riesgo y mitigación)
## 6. Resultados
   (PENDIENTE..., tal cual)
## 7. Decisiones principales del equipo
## 8. Fuentes y herramientas
```
Para la sección 8 incluye: Ultralytics YOLO11, ByteTrack (tracking de Ultralytics), OpenVINO,
COCO, OWLv2 (Minderer et al., 2023), Albumentations, Roboflow, Google Colab, y la frase:
"Se usaron asistentes de IA (Claude y Gemini) como apoyo en el desarrollo y la documentación;
el código fue revisado y probado por el equipo."

### 1.4 Verificación de la Tarea 1
- Abre el archivo y confirma que no quedan "TODO".
- Que estén las 8 secciones.
- Que **ningún número** contradiga la sección 1.2.
- Que la sección 6 diga PENDIENTE.

---

## TAREA 2: Pulir `scripts/ensayo_escenas.py` (el script que hiciste antes)

Claude revisó tu script y funciona bien. Aplica **solo** estos 5 ajustes:
1. **Duplicación:** el bloque detectar → verificar → filtrar → dibujar está escrito dos veces. Extráelo a una función auxiliar dentro de `main()` (o de módulo) y úsala en ambos lugares.
2. **Import sobrante:** quita `import numpy as np`, que no se usa.
3. **CSV para Excel:** abre el CSV con `encoding="utf-8-sig"`, para que Excel muestre bien los acentos.
4. **Cámara desconectada:** cuando `camara.leer()` devuelve `None` y `camara.estado != "ok"`, muestra en la ventana un mensaje (por ejemplo, `ui.pantalla_estado("Sin señal. Reintentando conexión...")` y `cv2.imshow`) en vez de dejarla congelada. Debe poder salirse con `q`.
5. **Largo:** si con lo anterior puedes simplificar sin cambiar el comportamiento, hazlo. No es obligatorio llegar a 170 líneas.

**No cambies** las 20 escenas, las funciones puras (`conteo_aceptadas`, `moda_conteos`, `es_acierto`, `resumen`) ni el formato del CSV.

### Verificación de la Tarea 2
```powershell
cd C:\Users\romer\MatchVision-gemini
C:\Users\romer\MatchVision\venv\Scripts\python.exe -m pytest -q tests
C:\Users\romer\MatchVision\venv\Scripts\python.exe scripts\ensayo_escenas.py --listar
```
Todos los tests pasan y `--listar` funciona. **No ejecutes el script sin `--listar`** (abre la cámara).

---

## 3. Archivos permitidos
- `respuestas.md`
- `scripts/ensayo_escenas.py`

Nada más. **No toques** `README.md`, `metricas.md`, `config.yaml`, `src/`, `main.py`, `.gitignore` ni `dataset/`.

## 4. Commits (uno por tarea)
```powershell
git -C C:\Users\romer\MatchVision-gemini add respuestas.md
git -C C:\Users\romer\MatchVision-gemini commit -m "respuestas.md: modelo, dataset, etiquetado, aumento de datos, hiperparámetros y falsos positivos"
git -C C:\Users\romer\MatchVision-gemini add scripts/ensayo_escenas.py
git -C C:\Users\romer\MatchVision-gemini commit -m "Ensayo: pipeline sin duplicar, CSV para Excel y aviso de cámara desconectada"
git -C C:\Users\romer\MatchVision-gemini status --short
```

## 5. Prohibido
- ❌ Inventar métricas, porcentajes o resultados que no estén en este documento.
- ❌ Trabajar en `C:\Users\romer\MatchVision` o cambiar de rama allí.
- ❌ Modificar archivos fuera de la sección 3.
- ❌ Abrir la cámara.
- ❌ `git push`, `merge`, `rebase`, `reset`, `stash`, `clean`.
- ❌ `pip install`.

## 6. Reporte final (breve)
1. Requisito previo OK o no.
2. Tarea 1: confirmación de las 8 secciones, sin TODO y con la sección 6 en PENDIENTE; número de líneas.
3. Tarea 2: qué cambiaste en cada uno de los 5 puntos.
4. Resultado de pytest (una línea).
5. `git log --oneline -3`.
6. Dudas o datos que no encontraste.
