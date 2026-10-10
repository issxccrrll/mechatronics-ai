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
| `src/ui.py` | Cajas, etiquetas, FPS, contadores y trackbars. |
| `src/logger.py` | Escribe un CSV por sesión en `logs/`. |

## Instalación (Windows, Python 3.11)

```powershell
# Si no tienes Python 3.11: uv python install 3.11  (convive con otras versiones)
py -3.11 -m venv venv            # o: & (uv python find 3.11) -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

En Linux o macOS: `python3.11 -m venv venv && source venv/bin/activate` y los mismos `pip install`.

Prepara el modelo y expórtalo a OpenVINO, que es unas 2.5 veces más rápido en CPU:

```powershell
cd models; python -c "from ultralytics import YOLO; YOLO('yolo11n.pt')"; cd ..
python scripts/export.py --model models/yolo11n.pt --formatos openvino
```

Si la carpeta OpenVINO no existe, el detector usa automáticamente el `.pt`.

## Conexión de cámara

- **Webcam de la laptop:** `--source 0` (o 1 o 2 si hay varias).
- **Celular con IP Webcam (Android):**
  1. Abre la app y pulsa *Iniciar servidor*.
  2. Usa la URL que muestra: `--source http://IP:8080/video`.
  3. El celular y la PC deben estar en la misma red Wi-Fi.
- **Celular con DroidCam:**
  - Por URL: `--source http://IP:4747/video`.
  - O instala el cliente de Windows, que crea una webcam virtual, y usa `--source 1`.

Si la cámara no está disponible o se pierde la conexión, la ventana muestra *"Reintentando conexión…"* y reintenta sin cerrarse.

## Ejecución

```powershell
python main.py --source 0
python main.py --source http://192.168.1.50:8080/video --conf 0.5
python main.py --model models/best_openvino_model --imgsz 480
```

- Tecla **q**: salir.
- **Trackbars**: umbral global (%) y persistencia N, ajustables en vivo.
- **Colores**: verde = aceptada, rojo = rechazada (con el filtro que la rechazó), gris = desconocido.
- Toda la configuración está en `config.yaml`.

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
python scripts/autolabel.py --vista               # pre-etiqueta; vistas en dataset/revision/
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

Entrenar en CPU es inviable con miles de fotos. Se entrena en GPU con `notebooks/train_colab.ipynb`:
- El notebook clona el repo, copia `dataset.zip` desde Drive y ejecuta `scripts/train.py --device 0`.
- La **inferencia** sigue en CPU, como exige el reto.

### Etiquetado

1. **Pre-etiquetar:** `python scripts/autolabel.py --vista`.
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

Para usar el modelo propio, en `config.yaml`:
- Cambia `modelo.ruta`.
- Reemplaza `mapeo_clases` por el mapeo identidad (`taza: taza`, …).

## Evaluación

```powershell
pytest                                             # pruebas de filtros
python scripts/eval.py --model models/best.pt      # escribe metricas.md y metricas.csv
python scripts/benchmark_fps.py --model models/yolo11n_openvino_model
```

## Decisiones técnicas

- **YOLO11n:** el modelo más ligero de la familia, suficiente para cumplir 15 FPS o más en CPU.
- **OpenVINO por defecto:** unos 62 FPS de inferencia frente a unos 22 FPS con PyTorch en esta laptop.
- **Cámara en un hilo:** siempre se procesa el frame más reciente, sin acumular retraso.
- **Umbral bajo en el modelo (0.25) y umbral real en los filtros:** así el trackbar puede bajar o subir el umbral en vivo.
- **Persistencia N de M con ByteTrack:** elimina falsos positivos de un solo frame. Si no hay `track_id`, empareja por IoU.
- **Split por sesión de captura:** evita fuga de datos entre train y test.

## Fuentes

- [Ultralytics YOLO11](https://docs.ultralytics.com/models/yolo11/)
- [Exportación a OpenVINO](https://docs.ultralytics.com/integrations/openvino/)
- [Tracking (ByteTrack)](https://docs.ultralytics.com/modes/track/)
- Dataset COCO (pesos preentrenados).
- [OWLv2](https://huggingface.co/google/owlv2-base-patch16-ensemble) (Minderer et al., 2023): pre-etiquetado zero-shot.
- [Albumentations](https://albumentations.ai/): aumento de datos.

## Limitaciones conocidas

- Con los pesos COCO **no existe la clase destornillador**: hace falta el modelo propio (`best.pt`).
- COCO no incluye "tijeras de mecánico" ni objetos de taller específicos. Es posible confundir celular con control remoto, o taza con vaso.
- La primera inferencia con OpenVINO tarda unos 10 s (compilación). Se hace al iniciar, antes de abrir la ventana.
- Los FPS dependen de la CPU y de la resolución de la cámara.

## Mejoras propuestas

- Entrenar `best.pt` con un dataset propio que incluya fondos negativos (clase vacía).
- Activar el verificador CNN (bonus) para las clases con más confusión.
- Cuantización INT8 con OpenVINO (NNCF) para ganar FPS.
- Calibrar los umbrales por clase con la curva PR del split val.
