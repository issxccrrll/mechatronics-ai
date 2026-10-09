# Respuestas del reto

## 1. Modelo elegido

- **Estado actual:** YOLO11n preentrenado en COCO, exportado a OpenVINO. Se mapean las clases COCO `cup`, `bottle`, `scissors` y `cell phone` a taza, botella, tijeras y celular. Toda otra clase se reporta como `desconocido`.
- **Modelo final:** TODO. Indicar si es `best.pt` propio (fine-tuning de YOLO11n) o el preentrenado.
  - Tamaño del dataset: TODO (imágenes por clase, número de sesiones).
  - Hiperparámetros: TODO (epochs, imgsz, batch, lr0).
  - Justificación: TODO.

## 2. Técnica contra falsos positivos y justificación

Pipeline de filtros en cascada (patrón Strategy, cada uno activable en `config.yaml`):

1. **UnknownRejectFilter:** ningún objeto fuera de las 5 clases recibe una clase oficial.
2. **ClassThresholdFilter:** umbral por clase. Las clases que más se confunden llevan un umbral más alto.
3. **AreaAspectFilter:** descarta cajas diminutas, gigantes o con proporciones imposibles.
4. **PersistenceFilter:** un objeto solo se acepta si aparece en N de los últimos M frames, usando ByteTrack. Elimina los destellos de un solo frame.
5. *(Bonus)* **Verificador CNN:** segunda opinión sobre el recorte de cada detección.

Justificación: TODO. Medir FP/minuto con y sin filtros usando el CSV de `logs/`.

## 3. Decisiones

- OpenVINO en lugar de PyTorch para la inferencia en CPU: TODO, poner los FPS medidos.
- Split por sesión de captura, no al azar.
- Umbral ajustable en vivo con trackbar.
- TODO: otras decisiones del equipo.
