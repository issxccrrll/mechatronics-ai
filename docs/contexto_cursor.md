# Contexto para Cursor: guion del video y revisión de la documentación

Vas a hacer **dos tareas que solo crean archivos nuevos**. **No modifiques ningún archivo
existente**: otra persona está editando el README y las métricas, y cualquier cambio tuyo chocaría
con el suyo.

## Proyecto (resumen)
- Detector de objetos en vivo para el reto "Mechatronics AI" (Hack Knights IA Challenge 2026).
- 5 clases: taza, botella, tijeras, destornillador, celular. Lo demás se ignora o se marca como desconocido.
- YOLO11n entrenado con fotos propias (transfer learning desde COCO), OpenVINO en CPU, 17.8–29.8 FPS en vivo.
- 4 filtros contra falsos positivos. En pantalla: cajas, clase, confianza, FPS, contador por clase e ID de seguimiento.
- Carpeta del proyecto: `C:\Users\romer\MatchVision`. Léelo para entender: `README.md`, `respuestas.md`,
  `metricas.md`, `main.py`, `config.yaml`, `src/`, `scripts/`.

## Qué pide la ficha del reto
- **Video de demostración:** máximo **2 minutos**, mostrando el sistema funcionando con la cámara del
  equipo y ejemplos de detección en vivo.
- **README:** instalación, configuración de la cámara, ejecución y evaluación; decisiones técnicas,
  fuentes, limitaciones y mejoras. **−10 puntos si el código no corre siguiendo el README.**
- **El jurado probará:** posiciones, distancias y orientaciones distintas; varios objetos a la vez;
  objetos parcialmente ocultos; cambios de luz; objetos ajenos; fondo vacío.

---

## TAREA 1: crear `docs/guion_video.md`

Un guion **toma por toma** para grabar el video de **1:45 a 2:00 minutos** con la pantalla de la
laptop: ventana del detector y voz en off, o celular grabando la laptop. Incluye:
1. **Tabla de tomas:** tiempo de inicio y fin, qué se ve en pantalla, qué objetos se colocan y el texto de la voz en off.
2. **Las tomas deben cubrir, en este orden:**
   - Arranque del programa (se ve el comando y la ventana).
   - Mesa vacía (no detecta nada).
   - Cada una de las 5 clases.
   - Varios objetos juntos (se ve el contador por clase).
   - Objeto ajeno (pluma o vaso, sin detección oficial).
   - Objeto parcialmente tapado.
   - Cambio de luz.
   - Mover el trackbar de umbral.
   - Desconectar la cámara (mensaje "Reintentando conexión") y cerrar con `q`.
3. **Consejos de grabación:** objetos sobre la mesa y quietos 1–2 s (el filtro de persistencia necesita 3 de 5 cuadros), buena luz, laptop conectada al cargador, cerrar otros programas, y cómo grabar la pantalla en Windows (Xbox Game Bar: Win+Alt+R, o la Herramienta Recortes en modo video).
4. **Checklist** antes de grabar.

**No inventes números.** Si mencionas métricas, usa solo las de `metricas.md` y aclara que son del modelo v0.

## TAREA 2: crear `docs/revision_cursor.md`

**Revisa sin editar** `README.md` y `respuestas.md`. En el archivo nuevo, lista:
1. **Errores de ortografía o redacción**, con el formato: archivo, texto actual → texto sugerido.
2. **Comandos del README que no coincidan con el código:** comprueba que cada script, archivo y opción mencionados existan (`scripts/*.py`, opciones de `argparse`, rutas en `models/` y `config.yaml`). Indica la línea y qué está mal.
3. **Contradicciones** entre README, `respuestas.md`, `metricas.md` y `config.yaml` (por ejemplo, umbrales o números distintos).
4. **Lo que falte según la ficha** (lista de arriba).

Ordena los hallazgos por importancia: primero lo que podría hacer que **el código no corra
siguiendo el README**.

---

## Reglas
- ✅ Solo puedes **crear**: `docs/guion_video.md` y `docs/revision_cursor.md`.
- ❌ No modifiques, borres ni renombres ningún archivo existente.
- ❌ No ejecutes `main.py`, `capture.py` ni nada que abra la cámara.
- ❌ No hagas commits ni push; no uses git, salvo para leer (`git log`, `git diff`).
- ❌ No instales paquetes.
- Escribe en español, claro y breve.
