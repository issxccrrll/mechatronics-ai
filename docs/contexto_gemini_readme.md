# Contexto para Gemini: actualizar README.md y completar los resultados de respuestas.md

Tarea de **documentación** con datos reales del modelo ya entrenado. Sigue las instrucciones al pie
de la letra. **No inventes números:** usa solo los de este documento y los de `metricas.md`. Si algo
no coincide, **DETENTE y repórtalo**.

---

## 0. Preparar tu carpeta (obligatorio)

Tu rama `gemini` quedó atrás: Claude ya unió tu trabajo y agregó el modelo entrenado. Ponla al día:
```powershell
cd C:\Users\romer\MatchVision-gemini
git status --short
git merge --ff-only camara-mejoras
git log --oneline -1
```
- `status` debe estar vacío **antes** del merge. Si no, DETENTE.
- El `merge --ff-only` debe terminar sin conflictos y el último commit debe ser `fd0bf9d` ("Modelo propio v0: ..."). Si falla, **DETENTE y reporta el error**. No uses otro tipo de merge.
- **Trabaja SOLO en `C:\Users\romer\MatchVision-gemini`.**

Después de esto, en tu carpeta tendrás `metricas.md` (con las métricas reales), `docs/confusion_matrix.png`, `models/best.pt`, etc. **Léelos.**

---

## 1. Qué pide la rúbrica de estos documentos

- **README.md (6 pts):** instalación, configuración de la cámara, ejecución y evaluación; decisiones técnicas, fuentes externas, limitaciones conocidas y mejoras propuestas.
- **Justificar la técnica contra falsos positivos en el README (4 pts).**
- **Penalización de −10** si el código no se ejecuta siguiendo el README: los comandos deben ser **exactos**.
- **Penalización de −10** por código copiado sin citar: las fuentes deben estar completas.
- **respuestas.md:** la sección "6. Resultados" hoy dice PENDIENTE y hay que completarla.

---

## 2. Hechos nuevos (fuente de verdad)

**Modelo propio v0**
- Archivo `models/best.pt`, entrenado en Google Colab (Tesla T4), 80 épocas, desde `yolo11n.pt`.
- Exportado a `models/best_openvino_model/`, que es el **modelo por defecto** en `config.yaml > modelo.ruta`, y a `models/best.onnx` (bonus de la rúbrica: exportación a ONNX).
- **Los tres archivos están incluidos en el repo.** No hay que descargar ni entrenar nada para ejecutar el detector.
- Si la carpeta OpenVINO no existiera, el detector usa `models/best.pt` automáticamente (más lento).
- Para volver al modelo COCO de respaldo: `python main.py --model models/yolo11n_openvino_model` (tiene `destornillador` como desconocido).
- El modelo propio devuelve directamente los nombres oficiales (taza, botella, tijeras, destornillador, celular); `mapeo_clases` solo aplica a modelos COCO.

**Rendimiento en CPU (laptop del equipo, Intel Core i7-13620H)**
- Inferencia OpenVINO del modelo propio: **~32–35 FPS** medidos con otros programas abiertos; en condiciones limpias el modelo base daba ~62 FPS.
- Pipeline completo en vivo (cámara + detector + tracking + filtros + dibujo): **17.8–29.8 FPS** medidos. Supera el mínimo de 15.
- Comando de medición: `python scripts/benchmark_fps.py --model models/best_openvino_model`.

**Métricas en test** (287 imágenes separadas de train/val). Están en `metricas.md` y `metricas.csv`; **cópialas de ahí, no las redondees distinto**. Resumen:
- Protocolo estándar (conf 0.001): **mAP50 = 0.8417**, mAP50-95 = 0.6619, precision 0.8562, recall 0.8064.
- Por clase (mAP50): botella 0.9907, celular 0.9493, destornillador 0.8487, taza 0.7588, tijeras 0.6608.
- Al umbral de operación (conf 0.40): precision **0.8743**, recall **0.7807**.
- **Matriz de confusión a conf 0.40** (está en `metricas.md`):
  - Casi no hay confusiones **entre clases oficiales**: 1 botella predicha como taza y 1 tijeras predicha como celular.
  - Errores principales:
    - Predicciones sobre fondo: taza 38, destornillador 5, celular 4, tijeras 2, botella 1.
    - No detectados: taza 19, tijeras 13, destornillador 19, botella 3, celular 3.
  - **Interpretación:** buena parte de las 38 "tazas sobre fondo" corresponden a tazas reales **sin etiquetar** en escenas con varias tazas (el pre-etiquetado automático no marcó todas). Es una limitación de las etiquetas, que se corregirá con revisión manual. Redáctalo como **"probablemente"**, no como hecho comprobado.
- **El test del v0 se dividió por bloques dentro de cada sesión.** Las métricas pueden ser algo optimistas; el modelo final se evaluará con división por sesión completa.
- Reproducir: `python scripts/eval.py --model models/best.pt`.

**Funciones extra (bonus) ya implementadas**
- Contador de objetos por clase en pantalla (panel inferior izquierdo) y ID de seguimiento (`taza #3 0.87`) con ByteTrack. Se activan o desactivan en `config.yaml > ui` (`mostrar_contador_clases`, `mostrar_id`).
- Exportación a ONNX: `models/best.onnx`.

**Ensayo de las 20 escenas**
- `python scripts/ensayo_escenas.py --listar` muestra las escenas.
- `python scripts/ensayo_escenas.py` abre la cámara: ESPACIO evalúa, s salta, q termina. Guarda `logs/ensayo_*.csv` e imprime el % de aciertos (meta 85%).
- **Todavía no se ha ejecutado un ensayo completo con el modelo v0.** No inventes un resultado.

**Pendiente (para Limitaciones y Mejoras)**
- Fotos de `fondo` (objetos ajenos y mesa vacía), botellas propias y destornilladores sobre la mesa: en captura **ahora mismo**. Se usarán para el modelo v1.
- Riesgo: un modelo de 5 clases puede asignar la clase más parecida a objetos ajenos (por ejemplo, una pluma como destornillador). Mitigación: fotos de fondo, umbrales por clase, persistencia y calibración de umbrales con la curva precision-recall.
- Revisión manual de etiquetas en Roboflow.

---

## 3. TAREA A: actualizar `README.md`

**Edita, no reescribas desde cero.** Conserva la estructura y todo lo que sigue siendo correcto. Cambios:
1. **Instalación:** aclara que los pesos del modelo propio ya vienen en `models/`. El paso de descargar `yolo11n.pt` y exportarlo solo hace falta para el pre-etiquetado (`autolabel.py`) o para usar el modelo COCO; muévelo a una nota.
2. **Ejecución:** menciona el contador por clase y el ID en pantalla, y agrega una subsección **"Ensayo de las 20 escenas"** con los comandos.
3. **Evaluación:** pon la tabla de resultados del test (cópiala de `metricas.md`), el comando para reproducirla y un enlace a `metricas.md`.
4. **Nueva sección `## Justificación de la técnica contra falsos positivos`** (vale 4 pts). Incluye:
   - Qué hace cada uno de los 4 filtros y por qué (toma la justificación de `respuestas.md` sección 5, resumida).
   - **Evidencia con datos:** la matriz de confusión a conf 0.40 (casi sin confusiones entre clases) y la diferencia entre precision a conf 0.001 (0.856) y a conf 0.40 (0.874).
   - Qué falta medir: el efecto individual de cada filtro se medirá con el ensayo de escenas. **No inventes ese número.**
5. **Decisiones técnicas:** agrega la del modelo propio con transfer learning y la del entrenamiento en la nube con inferencia en CPU.
6. **Limitaciones conocidas:** actualízalas con los datos reales: tijeras con recall bajo (mAP50 0.66), tazas sobre fondo, test por bloques, pendiente de fondo y botella, y el riesgo con objetos ajenos. Quita las que ya no aplican (por ejemplo, "COCO no tiene destornillador" ya está resuelto con el modelo propio).
7. **Mejoras propuestas:** modelo v1 con fondo, botellas y destornilladores sobre la mesa; revisión en Roboflow; división por sesión; calibración de umbrales con la curva PR; cuantización INT8 con OpenVINO.
8. **Fuentes:** que estén Ultralytics YOLO11, ByteTrack, OpenVINO, ONNX, COCO, OWLv2, Albumentations, Roboflow y Google Colab, más la frase: "Se usaron asistentes de IA (Claude y Gemini) como apoyo en el desarrollo y la documentación; el código fue revisado y probado por el equipo."

**Revisa que cada comando del README exista y sea correcto.** Compruébalo con:
```powershell
C:\Users\romer\MatchVision\venv\Scripts\python.exe scripts\ensayo_escenas.py --listar
C:\Users\romer\MatchVision\venv\Scripts\python.exe scripts\eval.py --help
C:\Users\romer\MatchVision\venv\Scripts\python.exe scripts\benchmark_fps.py --help
C:\Users\romer\MatchVision\venv\Scripts\python.exe main.py --help
```
(`main.py --help` solo imprime la ayuda; no abre la cámara.)

## 4. TAREA B: completar "6. Resultados" en `respuestas.md`

Reemplaza el texto PENDIENTE de la sección 6 por:
- La tabla por clase del protocolo estándar y la tabla de precision y recall a conf 0.40 (cópialas de `metricas.md`).
- La matriz de confusión a conf 0.40 y su interpretación (sección 2 de este documento).
- Un párrafo de lectura de resultados: qué clases van bien (botella, celular), cuáles hay que mejorar (tijeras, taza) y por qué; que el test por bloques puede ser optimista; y qué se hará en el v1.
- El rendimiento en CPU con el modelo propio.

**No cambies las demás secciones**, salvo para corregir algo que contradiga estos hechos (por ejemplo, si dicen que el modelo propio está "pendiente").

---

## 5. Archivos permitidos
- `README.md`
- `respuestas.md`

**No toques** código, `config.yaml`, `metricas.md`, `.gitignore`, `models/` ni `dataset/`.

## 6. Commit
```powershell
git -C C:\Users\romer\MatchVision-gemini add README.md respuestas.md
git -C C:\Users\romer\MatchVision-gemini status --short
git -C C:\Users\romer\MatchVision-gemini commit -m "README y respuestas.md con resultados reales del modelo v0 y justificación de falsos positivos"
```

## 7. Prohibido
- ❌ Inventar métricas, porcentajes de ensayo o resultados.
- ❌ Trabajar en `C:\Users\romer\MatchVision` o cambiar de rama allí.
- ❌ Modificar archivos fuera de la sección 5.
- ❌ Abrir la cámara (no ejecutes `main.py` sin `--help` ni el ensayo sin `--listar`).
- ❌ `git push`, `rebase`, `reset`, `stash` o un merge que no sea el `--ff-only` del paso 0.
- ❌ `pip install`.

## 8. Reporte final (breve)
1. Resultado del paso 0 (último commit).
2. Lista de cambios en el README, sección por sección.
3. Confirmación de que la sección 6 de `respuestas.md` ya no dice PENDIENTE.
4. Salida de los 4 comandos de verificación (solo la primera línea de cada uno).
5. `git log --oneline -2`.
6. Dudas o datos que no encontraste.
