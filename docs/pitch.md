# Guía del pitch: Mechatronics AI

**Formato: 5 minutos de presentación + 3 minutos de preguntas.**
- Se presenta la **versión de 5 minutos** (columna "5 min"). La de 3 minutos es el plan B si los apuran.
- En **3 minutos de preguntas** caben unas 4–6. Cada respuesta debe durar **20–30 segundos**: primero la
  **frase corta** (en negritas en la sección 5) y solo si el jurado pide más, el detalle.
- Al llegar a **4:30**, quien esté hablando salta directo al cierre. **Nunca pasarse de 5:00.**

> ⚠️ Los números marcados con **[v2]** se actualizan mañana con el modelo final.

---

## 1. Estructura (una sola persona presenta)

| # | Parte | 5 min | Plan B (3 min) |
|---|---|---|---|
| 1 | Problema y solución | 0:30 | 0:20 |
| 2 | Cómo funciona (arquitectura) | 0:50 | 0:30 |
| 3 | **Demo en vivo** | **2:00** | 1:20 |
| 4 | Datos, entrenamiento y resultados | 1:10 | 0:30 |
| 5 | Cierre | 0:30 | 0:20 |

**Cómo hacerlo solo:**
- **Antes de empezar,** deja la ventana del detector **ya abierta** y los objetos **ordenados a un lado** de la mesa, en el orden de la demo. Así no pierdes tiempo buscándolos.
- **Durante la demo, habla mientras colocas cada objeto**, nunca en silencio. Coloca, espera 1–2 s a que salga en verde, y narra lo que se ve.
- **Pantalla:** si se puede, conecta la laptop a un proyector o gira la pantalla hacia el jurado.
- **El resto del equipo puede estar presente y en silencio.** Si las reglas lo permiten, un compañero puede ayudar a mover objetos, pero el guion asume que lo haces tú.
- **Lleva un cronómetro visible,** por ejemplo el celular boca arriba en la mesa. A las **4:30**, pasa al cierre.

## 2. Guion

### Parte 1: Problema y solución (0:30)
> "Construimos un detector de objetos en tiempo real que funciona **en una laptop normal, sin GPU**.
> Reconoce cinco objetos: **taza, botella, tijeras, destornillador y celular**, y todo lo demás lo
> ignora o lo marca como desconocido. Corre a más de **25 cuadros por segundo** en CPU, cuando el
> reto pide 15."

### Parte 2: Cómo funciona (0:50)
> "El sistema es un pipeline de cinco pasos:
> 1. **La cámara** se lee en un hilo aparte y siempre entregamos el cuadro más reciente. Si se desconecta, reintenta sola y lo avisa en pantalla.
> 2. **El detector** es YOLO11n, la versión más ligera de YOLO, que **entrenamos con fotos propias** a partir de un modelo preentrenado (transfer learning), y lo optimizamos con **OpenVINO** para CPU.
> 3. **Cuatro filtros contra falsos positivos.**
> 4. **La pantalla**, con cajas, clase, confianza, FPS, un contador por clase y un número de seguimiento por objeto.
> 5. **Un registro** de cada detección en CSV."

*(Versión de 5 min: explicar los 4 filtros con la sección 4 de este documento.)*

### Parte 3: Demo en vivo (2:00)
Orden fijo, ensayado:
1. **Mesa vacía:** "No hay nada y no detecta nada."
2. **Taza sola:** "Taza, con su confianza y su número de seguimiento."
3. **Destornillador:** "Este objeto no existe en los modelos comerciales; lo aprendió con nuestras fotos."
4. **Tres objetos juntos** (taza, celular, tijeras): "Varios a la vez. Mira el contador por clase abajo a la izquierda."
5. **Objeto ajeno** (pluma o vaso): "Un objeto que no es de las 5 clases: no lo marca como oficial."
6. **Objeto parcialmente tapado con la mano** (botella).
7. *(5 min)* **Mover el umbral** con el trackbar en vivo, y **desconectar la cámara** para mostrar que no truena.

> **Regla de la demo:** objetos **sobre la mesa**, quietos 1–2 segundos. El filtro de persistencia
> necesita 3 de 5 cuadros. **No agitar** los objetos en el aire.

### Parte 4: Datos, entrenamiento y resultados (1:10)
> "Tomamos **más de 3 000 fotos** con la cámara del equipo, en varias sesiones con distinta luz,
> fondo y posición, incluyendo fotos de **fondo sin objetos** para que el modelo aprenda qué no es
> un objeto. Las etiquetamos de forma **semiautomática**: un modelo propone las cajas y nosotros las
> revisamos. Entrenamos en la nube con GPU y lo ejecutamos en CPU.
> En un conjunto de prueba separado logramos un **mAP50 de 0.84 [v2]**, con botella en 0.99 y
> celular en 0.95."

*(Versión de 5 min: el aumento de datos para generalizar, la calibración de umbrales con datos y la
división train/test por sesión de captura.)*

### Parte 5: Cierre (0:30)
> "En resumen: detección en vivo en CPU a más de 25 FPS, cinco clases con un modelo entrenado por
> nosotros, cuatro técnicas contra falsos positivos, métricas reproducibles con un script, y todo
> documentado en el repositorio. Gracias."

---

## 3. Números que todos deben saber de memoria

| Dato | Valor |
|---|---|
| FPS en vivo (CPU, i7-13620H) | **17.8–29.8** (última prueba: **28.2**). El reto pide 15 |
| Modelo | **YOLO11n**, transfer learning desde COCO, exportado a **OpenVINO** y **ONNX** |
| Por qué OpenVINO | ~62 FPS de inferencia frente a ~22 con PyTorch |
| Fotos propias | **~3 400** tomadas con la webcam; test separado de train |
| mAP50 en test **[v2]** | v0: **0.84** (botella 0.99, celular 0.95, destornillador 0.85, taza 0.76, tijeras 0.66) |
| Precision y recall a conf 0.40 **[v2]** | v0: **0.87 / 0.78** |
| Entrenamiento | Google Colab, GPU Tesla T4, 80 épocas, ~1 hora. En CPU tomaría ~12 h |
| Filtros | 4: desconocido, umbral por clase, área y aspecto, persistencia 3 de 5 cuadros |
| Bonus | Contador por clase, ID de seguimiento y exportación a ONNX |

---

## 4. Los 4 filtros (para explicar en 20 segundos)

1. **Rechazo de desconocidos:** lo que no es de las 5 clases nunca recibe una clase oficial.
2. **Umbral por clase:** cada clase tiene su propio umbral, **calibrado con datos** de validación (el más bajo que conserva el F1). Así se detectan objetos en la mano sin aceptar basura.
3. **Área y relación de aspecto:** descarta cajas diminutas, gigantes o con forma imposible.
4. **Persistencia:** un objeto solo se acepta si aparece en **3 de los últimos 5 cuadros**, usando el seguimiento de ByteTrack. Elimina destellos de un solo cuadro.

---

## 5. Preguntas probables del jurado y cómo responder

**Reglas para los 3 minutos de preguntas:**
- **Responde siempre el presentador.** Por eso debe dominar las 12 frases cortas de abajo.
- Para ganar tiempo al pensar: repite la pregunta en voz alta ("¿Por qué YOLO11n? Porque…").
- **Primero la frase corta (≤ 15 segundos).** Si el jurado sigue interesado, da el detalle.
- Si no sabes algo: "No lo medimos, pero lo haríamos así: …". **Nunca inventes números.**
- Si una pregunta se puede mostrar, **muéstrala en vivo** (por ejemplo, mover el umbral o tapar un objeto).

**Versión corta de cada respuesta (memorizar):**

| Pregunta | Frase corta |
|---|---|
| ¿Por qué YOLO11n? | **El más ligero; con OpenVINO pasa de 25 FPS en CPU y el reto pide 15.** |
| ¿Preentrenado o propio? | **Ambos: transfer learning desde COCO con nuestras fotos; así aprendió el destornillador.** |
| ¿Falsos positivos? | **Cuatro filtros, umbrales calibrados con datos y fotos de fondo en el entrenamiento.** |
| ¿Tijeras de otro color? | **Entrenamos cambiando colores y sombras para que aprenda la forma, no el color.** |
| ¿Por qué tijeras/taza más bajas? | **Menos fotos y tazas sin etiquetar en escenas con varias; lo atacamos con más fotos revisadas.** |
| ¿Cómo dividieron train/test? | **Por sesión de captura, nunca al azar, para no inflar las métricas.** |
| ¿Mejor en mesa que en mano? | **Había más fotos en la mesa; calibramos umbrales y agregamos fotos en la mano.** |
| ¿Si se desconecta la cámara? | **No truena: reintenta sola con aviso en pantalla y cierra limpio con q.** |
| ¿Cómo etiquetaron? | **Semiautomático: un modelo propone cajas y nosotros las revisamos.** |
| ¿Usaron IA? | **Sí, como asistente, declarado en el README; el equipo diseñó, probó y entiende todo el código.** |
| ¿Cómo se reproducen las métricas? | **Con `scripts/eval.py`, que genera metricas.md con la matriz de confusión.** |
| ¿Qué mejorarían? | **Más fotos en la mano, revisión total de etiquetas y cuantización INT8.** |

**Detalle (solo si piden más):**

**¿Por qué YOLO11n y no un modelo más grande?**
> Porque el reto exige 15 FPS en CPU. YOLO11n es el más ligero de la familia; con OpenVINO corre a
> más de 25 FPS en el pipeline completo. Uno más grande no cumpliría la velocidad en CPU.

**¿Usaron un modelo preentrenado o entrenaron el suyo?**
> Los dos: partimos de YOLO11n preentrenado en COCO (transfer learning) y lo entrenamos con nuestras
> fotos. COCO no tiene destornillador; nuestro modelo sí lo detecta.

**¿Cómo evitan los falsos positivos?**
> Con cuatro filtros en cascada (sección 4) y con fotos de fondo en el entrenamiento, que le
> enseñan al modelo qué no es un objeto. Los umbrales están calibrados con datos, no a ojo.

**¿Qué pasa si pongo unas tijeras de otro color o una taza que no han visto?**
> Entrenamos con aumento de datos que **cambia los colores y agrega sombras y desenfoque**, para que
> el modelo aprenda la forma y no el color. Además usamos varios ejemplares de cada objeto y
> reservamos uno para la prueba (ejemplar no visto).

**¿Por qué tijeras o taza tienen menos precisión?**
> Tijeras: tuvimos menos fotos y varían mucho entre abiertas y cerradas. Taza: parte de sus "falsos
> positivos" son tazas reales que no estaban etiquetadas en fotos con varias tazas. Lo atacamos con
> más fotos revisadas a mano **[actualizar con v2]**.

**¿Cómo dividieron entrenamiento y prueba?**
> Por **sesión de captura** (o por bloques consecutivos dentro de una sesión), nunca al azar. Así
> fotos casi iguales no quedan a la vez en train y en test, lo que inflaría las métricas.

**¿Por qué detecta mejor sobre la mesa que en la mano?**
> La mayoría de las fotos de entrenamiento eran sobre la mesa; en la mano hay desenfoque por
> movimiento y oclusión. Lo mejoramos calibrando los umbrales y agregando fotos en la mano
> **[v2]**. Es nuestra principal mejora propuesta.

**¿Qué pasa si se desconecta la cámara?**
> No truena: muestra "Reintentando conexión" y se recupera solo cuando vuelve la señal. Se cierra
> limpiamente con la tecla q.

**¿Cómo etiquetaron tantas fotos?**
> De forma semiautomática: un modelo propone las cajas (YOLO para las clases que conoce y OWLv2,
> un detector por texto, para el destornillador) y nosotros revisamos y corregimos.

**¿Usaron inteligencia artificial para programar?**
> Sí, usamos asistentes de IA como apoyo, y lo declaramos en el README. El equipo diseñó la
> solución, revisó y probó el código, y puede explicar cada parte.

**¿Cómo se reproducen sus métricas?**
> Con un script: `python scripts/eval.py --model models/best.pt`, que genera `metricas.md` con
> precision, recall, mAP y la matriz de confusión del conjunto de prueba.

**¿Qué mejorarían con más tiempo?**
> Más fotos en la mano y con otros fondos, revisión completa de etiquetas, cuantización INT8 para
> ganar velocidad y un clasificador secundario para las clases que más se confunden.

---

## 6. Si algo falla en la demo

| Problema | Qué hacer |
|---|---|
| La cámara no abre | Cerrar cualquier otro programa que use la cámara (Teams, Cámara, otro `main.py`) y volver a ejecutar. Decir: "el sistema reintenta solo" |
| No detecta un objeto | Ponerlo **sobre la mesa**, quieto, a 40–80 cm, de frente. Esperar 1–2 s |
| Aparece una detección falsa | Subir un poco el trackbar de umbral. Explicar que el umbral se ajusta en vivo |
| La ventana se congela | Presionar `q` y volver a abrir: `.\venv\Scripts\python.exe main.py --source 0` |
| Va lento | Cerrar navegador y programas pesados, y **conectar el cargador** |

---

## 7. Checklist del día

- [ ] Laptop **cargada** y con cargador.
- [ ] Energía en modo **Máximo rendimiento**.
- [ ] Efectos de cámara de Windows **desactivados**.
- [ ] Repo actualizado (`git pull`) y modelo final en `models/`.
- [ ] Objetos de la demo en una bolsa: **taza de cerámica, botella de plástico, tijeras de oficina, destornillador, celular**, más una **pluma** y un **vaso** como objetos ajenos.
- [ ] Probar `main.py` **5 minutos antes** de pasar.
- [ ] El presentador ensayó el pitch completo en voz alta **al menos 3 veces** con cronómetro, con la demo real.
- [ ] El equipo le hizo **simulacros de preguntas** al azar de la sección 5.
