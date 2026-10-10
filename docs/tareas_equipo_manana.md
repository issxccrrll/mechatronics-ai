# Tareas técnicas del equipo: mañana desde las 9:00

**Objetivo de la mañana:** dibujar las cajas de las **694 fotos nuevas** que faltan en Roboflow
(Claude ya revisó las propuestas automáticas: aprobó 141 y descartó las que estaban mal). Es lo que
hace que el modelo detecte bien los objetos **en la mano**, que hoy es su punto débil. Con eso se
entrena el modelo final (v2).

**Hora límite: 11:00.** A esa hora se entrena con lo que esté revisado, aunque no esté completo.

---

## Reparto

Las fotos vienen **sin cajas**: hay que dibujarlas todas (unos 10–15 s por foto).

| Quién | Qué etiqueta | Fotos |
|---|---|---|
| **Bryan** | `destornillador` + `taza` | 154 + 69 = **223** |
| **Jack** | `botella`, y después la **prueba de instalación** | **251** |
| **Chuy** | `tijeras` + `celular` + `mixto` | 79 + 45 + 96 = **220** |

Quien termine primero ayuda a otro, empezando por **Bryan**, que tiene más fotos para dibujar.

---

## Paso 1: Preparar Roboflow (una sola persona, 5 min, la primera en llegar)

1. Entra a **app.roboflow.com** con tu cuenta.
2. **Create New Project:**
   - Project Type: **Object Detection**.
   - Name: `mechatronics-ai`.
   - Annotation group: `objetos`.
3. Crea las clases **exactamente** con estos nombres (minúsculas):
   `taza`, `botella`, `tijeras`, `destornillador`, `celular`.
4. **Settings → Members → Invite** a los demás del equipo como **Labeler** o **Admin**.

## Paso 2: Descargar tus fotos (cada uno)

1. En Google Drive, carpeta `mechatronics-ai`, descarga **`roboflow_subir.zip`** (53 MB). Isaac lo sube esta noche.
2. Descomprímelo. Adentro hay una carpeta por clase: `botella/`, `destornillador/`, `tijeras/`, etc., solo con fotos `.jpg`.

## Paso 3: Subir tus fotos a Roboflow

1. En el proyecto: **Upload Data**.
2. Arrastra **tu carpeta completa** (por ejemplo `botella/`).
3. Las fotos aparecen sin cajas; eso es lo esperado.
4. En el diálogo de subida:
   - Elige el nombre del lote (**batch name**) con tu clase, por ejemplo `botella`.
   - Si pregunta cómo repartir train/valid/test, elige **todo a Train**. Nosotros dividimos después.
5. **Save and Continue.**

## Paso 4: Dibujar las cajas foto por foto (lo más importante)

En **Annotate**, abre tu lote y revisa **cada foto**:

| Si ves… | Haz esto |
|---|---|
| Caja correcta y bien ajustada | Nada, pasa a la siguiente |
| Caja sobre algo que **no es** el objeto (un bol, una TV, un cojín, una persona) | **Bórrala** |
| Caja mal ajustada (corta el objeto o es muy grande) | **Ajústala** a los bordes del objeto |
| Objeto del reto **sin caja** (también en la mano o al fondo) | **Dibuja la caja** y elige la clase correcta |
| Hay **otro** objeto del reto en la foto (por ejemplo, una taza en tu foto de botella) | Dibújalo también, **con su clase** |
| Foto inútil (borrosa al extremo, objeto irreconocible) | **Borra la foto** |

**Reglas:**
- La caja va **ajustada al objeto completo**. Si está tapado, se marca la **parte visible**.
- **Todos** los objetos del reto que aparezcan llevan caja. Un objeto sin caja le enseña al modelo que *eso no es nada*.
- **No** se marcan personas, plumas, vasos, controles ni nada fuera de las 5 clases.
- Atajos útiles en Roboflow: `B` caja nueva, `Supr` borrar caja, flechas para cambiar de foto.

## Paso 5: Exportar (cada uno, al terminar)

1. **Generate → Create New Version.**
2. **Preprocessing:** quita todo (sobre todo **Resize** y **Auto-Orient**).
3. **Augmentation:** **ninguna**. El aumento lo hace nuestro script de entrenamiento.
4. **Create**, luego **Export Dataset** → formato **YOLOv11** → **Download zip to computer**.
5. Sube ese zip a Google Drive, en `mechatronics-ai/roboflow_export/`, con el nombre de tu clase, por ejemplo `botella.zip`.
6. Avisa en el grupo: "Terminé `botella`, X fotos".

**No cambies los nombres de los archivos.** Roboflow les agrega un sufijo (`_jpg.rf.xxxx`), pero el
inicio (`s20261009_230346_0001`) se conserva. De ahí sale la sesión de captura que usamos para dividir
train y test.

---

## Tarea extra para Jack: probar la instalación desde cero (después de botella)

La ficha resta **−10 puntos** si el código no corre siguiendo el README. Hay que probarlo en una
laptop **que no sea la de Isaac**.

1. En una carpeta nueva:
   ```
   git clone https://github.com/issxccrrll/mechatronics-ai.git
   cd mechatronics-ai
   ```
2. Sigue la sección **Instalación** del README **al pie de la letra**, sin saltarte ni agregar nada.
3. Ejecuta, en este orden:
   ```
   .\venv\Scripts\python.exe -m pytest
   .\venv\Scripts\python.exe scripts\ensayo_escenas.py --listar
   .\venv\Scripts\python.exe main.py --source 0
   ```
4. Anota **cualquier** paso que falle o que no se entienda, con captura del error, y pásalo al grupo.
   Que todo funcione también es un resultado: avísalo.

---

## Cuando llegue Isaac
- Isaac y Claude integran los zips de Roboflow, entrenan el v2 y lo comparan con el v1.
- Mientras se entrena (~1 hora): **simulacros de preguntas del pitch** al presentador.
