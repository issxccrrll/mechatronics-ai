# Contexto para Cursor: rediseño visual de la ventana del detector (UI/UX)

Actúa como **diseñador UI/UX con criterio de ingeniería**. Hay que mejorar el aspecto de la ventana
del detector en vivo **sin afectar su funcionamiento ni su velocidad**. El jurado la verá durante la
demo y en el video. **Hora límite: 12:45.** Si no está terminado y verificado para entonces, se descarta.

---

## 0. Dónde trabajar (obligatorio)

Trabaja **solo** en la carpeta `C:\Users\romer\MatchVision-gemini` (rama `gemini`). Primero ponla al día:
```powershell
cd C:\Users\romer\MatchVision-gemini
git status --short
git merge --ff-only camara-mejoras
git log --oneline -1
```
- `status` debe salir vacío antes del merge, y el merge debe ser fast-forward sin conflictos. Si algo falla, **DETENTE y repórtalo**.
- **Nunca** trabajes en `C:\Users\romer\MatchVision`: ahí está trabajando otra persona.
- Python: `C:\Users\romer\MatchVision\venv\Scripts\python.exe`, con ruta completa.

## 1. Qué hay hoy (léelo antes de diseñar)

- `src/ui.py`, clase `UI`:
  - `dibujar(frame, detecciones, fps)` dibuja una caja por detección. El color va según el **estado**: verde aceptada, rojo rechazada, gris desconocido.
  - Etiqueta `clase #id 0.87`. Las rechazadas llevan además `[Filtro]`.
  - Barra negra superior (24 px) con FPS y conteos por estado.
  - Funciones puras: `contar_por_clase`, `texto_etiqueta` y `dibujar_contador` (panel inferior izquierdo con las 5 clases).
  - `pantalla_estado(mensaje, ancho, alto)`: frame negro con un mensaje (cámara desconectada).
  - Trackbars `Umbral %` y `Persistencia N` (los crea OpenCV en la misma ventana).
- `main.py` llama: `ui.dibujar(frame, detecciones, fps.tick())`, `ui.mostrar(frame)`, `ui.procesar_eventos()`, `ui.pantalla_estado(...)`, `ui.leer_trackbars()`, `ui.cerrar()`. `scripts/ensayo_escenas.py` también usa `ui.dibujar`, `ui.nombre` y `ui.pantalla_estado`.
- Tipos: `src/types.py` (`Detection`, `Estado`). Clases oficiales en `config.yaml > clases`.
- Tests actuales: `tests/test_ui.py`.

## 2. Requisitos que NO se pueden romper (rúbrica del reto)

Siempre visibles en pantalla:
1. **Caja delimitadora** de cada detección.
2. **Clase y confianza** de cada detección (ID si existe).
3. **FPS**, con el valor numérico legible.
4. **Contador de objetos por clase** (bonus de la rúbrica).
5. Diferencia clara entre **aceptada**, **rechazada** y **desconocido**.

Además:
- **No cambies firmas ni nombres públicos** de `UI` ni de las funciones existentes. `main.py` y `ensayo_escenas.py` no se tocan.
- `cv2.FONT_HERSHEY_*` **no dibuja acentos ni ñ**. Todo texto en pantalla va **sin acentos** ("Senal", "conexion").
- **Prohibido** usar PIL, fuentes externas o cualquier dependencia nueva: solo `cv2` y `numpy`.
- **Rendimiento:** dibujar un frame de 640×480 con 8 detecciones debe tardar **≤ 3 ms de media** (ver la verificación). El detector corre a ~25–30 FPS y no podemos perder margen.

## 3. Propuesta de diseño (puedes mejorarla, sin salirte de las reglas)

1. **Paleta por clase** (BGR, consistente en caja, etiqueta y contador), con buen contraste sobre fondos claros y oscuros:
   - taza: ámbar.
   - botella: azul.
   - tijeras: magenta.
   - destornillador: verde.
   - celular: cian o rojo coral.
   - **Aceptada:** color de su clase, línea de 2 px y **esquinas reforzadas** (marcas en L en las 4 esquinas).
   - **Rechazada:** rojo, línea fina de 1 px, sin esquinas. Etiqueta pequeña con el filtro.
   - **Desconocido:** gris, línea fina, etiqueta "desconocido".
2. **Etiqueta tipo "píldora":** rectángulo relleno del color de la clase, texto blanco o negro según el contraste, y una **mini barra de confianza** (proporcional a la confianza) debajo del texto o dentro de la etiqueta. Si la etiqueta se sale por arriba del frame, va **dentro** de la caja.
3. **Barra superior semitransparente** (unos 30 px):
   - A la izquierda: `MECHATRONICS AI`.
   - Al centro: conteos por estado.
   - A la derecha: **insignia de FPS** coloreada (verde ≥ 20, ámbar 15–20, rojo < 15).
4. **Panel contador** con fondo semitransparente redondeado (o con esquinas), un **punto de color** por clase, el nombre y la cantidad. Resaltar las clases con cantidad mayor que 0.
5. **Pantalla de estado** (sin cámara): centrada, un ícono simple dibujado con primitivas (por ejemplo, un círculo con barra), el mensaje y "Presiona q para salir".
6. Mantener la legibilidad a 640×480 y al **maximizar** la ventana.

Toda la transparencia se hace **solo sobre la región** del panel o la barra (como hace hoy `dibujar_contador`), **nunca** con `addWeighted` sobre el frame completo.

## 4. Pruebas y verificación (obligatorio antes de commitear)

1. **Actualiza o agrega pruebas** en `tests/test_ui.py` para las funciones nuevas (por ejemplo, color por clase, posición de la etiqueta cuando la caja toca el borde superior, color de la insignia de FPS). Las pruebas existentes deben seguir pasando; si alguna dependía de un detalle visual que cambiaste a propósito, ajústala y explícalo en el reporte.
2. **Tests completos:**
   ```powershell
   C:\Users\romer\MatchVision\venv\Scripts\python.exe -m pytest -q tests
   ```
3. **Render de prueba y tiempo de dibujo** (sin cámara). Guarda las imágenes en `%TEMP%`:
   ```powershell
   @'
   import sys, os, time, numpy as np, cv2
   sys.path.insert(0, ".")
   from src.types import Detection, Estado
   from src.ui import UI
   cfg = {"ui": {"ventana": "prueba", "trackbars": False, "mostrar_rechazadas": True,
                 "mostrar_contador_clases": True, "mostrar_id": True},
          "clases": ["taza", "botella", "tijeras", "destornillador", "celular"]}
   ui = UI(cfg)
   dets = [Detection("taza", 0.91, (40, 10, 180, 170), track_id=3),
           Detection("botella", 0.77, (220, 90, 300, 330), track_id=7),
           Detection("tijeras", 0.62, (330, 260, 470, 340), track_id=9),
           Detection("destornillador", 0.55, (480, 120, 630, 170), track_id=11),
           Detection("celular", 0.88, (360, 60, 450, 220), track_id=12),
           Detection("tijeras", 0.31, (60, 330, 200, 420), track_id=5, estado=Estado.RECHAZADA, rechazada_por="ClassThresholdFilter"),
           Detection("desconocido", 0.80, (230, 350, 330, 470), estado=Estado.DESCONOCIDO),
           Detection("celular", 0.40, (560, 300, 638, 470), track_id=21)]
   for nombre, fondo in (("claro", 200), ("oscuro", 30)):
       f = np.full((480, 640, 3), fondo, np.uint8)
       cv2.imwrite(os.path.join(os.environ["TEMP"], f"ui_{nombre}.png"), ui.dibujar(f, dets, 27.4))
   cv2.imwrite(os.path.join(os.environ["TEMP"], "ui_estado.png"), ui.pantalla_estado("Sin senal. Reintentando conexion (3)..."))
   base = np.full((480, 640, 3), 120, np.uint8)
   t0 = time.perf_counter()
   for _ in range(300): ui.dibujar(base.copy(), dets, 27.4)
   print(f"dibujo medio: {(time.perf_counter()-t0)/300*1000:.2f} ms (limite 3 ms)")
   ui.cerrar()
   '@ | C:\Users\romer\MatchVision\venv\Scripts\python.exe -
   ```
   - **El tiempo medio debe ser ≤ 3 ms.**
   - Revisa las 3 imágenes: todo legible, nada fuera del frame, la etiqueta de la caja que toca el borde superior (la taza) dentro del frame, y la barra superior sin tapar los trackbars.
4. **No ejecutes `main.py`** (la cámara está en uso). Isaac lo probará en vivo.

## 5. Archivos permitidos
- `src/ui.py`
- `tests/test_ui.py`

Nada más.

## 6. Commit
```powershell
git -C C:\Users\romer\MatchVision-gemini add src/ui.py tests/test_ui.py
git -C C:\Users\romer\MatchVision-gemini commit -m "UI: paleta por clase, etiquetas con barra de confianza, barra superior e insignia de FPS"
```

## 7. Prohibido
- ❌ Trabajar en `C:\Users\romer\MatchVision`.
- ❌ Tocar archivos fuera de la sección 5.
- ❌ Dependencias nuevas, PIL o fuentes externas.
- ❌ Cambiar firmas públicas de `UI` o de las funciones existentes.
- ❌ `git push`, `rebase`, `reset`, `stash` o un merge que no sea el `--ff-only` del paso 0.
- ❌ Abrir la cámara.

## 8. Reporte final (breve)
1. Resultado del paso 0.
2. Qué cambiaste en el diseño (lista corta).
3. Salida de pytest (una línea).
4. Tiempo medio de dibujo y rutas de las 3 imágenes.
5. `git log --oneline -2`.
