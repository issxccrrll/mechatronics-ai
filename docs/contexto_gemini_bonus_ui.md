# Contexto para Gemini: bonus de pantalla (contador por clase + ID de seguimiento)

Vas a implementar **una mejora acotada** en un solo módulo. Sigue las instrucciones al pie de la
letra. Si algo no coincide con lo descrito (un archivo no existe, un test falla antes de que
cambies nada, la carpeta no está), **DETENTE y repórtalo**. No modifiques archivos fuera de la
lista permitida.

---

## 1. Requisito previo (obligatorio)

Esta tarea va **después** de `docs/contexto_gemini_separar_ramas.md`. Antes de empezar comprueba:
```powershell
git -C C:\Users\romer\MatchVision-gemini branch --show-current
```
- **Si responde `gemini`:** continúa.
- **Si la carpeta no existe o la rama es otra:** DETENTE y responde "Primero hay que completar la separación de ramas".

**Trabaja SOLO en `C:\Users\romer\MatchVision-gemini`.** Nunca en `C:\Users\romer\MatchVision`, que es de Claude.
Python: `C:\Users\romer\MatchVision\venv\Scripts\python.exe`, con ruta completa. No crees otro venv.

---

## 2. Proyecto (lo que necesitas saber)

- Detector de objetos en vivo (YOLO11n + OpenCV) para el reto "Mechatronics AI" (Hack Knights IA Challenge 2026).
- **Clases oficiales** (en `config.yaml > clases`): `taza, botella, tijeras, destornillador, celular`.
  Cualquier otro objeto se marca como `desconocido`.
- **Pipeline:** `main.py` orquesta Cámara → Detector → Verificador → Filtros → **UI** + Logger.
- **`src/types.py`** define:
  ```python
  class Estado(str, Enum): ACEPTADA, RECHAZADA, DESCONOCIDO
  @dataclass
  class Detection:
      clase: str; confianza: float; bbox: tuple[int,int,int,int]  # x1,y1,x2,y2
      track_id: int | None = None; estado: Estado = Estado.ACEPTADA
      rechazada_por: str | None = None; clase_original: str = ""
  ```
- **`src/ui.py`**, clase `UI`:
  - `UI(cfg)` crea la ventana y los trackbars.
  - `dibujar(frame, detecciones, fps)` dibuja cajas, etiquetas y una barra superior negra (24 px) con FPS y conteos por **estado**.
  - `main.py` llama `ui.dibujar(frame, detecciones, fps.tick())`. **Esa firma no debe cambiar.**
- **`track_id`** ya viene calculado por ByteTrack desde el detector, pero **hoy no se muestra**.
- **Convenciones (ver `CLAUDE.md`):** comentarios y textos en español, módulos cortos y legibles, mismo estilo que el código existente.

---

## 3. La tarea: bonus de la rúbrica (+7 puntos)

La rúbrica del reto da puntos extra por:
- **Contador de objetos por clase en pantalla (+4).**
- **Seguimiento con ID por objeto entre frames (+3).**

### 3.1 Contador por clase
- Un panel semitransparente en la **esquina inferior izquierda** del frame que liste **las 5 clases
  oficiales siempre**, en el orden de `config.yaml > clases`, con el número de detecciones **ACEPTADAS** de cada una en el frame actual. Ejemplo:
  ```
  taza           2
  botella        0
  tijeras        1
  destornillador 0
  celular        1
  ```
- **Solo cuenta detecciones con `estado == Estado.ACEPTADA`.** Las rechazadas y las desconocidas no suman.
- Las clases en 0 se muestran en un tono más tenue (gris) y las mayores que 0 en blanco, para que se lea de un vistazo.
- **No tapes la barra superior** (los primeros 24 px).
- Debe verse bien en frames de 640×480.

### 3.2 ID de seguimiento
- Si `d.track_id` no es `None`, la etiqueta pasa de `taza 0.87` a **`taza #3 0.87`**.
- Si `track_id` es `None`, la etiqueta queda igual que hoy.
- En las rechazadas se conserva el sufijo actual ` [NombreDelFiltro]`.

### 3.3 Configuración
Agrega en `config.yaml`, **dentro del bloque `ui:` existente**, solo estas dos claves:
```yaml
  mostrar_contador_clases: true
  mostrar_id: true
```
En `UI.__init__` léelas con `cfg["ui"].get("...", True)`, para que configs viejas sin esas claves sigan funcionando.

### 3.4 Diseño para poder probarlo sin ventana
`UI.__init__` abre una ventana de OpenCV, así que **los tests no pueden instanciar `UI`**. Por eso:
- Crea en `src/ui.py` **funciones de módulo puras** (sin ventana), por ejemplo:
  - `contar_por_clase(detecciones, clases) -> dict[str, int]`
  - `texto_etiqueta(d: Detection, mostrar_id: bool) -> str`
  - `dibujar_contador(frame, conteo: dict[str, int]) -> None`
- `UI.dibujar` debe **usar esas funciones**. No dupliques lógica.

### 3.5 Pruebas
Crea `tests/test_ui.py`, con el mismo estilo que `tests/test_filters.py` (usa `sys.path.insert` como ese archivo). Cubre al menos:
- `contar_por_clase` incluye las 5 clases aunque estén en 0.
- `contar_por_clase` ignora RECHAZADA y DESCONOCIDO.
- `contar_por_clase` cuenta varias detecciones de la misma clase.
- `texto_etiqueta` con `track_id` → contiene `#<id>`.
- `texto_etiqueta` sin `track_id` → no contiene `#`.
- `texto_etiqueta` con `mostrar_id=False` → no contiene `#` aunque haya `track_id`.
- `texto_etiqueta` de una rechazada → termina con `[<rechazada_por>]`.
- `dibujar_contador` sobre un `np.zeros((480, 640, 3), np.uint8)` no lanza error y **modifica** píxeles en la esquina inferior izquierda, pero **no** en la barra superior (filas 0–23).

---

## 4. Archivos permitidos

Solo puedes modificar o crear:
- `src/ui.py`
- `config.yaml` (únicamente las 2 claves de la sección 3.3)
- `tests/test_ui.py` (nuevo)

**Todo lo demás es solo lectura**, en especial `main.py`, `src/camera.py`, `src/filters.py`,
`src/detector.py`, `README.md`, `scripts/` y `dataset/`.

---

## 5. Verificación (obligatoria antes de commitear)

Desde `C:\Users\romer\MatchVision-gemini`:

**a) Todos los tests:**
```powershell
C:\Users\romer\MatchVision\venv\Scripts\python.exe -m pytest -q tests
```
Deben pasar **todos**, los existentes y los nuevos.

**b) Render de prueba sin cámara.** Genera una imagen con detecciones simuladas y revísala:
```powershell
@'
import sys, os, numpy as np, cv2
sys.path.insert(0, ".")
from src.types import Detection, Estado
from src.ui import contar_por_clase, texto_etiqueta, dibujar_contador
clases = ["taza", "botella", "tijeras", "destornillador", "celular"]
dets = [Detection("taza", 0.91, (60, 80, 200, 240), track_id=3),
        Detection("taza", 0.77, (240, 90, 360, 250), track_id=7),
        Detection("celular", 0.66, (400, 120, 520, 300), track_id=12),
        Detection("tijeras", 0.30, (420, 320, 600, 420), track_id=5, estado=Estado.RECHAZADA, rechazada_por="ClassThresholdFilter"),
        Detection("desconocido", 0.80, (20, 300, 120, 460), estado=Estado.DESCONOCIDO)]
frame = np.full((480, 640, 3), 90, np.uint8)
conteo = contar_por_clase(dets, clases)
print("conteo:", conteo)
print("etiquetas:", [texto_etiqueta(d, True) for d in dets])
dibujar_contador(frame, conteo)
ruta = os.path.join(os.environ["TEMP"], "render_bonus_ui.png")
cv2.imwrite(ruta, frame); print("imagen:", ruta)
'@ | C:\Users\romer\MatchVision\venv\Scripts\python.exe -
```
Esperado:
- `conteo: {'taza': 2, 'botella': 0, 'tijeras': 0, 'destornillador': 0, 'celular': 1}`. Las tijeras están rechazadas, así que no cuentan.
- Las etiquetas con `#3`, `#7` y `#12`.
- La imagen con el panel legible en la esquina inferior izquierda.

**c) No pruebes con la cámara** (`main.py`). La cámara la está usando otra sesión; Isaac lo probará en vivo.

---

## 6. Commit

Solo si (a) y (b) salen bien:
```powershell
git -C C:\Users\romer\MatchVision-gemini add src/ui.py config.yaml tests/test_ui.py
git -C C:\Users\romer\MatchVision-gemini status --short
git -C C:\Users\romer\MatchVision-gemini commit -m "Bonus: contador de objetos por clase e ID de seguimiento en pantalla"
```
Verifica con `status` que solo esos 3 archivos estén en el commit.

## 7. Prohibido

- ❌ Trabajar o cambiar de rama en `C:\Users\romer\MatchVision`.
- ❌ Modificar archivos fuera de la sección 4.
- ❌ Cambiar la firma de `UI.dibujar(frame, detecciones, fps)` o romper el comportamiento actual (cajas, colores, barra superior, trackbars).
- ❌ `git push`, `merge`, `rebase`, `reset`, `stash`, `clean` o borrar ramas.
- ❌ `pip install` o dependencias nuevas.
- ❌ Ejecutar `main.py`, `capture.py` o cualquier cosa que abra la cámara.

## 8. Reporte final

Responde **solo** con:
1. Resultado del requisito previo (rama).
2. Resumen de los cambios (qué funciones agregaste y dónde).
3. Salida completa de pytest.
4. Salida del render de prueba (conteo, etiquetas y ruta de la imagen).
5. `git log --oneline -3` de tu carpeta.
6. Cualquier cosa inesperada.
