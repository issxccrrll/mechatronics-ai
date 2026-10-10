# Contexto para Gemini: script de ensayo de las 20 escenas del jurado

Vas a crear **una herramienta nueva** sin modificar nada existente. Sigue las instrucciones al
pie de la letra. Si algo no coincide con lo descrito, **DETENTE y repórtalo**.

---

## 1. Requisito previo y dónde trabajar

```powershell
git -C C:\Users\romer\MatchVision-gemini branch --show-current
git -C C:\Users\romer\MatchVision-gemini status --short
```
- La rama debe ser `gemini` y el `status` debe salir **vacío** (todo commiteado). Si no, DETENTE y repórtalo.
- **Trabaja SOLO en `C:\Users\romer\MatchVision-gemini`.** Nunca en `C:\Users\romer\MatchVision`, que es de Claude.
- **Python:** `C:\Users\romer\MatchVision\venv\Scripts\python.exe`, con ruta completa. No crees otro venv.

---

## 2. Para qué sirve

La rúbrica del reto ("Mechatronics AI", Hack Knights IA Challenge 2026) da **18 puntos** por:
> "Alcanzar al menos 85 % de aciertos en 20 escenas evaluadas en vivo por el jurado."

El jurado probará:
- distintas posiciones, distancias y orientaciones;
- varios objetos a la vez;
- objetos parcialmente ocultos;
- cambios de iluminación;
- objetos ajenos a las clases oficiales;
- un fondo vacío.

**Tu tarea:** un script que le permita al equipo **ensayar 20 escenas similares** con la cámara,
usando el **mismo pipeline** del detector, y que calcule el porcentaje de aciertos. Así llegan a la
demo sabiendo si pasan del 85%.

---

## 3. Lo que necesitas saber del proyecto (solo lectura)

- **Clases oficiales:** `config.yaml > clases` = `taza, botella, tijeras, destornillador, celular`.
- **`src/types.py`:** `Detection(clase, confianza, bbox, track_id=None, estado=Estado.ACEPTADA, rechazada_por=None, clase_original="")`,
  con `Estado`: `ACEPTADA`, `RECHAZADA`, `DESCONOCIDO`.
- **Así arma el pipeline `main.py`.** Léelo; tu script debe construirlo **igual**:
  ```python
  cfg = cargar_config()                       # src/config.py; acepta --source --conf --model --imgsz
  detector = YoloDetector(cfg["modelo"], cfg["mapeo_clases"], cfg["clases"])
  verificador = construir_verificador(cfg["verificador"])
  filtros = construir_filtros(cfg)
  ui = UI(cfg)
  camara = Camara(cam["source"], cam["ancho"], cam["alto"], cam["reintentos_max"], cam["espera_reintento_s"])
  # por frame:
  frame = camara.leer()                       # None si no hay frame nuevo
  dets = detector.detectar(frame)
  dets = verificador.verificar(frame, dets)
  dets = aplicar_filtros(filtros, dets, frame.shape)
  frame = ui.dibujar(frame, dets, fps)
  seguir = ui.mostrar(frame)                  # False si se presionó q o se cerró la ventana
  ```
- **`cargar_config(argv)`** usa `argparse`. Para agregar argumentos propios (por ejemplo `--listar`)
  sin romperlo, separa tus argumentos con `parse_known_args` y pasa el resto a `cargar_config(resto)`.
- **Para leer teclas**, `UI.mostrar()` ya llama `cv2.waitKey(1)` y solo reconoce `q`. Para detectar
  ESPACIO o `s`, dibuja y muestra el frame **tú mismo** con `cv2.imshow(ui.nombre, frame)` y
  `cv2.waitKey(1)`, en lugar de `ui.mostrar`.

---

## 4. Especificación

### 4.1 Archivos que creas (los únicos permitidos)
- `scripts/ensayo_escenas.py`: el script.
- `tests/test_ensayo.py`: pruebas de la lógica pura.

**No modifiques ningún archivo existente** (`main.py`, `src/*`, `config.yaml`, `README.md`, etc.).

### 4.2 Las 20 escenas
Defínelas como una lista constante `ESCENAS` en el script. Cada escena es un dict con `descripcion`
(texto corto en español) y `esperado` (dict `clase -> cantidad`; vacío `{}` si no debe detectarse nada oficial).
Usa exactamente estas:

| # | descripcion | esperado |
|---|---|---|
| 1 | Taza de frente, distancia media | {taza: 1} |
| 2 | Botella de plástico de pie | {botella: 1} |
| 3 | Tijeras abiertas sobre la mesa | {tijeras: 1} |
| 4 | Destornillador acostado sobre la mesa | {destornillador: 1} |
| 5 | Celular boca arriba sobre la mesa | {celular: 1} |
| 6 | Taza lejos (~1.5 m) | {taza: 1} |
| 7 | Celular muy cerca de la cámara | {celular: 1} |
| 8 | Botella acostada | {botella: 1} |
| 9 | Destornillador en la mano, inclinado | {destornillador: 1} |
| 10 | Tijeras cerradas, de lado | {tijeras: 1} |
| 11 | Taza y celular juntos | {taza: 1, celular: 1} |
| 12 | Botella, tijeras y destornillador juntos | {botella: 1, tijeras: 1, destornillador: 1} |
| 13 | Las 5 clases juntas | {taza: 1, botella: 1, tijeras: 1, destornillador: 1, celular: 1} |
| 14 | Taza parcialmente tapada con la mano | {taza: 1} |
| 15 | Destornillador parcialmente oculto detrás de la botella | {botella: 1, destornillador: 1} |
| 16 | Luz tenue: celular y tijeras | {celular: 1, tijeras: 1} |
| 17 | Contraluz o luz fuerte: botella | {botella: 1} |
| 18 | Objeto ajeno: pluma o lápiz | {} |
| 19 | Objetos ajenos: vaso, control remoto, cuchara | {} |
| 20 | Fondo vacío (mesa sin objetos) | {} |

### 4.3 Flujo del script
1. `--listar`: imprime las 20 escenas (número, descripción y esperado) y **sale sin abrir cámara ni modelo**.
2. Sin `--listar`: arma el pipeline igual que `main.py` y abre la ventana con detección en vivo.
3. Sobre el video dibuja (además de lo que dibuja `ui.dibujar`) un rótulo con:
   `Escena N/20: <descripcion>` y `ESPACIO = evaluar | s = saltar | q = terminar`.
4. Con **ESPACIO**: toma las **siguientes 30 detecciones de frame** (~1–2 s) a través del pipeline completo.
   Por cada frame calcula el conteo de detecciones **ACEPTADAS** por clase oficial. El resultado de la
   escena es el **conteo más frecuente** (la moda) entre esos 30 frames.
5. **Acierto** si el resultado es **exactamente igual** al esperado: mismas clases y mismas cantidades.
   Una clase oficial de más o de menos es fallo; en las escenas `{}`, cualquier detección oficial aceptada es fallo.
   Las detecciones `DESCONOCIDO` y `RECHAZADA` **no cuentan**.
6. Muestra **ACIERTO** (verde) o **FALLO** (rojo) con lo obtenido frente a lo esperado durante ~2 s,
   y pasa a la siguiente escena.
7. **`s`** salta la escena (no cuenta en el total). **`q`** termina antes.
8. **Al terminar** (o con `q`):
   - Imprime una tabla en consola y el resumen: `Aciertos: X/Y = Z% (meta: 85%)`, donde Y son las escenas evaluadas, sin contar las saltadas.
   - Guarda un CSV en `logs/ensayo_AAAAMMDD_HHMMSS.csv` con columnas `escena, descripcion, esperado, obtenido, acierto, frames`. Crea `logs/` si no existe; git ya la ignora.
   - Cierra cámara y ventana **siempre** (`try/finally`), como `main.py`.

### 4.4 Diseño para poder probarlo sin cámara
Separa la lógica en **funciones puras de módulo** (sin cámara ni ventana), por ejemplo:
- `conteo_aceptadas(detecciones, clases) -> dict[str, int]`: solo clases oficiales con cantidad mayor que 0.
- `moda_conteos(conteos: list[dict]) -> dict`: el conteo más frecuente; en empate, el que aparece primero.
- `es_acierto(esperado: dict, obtenido: dict) -> bool`: ignora claves con valor 0.
- `resumen(resultados: list[dict]) -> tuple[int, int, float]`: aciertos, evaluadas y porcentaje (0.0 si no hay evaluadas).

`main()` solo orquesta. Mantén el script corto y legible (idealmente menos de ~170 líneas), con comentarios en español.

### 4.5 Pruebas (`tests/test_ensayo.py`)
Mismo estilo que `tests/test_filters.py` (usa `sys.path.insert`; importa desde `scripts/ensayo_escenas.py`
agregando `scripts` al path, o con `importlib`). **El import no debe abrir cámara ni cargar el modelo**:
todo eso va dentro de `main()`. Cubre al menos:
- `conteo_aceptadas` ignora RECHAZADA y DESCONOCIDO y cuenta repetidas.
- `moda_conteos` devuelve el conteo mayoritario, y `{}` cuando la mayoría de frames no tiene nada.
- `es_acierto`: igual → True; clase de más → False; cantidad distinta → False; `{}` frente a `{}` → True; `{}` frente a `{"taza": 1}` → False.
- `resumen` con lista vacía → `(0, 0, 0.0)`, y un caso normal, por ejemplo 17 de 20 = 85.0.
- `ESCENAS` tiene 20 elementos y todas las claves de `esperado` son clases oficiales.

---

## 5. Verificación (obligatoria antes de commitear)

Desde `C:\Users\romer\MatchVision-gemini`:
```powershell
C:\Users\romer\MatchVision\venv\Scripts\python.exe -m pytest -q tests
C:\Users\romer\MatchVision\venv\Scripts\python.exe scripts\ensayo_escenas.py --listar
C:\Users\romer\MatchVision\venv\Scripts\python.exe scripts\ensayo_escenas.py --help
```
- **Todos** los tests pasan, los existentes y los nuevos.
- `--listar` imprime las 20 escenas **sin** abrir cámara ni cargar el modelo.
- `--help` muestra tus opciones (y no se cae).

**NO ejecutes el script sin `--listar`.** Eso abre la cámara, que está en uso por otra sesión.
Isaac lo probará en vivo.

## 6. Commit

```powershell
git -C C:\Users\romer\MatchVision-gemini add scripts/ensayo_escenas.py tests/test_ensayo.py
git -C C:\Users\romer\MatchVision-gemini status --short
git -C C:\Users\romer\MatchVision-gemini commit -m "Script de ensayo de 20 escenas del jurado con porcentaje de aciertos"
```
Verifica que **solo** esos 2 archivos estén en el commit.

## 7. Prohibido

- ❌ Trabajar o cambiar de rama en `C:\Users\romer\MatchVision`.
- ❌ Modificar cualquier archivo existente.
- ❌ Abrir la cámara (no ejecutes `main.py`, `capture.py` ni el script sin `--listar`).
- ❌ `git push`, `merge`, `rebase`, `reset`, `stash`, `clean` o borrar ramas o archivos.
- ❌ `pip install` o dependencias nuevas (solo lo que ya usa el proyecto: opencv, numpy, pyyaml, ultralytics).
- ❌ Tocar `dataset\` o detener procesos de Python.

## 8. Reporte final

Responde **solo** con:
1. Resultado del requisito previo.
2. Resumen del diseño: funciones y flujo.
3. Salida completa de pytest.
4. Salida de `--listar`.
5. `git log --oneline -3` de tu carpeta.
6. Cualquier cosa inesperada o decisión que hayas tenido que tomar.
