# Contexto para Gemini: Paso 2 (dividir el dataset y generar dataset.zip)

Eres un asistente que va a ejecutar **una sola tarea acotada** en un proyecto de visión
artificial. Sigue las instrucciones **al pie de la letra**. Si algo no coincide con lo que se
describe aquí (un comando falla, un número es muy distinto al esperado, falta una carpeta),
**DETENTE y repórtalo**. No improvises soluciones ni modifiques código.

---

## 1. Proyecto

- **Qué es:** detector de objetos en vivo (YOLO11n + OpenCV) para el reto "Mechatronics AI" del
  Hack Knights IA Challenge 2026. Detecta 5 clases: `taza`, `botella`, `tijeras`, `destornillador`, `celular`.
- **Sistema:** Windows 11, terminal **PowerShell 5.1**.
- **Carpeta del proyecto:** `C:\Users\romer\MatchVision`. Ejecuta **todos** los comandos desde aquí.
- **Python:** usa **siempre** `.\venv\Scripts\python.exe`. Nunca uses `python` o `py` a secas:
  el Python global es 3.14 y no tiene las librerías.

### Estructura relevante
```
C:\Users\romer\MatchVision\
  dataset\
    data.yaml            <- NO TOCAR (define las 5 clases; no lleva "path" a propósito)
    raw\                 <- FOTOS ORIGINALES + etiquetas. NO TOCAR NADA AQUÍ
      taza\  tijeras\  destornillador\  celular\  mixto\
        s20261009_115755_0001.jpg   (foto)
        s20261009_115755_0001.txt   (etiqueta YOLO de esa foto)
      classes.txt  revisar.txt
    revision\            <- vistas previas; ignorar
  scripts\split_dataset.py  <- script que divide (ya probado; NO modificarlo)
  mechatronics-ai\       <- copia del repo que alguien clonó aquí; IGNORAR, no tocar
```

### Formato de las etiquetas (YOLO)
Cada `.txt` tiene una línea por objeto: `clase cx cy w h`.
- `clase` es un entero de 0 a 4 (0 taza, 1 botella, 2 tijeras, 3 destornillador, 4 celular).
- `cx cy w h` son decimales entre 0 y 1.

Una foto **sin** `.txt` está pendiente de etiquetar, y el script la omite a propósito.

---

## 2. Tu tarea (en este orden exacto)

### Paso A: Verificar que el pre-etiquetado terminó
Hay un proceso de pre-etiquetado (OWLv2) que escribe `.txt` en `dataset\raw\destornillador` y
`dataset\raw\celular`. **No continúes si sigue corriendo.** Comprueba:
```powershell
Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'autolabel' } | Select-Object ProcessId, CommandLine
```
- **Si sale algún proceso:** DETENTE y responde "El pre-etiquetado sigue corriendo; avísame cuando termine".
- **Si no sale nada:** continúa.

### Paso B: Inventario (solo lectura)
```powershell
Get-ChildItem dataset\raw -Directory | ForEach-Object { "{0}: {1} jpg, {2} txt" -f $_.Name, (Get-ChildItem $_.FullName -Filter *.jpg).Count, (Get-ChildItem $_.FullName -Filter *.txt).Count }
```
Valores aproximados esperados:

| Carpeta | jpg | txt |
|---|---|---|
| taza | 505 | ~475 |
| tijeras | 200 | ~195 |
| destornillador | 834 | 650–800 |
| celular | 549 | 420–540 |
| mixto | 200 | 200 |

Si alguna carpeta tiene 0 jpg, o aparece otra carpeta con fotos dentro de una subcarpeta (por
ejemplo `celular\celular\`), DETENTE y repórtalo.

### Paso C: Comprobar si ya existe una división previa
```powershell
Test-Path dataset\train, dataset\val, dataset\test, dataset.zip
```
- **Si alguno es `True`:** DETENTE y pregunta a Isaac si se puede reemplazar. **No borres nada por tu cuenta.**
- **Si todos son `False`:** continúa.

### Paso D: Dividir el dataset
```powershell
.\venv\Scripts\python.exe scripts\split_dataset.py
```
Qué hace el script:
- Copia (no mueve) las fotos etiquetadas de `dataset\raw\` a `dataset\train\`, `dataset\val\` y `dataset\test\`, cada una con `images\` y `labels\`.
- Agrupa por sesión de captura (la parte `sAAAAMMDD_HHMMSS` del nombre).
- Si una clase tiene menos de 3 sesiones, divide cada sesión en bloques 70/15/15.

Mensajes **normales** que puede imprimir (no son errores):
- `[aviso] <clase>: solo N sesión(es); se divide cada sesión en bloques 70/15/15...`
- `[aviso] N imágenes omitidas por no tener etiqueta (.txt)...`

Al final imprime `Imágenes por split: {...}`. **Copia esa línea en tu reporte.**

### Paso E: Validar la división (solo lectura)
Pega este bloque completo en PowerShell:
```powershell
@'
from pathlib import Path
from collections import Counter
raiz = Path("dataset")
problemas = 0
for sp in ("train", "val", "test"):
    imgs = {p.stem for p in (raiz / sp / "images").glob("*.jpg")}
    labs = {p.stem for p in (raiz / sp / "labels").glob("*.txt")}
    clases = Counter()
    for t in (raiz / sp / "labels").glob("*.txt"):
        for ln in t.read_text(encoding="utf-8").split("\n"):
            if not ln.strip():
                continue
            v = ln.split()
            ok = len(v) == 5 and v[0] in ("0", "1", "2", "3", "4") and all(0 <= float(x) <= 1 for x in v[1:])
            if not ok:
                problemas += 1; print("ETIQUETA MALA:", t.name, ln)
            else:
                clases[int(v[0])] += 1
    sin_label = len(imgs - labs)
    print(f"{sp}: {len(imgs)} imagenes, {len(labs)} etiquetas, imagenes sin etiqueta: {sin_label}, cajas por clase: {dict(sorted(clases.items()))}")
nombres = [p.name for sp in ("train", "val", "test") for p in (raiz / sp / "images").glob("*.jpg")]
dup = len(nombres) - len(set(nombres))
print("imagenes repetidas entre splits:", dup)
print("PROBLEMAS:", problemas + dup)
'@ | .\venv\Scripts\python.exe -
```
Resultado esperado:
- `PROBLEMAS: 0`.
- train, val y test con imágenes.
- `imagenes sin etiqueta: 0`.
- Cajas de las clases 0, 2, 3 y 4 en los tres splits. La clase 1 (botella) solo aparece por las fotos de mixto; es normal que tenga pocas.

**Si `PROBLEMAS` es mayor que 0, DETENTE y repórtalo.**

### Paso F: Crear dataset.zip
El zip debe contener **una carpeta `dataset/` en la raíz** con `data.yaml`, `train/`, `val/` y
`test/`. **No incluyas** `raw/` ni `revision/`. Pega este bloque completo:
```powershell
@'
import zipfile
from pathlib import Path
raiz = Path("dataset")
n = 0
with zipfile.ZipFile("dataset.zip", "w", zipfile.ZIP_STORED) as z:
    z.write(raiz / "data.yaml", "dataset/data.yaml"); n += 1
    for sp in ("train", "val", "test"):
        for f in sorted((raiz / sp).rglob("*")):
            if f.is_file():
                z.write(f, f"dataset/{f.relative_to(raiz).as_posix()}"); n += 1
print("archivos en el zip:", n)
'@ | .\venv\Scripts\python.exe -
```

### Paso G: Verificar el zip (solo lectura)
```powershell
@'
import zipfile
z = zipfile.ZipFile("dataset.zip")
nombres = z.namelist()
print("total:", len(nombres))
print("raices:", sorted({n.split("/")[0] for n in nombres}))
print("subcarpetas:", sorted({"/".join(n.split("/")[:2]) for n in nombres}))
print("tiene data.yaml:", "dataset/data.yaml" in nombres)
print("incluye raw o revision:", any(n.startswith(("dataset/raw", "dataset/revision")) for n in nombres))
'@ | .\venv\Scripts\python.exe -
(Get-Item dataset.zip).Length / 1MB
```
Esperado:
- `raices: ['dataset']`.
- Subcarpetas: `dataset/data.yaml`, `dataset/test`, `dataset/train`, `dataset/val`.
- `tiene data.yaml: True`.
- `incluye raw o revision: False`.
- Tamaño de unos cientos de MB.

---

## 3. Prohibido (no lo hagas aunque parezca buena idea)

- ❌ Modificar, mover, renombrar o borrar **cualquier cosa** dentro de `dataset\raw\`.
- ❌ Editar archivos de código o configuración: `scripts\*.py`, `src\*`, `main.py`, `config.yaml`, `dataset\data.yaml`, `.gitignore`, `README.md`.
- ❌ Borrar `dataset\train`, `dataset\val`, `dataset\test` o `dataset.zip` sin permiso de Isaac.
- ❌ Usar la opción `--reemplazar` de `split_dataset.py` sin permiso de Isaac.
- ❌ Comandos de git: `commit`, `push`, `pull`, `checkout`, `reset`, `clean`, etc.
- ❌ `pip install` o instalar cualquier cosa.
- ❌ Ejecutar `autolabel.py`, `autolabel_zeroshot.py`, `train.py`, `main.py` o `capture.py`.
- ❌ Tocar la carpeta `mechatronics-ai\`.
- ❌ Subir archivos a internet (Drive, GitHub, etc.). Isaac sube el zip a mano.
- ❌ Crear scripts nuevos en el proyecto. Usa solo los bloques de este documento.

## 4. Reporte final (formato)

Al terminar, responde **solo** con:
1. Resultado del Paso A (proceso terminado: sí/no).
2. Tabla del inventario (Paso B).
3. La línea `Imágenes por split: {...}` y los avisos que imprimió el Paso D.
4. La salida completa del Paso E.
5. La salida del Paso G y el tamaño del zip en MB.
6. Cualquier cosa inesperada.

Si en cualquier paso algo no coincide, detente en ese paso y reporta hasta ahí.
