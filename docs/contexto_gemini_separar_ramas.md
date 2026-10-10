# Contexto para Gemini: separar tu trabajo en tu propia carpeta

Vas a ejecutar **una tarea acotada de git**. Sigue los pasos **en orden y al pie de la letra**.
Si la salida de cualquier comando no coincide con lo esperado, **DETENTE y repórtalo**. No
improvises: no uses `stash`, `reset`, `checkout -- <archivo>`, `clean` ni `rebase`.

---

## 1. Situación actual (por qué hay que hacer esto)

- **Proyecto:** `C:\Users\romer\MatchVision`. Sistema: Windows 11, **PowerShell 5.1**.
- **Hay dos asistentes** trabajando en el mismo repositorio git:
  - **Claude**, en la rama `camara-mejoras`.
  - **Tú (Gemini)**, en la rama `gemini`.
- **El problema:** ambos usan la **misma carpeta**, que ahora está en la rama `gemini`. En esa carpeta
  hay cambios **sin commitear de los dos, mezclados**:

| Dueño | Archivos | Qué hacer con ellos |
|---|---|---|
| **Gemini (tú)** | `src/camera.py`, `main.py`, `tests/test_camera.py` | Commitearlos en `gemini` |
| **Claude** | `.gitignore`, `README.md`, `scripts/split_dataset.py`, `scripts/autolabel_zeroshot.py`, `pytest.ini`, `docs/` | **NO tocarlos.** Se quedan sin commitear; Claude los commitea después |

- **El objetivo:** que tus cambios queden en `gemini`, que esta carpeta vuelva a `camara-mejoras`
  (para Claude) y que tú tengas **tu propia carpeta**: `C:\Users\romer\MatchVision-gemini`, creada con
  `git worktree`. Es el mismo repo, con otra carpeta y tu rama.
- **Hecho conocido:** las ramas locales `gemini`, `camara-mejoras` y `main` apuntan al mismo commit, `bb6634b`.

**Importante:** en la carpeta hay un proceso en segundo plano (pre-etiquetado) que escribe
archivos en `dataset\raw\`. Esa carpeta está ignorada por git y **no debes tocarla**.

---

## 2. Pasos

Ejecuta todo desde `C:\Users\romer\MatchVision`:
```powershell
cd C:\Users\romer\MatchVision
```

### Paso 1: Verificar el estado (solo lectura)
```powershell
git branch --show-current
git status --short
git rev-parse --short gemini camara-mejoras
```
Esperado:
- Rama actual: `gemini`.
- Ambos hashes: `bb6634b`.
- `git status --short` muestra **exactamente** estos archivos (el orden puede variar):
```
 M .gitignore
 M README.md
 M main.py
 M scripts/split_dataset.py
 M src/camera.py
?? docs/
?? pytest.ini
?? scripts/autolabel_zeroshot.py
?? tests/test_camera.py
```
**Si la rama no es `gemini`, los hashes no son `bb6634b` o aparecen archivos distintos, DETENTE y reporta la salida.**

### Paso 2: Commitear SOLO tus 3 archivos en `gemini`
```powershell
git add src/camera.py main.py tests/test_camera.py
git status --short
```
Verifica que **solo** esos 3 tengan la marca de agregado (`M ` o `A ` en la primera columna) y que
los demás sigan como ` M` o `??`. Si se agregó otro archivo, quítalo **solo del área de preparación** con
`git restore --staged <archivo>`. **Nunca** uses `git restore` sin `--staged`: borraría los cambios de Claude.

Luego:
```powershell
git commit -m "Cámara con variable de condición (sin busy-waiting) y pruebas de sincronización"
git log --oneline -2
```
Esperado: un commit nuevo encima de `bb6634b`.

### Paso 3: Devolver esta carpeta a `camara-mejoras` (los cambios de Claude viajan solos)
```powershell
git switch camara-mejoras
git branch --show-current
git status --short
```
Esperado:
- Rama actual: `camara-mejoras`.
- Solo quedan los archivos de Claude: `.gitignore`, `README.md`, `scripts/split_dataset.py`, `docs/`, `pytest.ini`, `scripts/autolabel_zeroshot.py`.
- `src/camera.py`, `main.py` y `tests/test_camera.py` **ya no aparecen**: están guardados en tu commit de `gemini`.

**Si `git switch` da error o pide hacer stash, DETENTE y reporta el error textual. No fuerces nada.**

### Paso 4: Crear tu carpeta propia
```powershell
git worktree add C:\Users\romer\MatchVision-gemini gemini
git worktree list
```
Esperado: dos líneas.
- `C:/Users/romer/MatchVision` en `[camara-mejoras]`.
- `C:/Users/romer/MatchVision-gemini` en `[gemini]`.

### Paso 5: Comprobar tu carpeta (solo lectura)
```powershell
cd C:\Users\romer\MatchVision-gemini
git branch --show-current
git log --oneline -2
Test-Path src\camera.py, tests\test_camera.py
C:\Users\romer\MatchVision\venv\Scripts\python.exe -m pytest -q tests
```
Esperado:
- Rama `gemini`, con tu commit arriba.
- Los dos `Test-Path` en `True`.
- Los tests pasan.

---

## 3. Cómo trabajar desde ahora

- **Tu carpeta es `C:\Users\romer\MatchVision-gemini`.** Todos tus cambios de código van ahí, en la rama `gemini`.
- **No hay `venv` en tu carpeta.** Usa el Python del proyecto con ruta completa:
  `C:\Users\romer\MatchVision\venv\Scripts\python.exe`. No crees otro venv.
- **El dataset (`dataset\raw\`) solo existe en `C:\Users\romer\MatchVision`.** Git lo ignora.
- **Para el Paso 2 del dataset** (`docs/contexto_gemini_paso2.md`), trabaja en `C:\Users\romer\MatchVision`,
  pero **solo** con los comandos de ese documento. Ahí no se edita código.
- **No cambies de rama en `C:\Users\romer\MatchVision`.** Esa carpeta es de Claude.
- **Al final, Isaac unirá las ramas** con un Pull Request (`gemini` → `camara-mejoras` o `main`).

## 4. Prohibido

- ❌ Commitear, modificar o descartar los archivos de Claude de la tabla de arriba.
- ❌ `git stash`, `git reset`, `git restore` sin `--staged`, `git checkout -- <archivo>`, `git clean`, `git rebase`, `git merge`.
- ❌ `git push` (Isaac decide cuándo subir).
- ❌ Borrar ramas, archivos o carpetas.
- ❌ Tocar `dataset\` o detener procesos de Python que estén corriendo.

## 5. Reporte final

Responde **solo** con:
1. Salida del Paso 1.
2. Salida de `git log --oneline -2` del Paso 2.
3. Salida de `git status --short` del Paso 3.
4. Salida de `git worktree list` del Paso 4.
5. Salida del Paso 5, incluido el resultado de pytest.
6. Cualquier cosa inesperada.

Si algo falla en un paso, detente ahí y reporta hasta ese punto.
