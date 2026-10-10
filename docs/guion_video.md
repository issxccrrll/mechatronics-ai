# Guion del video de demostración (máx. 2:00)

Video de **1:50**. Pantalla de la laptop: terminal + ventana `Mechatronics AI`.
Voz en off (o quien graba habla en vivo). Un solo take continuo, sin cortes si se puede.

**Modelo:** v0 (`models/best_openvino_model`). No citar números que no estén en `metricas.md`.

La primera inferencia OpenVINO tarda ~10 s (`[detector] Calentando modelo...`) **antes** de abrir la ventana. Eso entra en la toma 1.

---

## 1. Tabla de tomas

| # | Tiempo | En pantalla | Objetos / acción | Voz en off |
|---|---|---|---|---|
| 1 | 0:00–0:12 | Terminal en la raíz del repo. Se escribe y ejecuta el comando. Sale `[detector] Calentando modelo...` y luego la ventana `Mechatronics AI` (FPS, cajas, contador abajo a la izquierda, trackbars). | Mesa **vacía**. Cámara ya enchufada y apuntando a la mesa. | «Detector en vivo para el reto Mechatronics AI. Cinco clases: taza, botella, tijeras, destornillador y celular. Lo demás no recibe una clase oficial.» |
| 2 | 0:12–0:18 | Ventana: mesa vacía. Contador en 0. Sin cajas verdes. | Nada sobre la mesa. Esperar 2 s. | «Mesa vacía: el sistema no inventa detecciones.» |
| 3 | 0:18–0:28 | Caja verde `taza`, confianza e ID. Contador: taza 1. | Colocar **taza**. Quietos 1–2 s (el filtro de persistencia pide 3 de 5 cuadros). | «Taza.» |
| 4 | 0:28–0:38 | Caja verde `botella`. | Quitar la taza. Colocar **botella** de pie. Quietos 1–2 s. | «Botella.» |
| 5 | 0:38–0:48 | Caja verde `tijeras`. | Quitar la botella. Colocar **tijeras** abiertas. Quietos 1–2 s. | «Tijeras.» |
| 6 | 0:48–0:58 | Caja verde `destornillador`. | Quitar las tijeras. Colocar **destornillador** sobre la mesa. Quietos 1–2 s. | «Destornillador.» |
| 7 | 0:58–1:08 | Caja verde `celular`. | Quitar el destornillador. Colocar **celular** boca arriba. Quietos 1–2 s. | «Celular.» |
| 8 | 1:08–1:20 | Varias cajas verdes. El **contador por clase** muestra más de un 1. | Dejar celular. Añadir taza, botella, tijeras y destornillador (las 5 juntas, sin amontonar). Quietos 2 s. | «Varios objetos a la vez. El panel cuenta cada clase aceptada.» |
| 9 | 1:20–1:28 | Sin caja verde sobre el objeto ajeno. Si aparece caja, debe ser gris `desconocido` o roja rechazada, **nunca** una clase oficial. | Quitar todo. Poner **pluma o vaso**. Quietos 2 s. | «Objeto ajeno: no se le asigna taza, botella, tijeras, destornillador ni celular.» |
| 10 | 1:28–1:36 | Caja verde sobre la parte visible (p. ej. taza). | Quitar el ajeno. Taza **parcialmente tapada** con la mano (que se vea un buen trozo). Quietos 2 s. | «Aunque esté tapado a medias, sigue detectando lo visible.» |
| 11 | 1:36–1:44 | Sigue la detección (taza o botella) con la luz distinta. | Misma taza. Atenuar la lámpara o tapar un poco la luz (no a oscuras). 2 s. | «Cambio de iluminación.» |
| 12 | 1:44–1:50 | Trackbar **Umbral %** se mueve; cajas pueden pasar a rojo `[ClassThresholdFilter]`. | Dejar la taza. Subir el umbral con el mouse (p. ej. de 30 hacia 70–80) y bajarlo otra vez. | «El umbral se ajusta en vivo con la barra.» |
| 13 | 1:50–2:00 | Frame negro: *Sin senal. Reintentando conexion (N)...* (así está en el código; el README dice «Reintentando conexión»). Luego la ventana se cierra. | **Desenchufar la webcam USB.** Esperar el mensaje. Tecla **q**. | «Si se pierde la cámara, reintenta sin cerrarse. Con q salimos.» |

Duración total: **2:00**. Si vas corto, alarga las tomas 8 y 9. Si vas largo, recorta 3–7 a ~8 s cada una.

---

## 2. Consejos de grabación

- Objetos **sobre la mesa y quietos 1–2 s** antes de hablar de “detectado”. El `PersistenceFilter` exige aparecer en **3 de los últimos 5** cuadros; si mueves la mano al hablar, la caja parpadea o se pone roja.
- Luz uniforme de frente. Evita contraluz fuerte salvo la toma 11.
- Laptop **conectada al cargador**. Cierra navegador, Discord, actualizaciones y otros programas. No grabes con la CPU al 100 %.
- Webcam **USB** (no la integrada): es la única forma limpia de hacer la toma 13. Prueba el desenchufe una vez antes de grabar. Si solo hay webcam integrada, esa toma no se puede hacer desconectando el cable; no improvises tapar la lente (siguen llegando frames negros y no sale el mensaje de reconexión).
- Encuadre: se debe leer el contador, las etiquetas y los trackbars. Ventana maximizada; terminal visible solo en la toma 1 (o Picture-in-Picture del comando).
- Resolución de cámara del demo: 640×480 (`config.yaml`). No cambies `--imgsz` para el video.
- Cómo grabar en Windows:
  - **Xbox Game Bar:** `Win+G`, luego `Win+Alt+R` para iniciar/parar. Graba la pantalla (y el micrófono si la voz va en el mismo archivo).
  - **Herramienta Recortes:** `Win+Shift+S` no graba video continuo; usa **Grabar pantalla** en Recortes (Windows 11) o Game Bar.
- Voz: un poco más lenta que al hablar normal. No cites mAP ni precision; esos números están en `metricas.md` y son del **modelo v0**.
- Si una clase no sale en verde: no la fuerces en cámara. Corta, coloca el objeto de frente y a distancia media, espera 2 s, sigue. Las tijeras son la clase más débil del v0 (recall bajo en `metricas.md`).

### Comando de la toma 1

En PowerShell, **desde la raíz del repo**, venv activado:

```powershell
python main.py --source 0
```

Si la USB no es el índice 0, prueba `--source 1` **antes** de grabar y deja ese número en el video.

---

## 3. Checklist antes de grabar

- [ ] Cargador conectado; batería no en ahorro.
- [ ] `venv` activado; `python main.py --source 0` (o 1) abre la ventana sin error.
- [ ] Calentamiento OpenVINO ya visto una vez (la **primera** vez del día tarda ~10 s; en el video puede verse).
- [ ] Webcam USB en `--source` correcto; cable accesible para desenchufar.
- [ ] Mesa despejada; fondo simple.
- [ ] A la mano, en orden: taza, botella, tijeras, destornillador, celular, pluma o vaso.
- [ ] Luz lista (y cómo atenuarla en la toma 11).
- [ ] Trackbar **Umbral %** visible; mouse listo.
- [ ] Notificaciones de Windows silenciadas; barra de tareas oculta si estorba.
- [ ] Micrófono de prueba (5 s) o voz aparte para mezclar después.
- [ ] Cronómetro visible para quien coloca objetos (celular boca arriba).
- [ ] Un ensayo completo **sin grabar** (sobre todo toma 9 y 13).
- [ ] Game Bar o Recortes ya abiertos; destino de los clips conocido.
- [ ] Tras grabar: el archivo dura ≤ 2:00; se oye la voz; se lee el contador en la toma 8; se ve el mensaje al desconectar.
