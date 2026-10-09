"""Pre-etiquetado automático en formato YOLO con un modelo existente.

Genera un .txt junto a cada imagen de dataset/raw/<carpeta>/ para luego solo
CORREGIR en la herramienta de etiquetado en vez de dibujar desde cero.

- Con yolo11n.pt (COCO) etiqueta taza, botella, tijeras y celular vía mapeo_clases.
  El destornillador NO existe en COCO: esas imágenes quedan para etiquetar a mano.
- Con best.pt propio (clases ya oficiales) etiqueta las 5 clases.
- Carpeta de una clase (ej. taza/): toda caja detectada se etiqueta con esa clase.
  En mixto/ se respeta la clase del modelo.
- Si no hay cajas, pasadas extra: umbral bajo (--conf-bajo), TTA y foto rotada 180°
  (tazas boca abajo), quedándose con la mejor caja. En --vista salen marcadas "REVISAR".
- Carpeta "fondo": siempre .txt vacío (negativos).
- Imagen sin detecciones (fuera de "fondo"): NO se crea .txt; queda listada en revisar.txt.
- No sobrescribe .txt existentes (protege correcciones manuales) salvo --sobrescribir.

Ej.: python scripts/autolabel.py
     python scripts/autolabel.py --carpetas taza botella --vista
     python scripts/autolabel.py --model models/best.pt --carpetas destornillador
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

import cv2
import yaml
from ultralytics import YOLO

RAIZ = Path(__file__).resolve().parent.parent
EXT = {".jpg", ".jpeg", ".png"}


def cargar_clases():
    with open(RAIZ / "dataset" / "data.yaml", encoding="utf-8") as f:
        names = yaml.safe_load(f)["names"]
    return [names[i] for i in sorted(names)]


def construir_mapeo(nombres_modelo: dict, clases: list[str]) -> dict[int, int]:
    """id del modelo -> id oficial. Usa mapeo_clases de config.yaml o el nombre si ya es oficial."""
    with open(RAIZ / "config.yaml", encoding="utf-8") as f:
        mapeo_cfg = yaml.safe_load(f).get("mapeo_clases") or {}
    salida = {}
    for idm, nombre in nombres_modelo.items():
        oficial = nombre if nombre in clases else mapeo_cfg.get(nombre)
        if oficial in clases:
            salida[idm] = clases.index(oficial)
    return salida


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=str(RAIZ / "models" / "yolo11n.pt"))
    p.add_argument("--raw", default=str(RAIZ / "dataset" / "raw"))
    p.add_argument("--carpetas", nargs="*", help="por defecto todas las de dataset/raw")
    p.add_argument("--conf", type=float, default=0.30)
    p.add_argument("--conf-bajo", type=float, default=0.10, help="segunda pasada si no hubo cajas")
    p.add_argument("--area-min", type=float, default=0.008,
                   help="área mínima (fracción de la imagen) de cajas en las pasadas extra")
    p.add_argument("--no-forzar", action="store_true",
                   help="no forzar la clase de la carpeta (por defecto taza/ -> todo taza)")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--sobrescribir", action="store_true")
    p.add_argument("--vista", action="store_true", help="guarda imágenes con cajas en dataset/revision/")
    args = p.parse_args()

    clases = cargar_clases()
    modelo = YOLO(args.model)
    mapeo = construir_mapeo(modelo.names, clases)
    print(f"Clases que este modelo puede pre-etiquetar: {sorted(clases[i] for i in set(mapeo.values()))}")

    raw = Path(args.raw)
    carpetas = [raw / c for c in args.carpetas] if args.carpetas else sorted(d for d in raw.iterdir() if d.is_dir())
    revisar = []
    detectables = set(mapeo.values())
    for carpeta in carpetas:
        imgs = sorted(f for f in carpeta.glob("*") if f.suffix.lower() in EXT)
        if carpeta.name in clases and clases.index(carpeta.name) not in detectables:
            # Ej. destornillador con COCO: forzar la clase crearía etiquetas basura
            print(f"{carpeta.name:15s} {len(imgs):4d} imgs | este modelo no conoce la clase: etiquetar a mano "
                  f"(o usar --model best.pt tras el primer entrenamiento)")
            revisar += [str(f.relative_to(raw)) for f in imgs if not f.with_suffix(".txt").exists()]
            continue
        conteo, pasadas, vacias, saltadas = Counter(), Counter(), 0, 0
        for img in imgs:
            txt = img.with_suffix(".txt")
            if txt.exists() and not args.sobrescribir:
                saltadas += 1
                continue
            lineas = []
            if carpeta.name != "fondo":
                # En carpeta de una sola clase, toda caja se asigna a esa clase (ej. taza boca abajo -> "botella")
                forzada = clases.index(carpeta.name) if carpeta.name in clases and not args.no_forzar else None
                cajas, pasada = detectar_con_pasadas(modelo, img, args, mapeo)
                pasadas[pasada] += 1
                if forzada is not None and cajas:
                    propias = [c for c in cajas if c[0] == forzada]
                    # Si el modelo ya vio la clase de la carpeta, otras clases son objetos del fondo;
                    # si no, la mejor caja es el objeto mal clasificado (ej. taza boca abajo -> "botella")
                    cajas = propias or [(forzada, cajas[0][1])]
                for cls, (cx, cy, w, h) in cajas:
                    lineas.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
                    conteo[clases[cls]] += 1
                if args.vista:
                    guardar_vista(img, lineas, clases, RAIZ / "dataset" / "revision" / carpeta.name, pasada)
                if not lineas:
                    # Sin .txt queda pendiente; un .txt vacío la convertiría en "fondo"
                    vacias += 1
                    revisar.append(str(img.relative_to(raw)))
                    continue
            txt.write_text("\n".join(lineas) + ("\n" if lineas else ""), encoding="utf-8")
        print(f"{carpeta.name:15s} {len(imgs):4d} imgs | sin cajas: {vacias:3d} | ya tenían .txt: {saltadas:3d} | "
              f"cajas: {dict(conteo)} | por pasada: {dict(pasadas)}")

    (raw / "classes.txt").write_text("\n".join(clases) + "\n", encoding="utf-8")
    if revisar:
        (raw / "revisar.txt").write_text("\n".join(revisar) + "\n", encoding="utf-8")
        print(f"\n{len(revisar)} imágenes sin ninguna caja -> listadas en {raw / 'revisar.txt'} (etiquetar a mano)")
    print("Listo. Ahora corrige las etiquetas en tu herramienta (ver README > Etiquetado).")


def detectar(modelo, imagen, conf: float, imgsz: int, mapeo: dict, augment=False) -> list:
    """[(id_oficial, (cx, cy, w, h) normalizados)] ordenadas por confianza descendente."""
    res = modelo.predict(imagen, conf=conf, imgsz=imgsz, device="cpu", verbose=False, augment=augment)[0]
    filas = zip(res.boxes.conf.tolist(), res.boxes.cls.int().tolist(), res.boxes.xywhn.tolist())
    return [(mapeo[c], tuple(b)) for _, c, b in sorted(filas, reverse=True) if c in mapeo]


def grandes(cajas: list, area_min: float) -> list:
    """Descarta cajas diminutas: en pasadas de umbral bajo suelen ser objetos del fondo."""
    return [c for c in cajas if c[1][2] * c[1][3] >= area_min]


def detectar_con_pasadas(modelo, img: Path, args, mapeo: dict) -> tuple[list, str]:
    """Pasadas de más fiable a menos; solo se usa la siguiente si la anterior no encontró nada.

    normal -> umbral bajo (mejor caja) -> TTA (multi-escala + flip) -> foto rotada 180° (objetos boca abajo).
    """
    imagen = cv2.imread(str(img))
    cajas = detectar(modelo, imagen, args.conf, args.imgsz, mapeo)
    if cajas:
        return cajas, "normal"
    cajas = grandes(detectar(modelo, imagen, args.conf_bajo, args.imgsz, mapeo), args.area_min)[:1]
    if cajas:
        return cajas, "umbral_bajo"
    cajas = grandes(detectar(modelo, imagen, args.conf_bajo, args.imgsz, mapeo, augment=True), args.area_min)[:1]
    if cajas:
        return cajas, "tta"
    cajas = grandes(detectar(modelo, cv2.rotate(imagen, cv2.ROTATE_180), args.conf_bajo, args.imgsz, mapeo),
                    args.area_min)[:1]
    if cajas:  # deshacer la rotación: el centro se refleja, el tamaño no cambia
        cls, (cx, cy, w, h) = cajas[0]
        return [(cls, (1 - cx, 1 - cy, w, h))], "rot180"
    return [], "ninguna"


def guardar_vista(img: Path, lineas: list[str], clases: list[str], destino: Path, pasada: str = ""):
    destino.mkdir(parents=True, exist_ok=True)
    im = cv2.imread(str(img))
    alto, ancho = im.shape[:2]
    for ln in lineas:
        c, cx, cy, w, h = ln.split()
        cx, cy, w, h = float(cx) * ancho, float(cy) * alto, float(w) * ancho, float(h) * alto
        x1, y1, x2, y2 = int(cx - w / 2), int(cy - h / 2), int(cx + w / 2), int(cy + h / 2)
        cv2.rectangle(im, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(im, clases[int(c)], (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    if pasada not in ("", "normal"):  # las pasadas extra son menos fiables: marcarlas para revisar primero
        cv2.putText(im, f"REVISAR ({pasada})", (8, im.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.imwrite(str(destino / img.name), im)


if __name__ == "__main__":
    sys.exit(main())
