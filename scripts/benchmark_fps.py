"""Mide FPS promedio en CPU de un modelo (solo inferencia, o pipeline con cámara).

Ej.: python scripts/benchmark_fps.py --model models/yolo11n.pt --frames 200
     python scripts/benchmark_fps.py --model models/yolo11n_openvino_model --source 0
"""
import argparse
import time

import cv2
import numpy as np
from ultralytics import YOLO


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="models/yolo11n.pt")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--frames", type=int, default=200)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--source", default=None, help="si se indica, usa frames reales de la cámara")
    args = p.parse_args()

    modelo = YOLO(args.model, task="detect")
    cap = None
    if args.source is not None:
        cap = cv2.VideoCapture(int(args.source) if args.source.isdigit() else args.source)
    rng = np.random.default_rng(0)
    sintetico = rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)

    def frame():
        if cap is not None:
            ok, f = cap.read()
            if ok:
                return f
        return sintetico

    for _ in range(args.warmup):
        modelo.predict(frame(), imgsz=args.imgsz, device="cpu", verbose=False)
    tiempos = []
    for _ in range(args.frames):
        f = frame()
        t0 = time.perf_counter()
        modelo.predict(f, imgsz=args.imgsz, device="cpu", verbose=False)
        tiempos.append(time.perf_counter() - t0)
    if cap is not None:
        cap.release()

    t = np.array(tiempos)
    print(f"Modelo: {args.model} | imgsz {args.imgsz} | {args.frames} frames")
    print(f"FPS promedio: {1 / t.mean():.1f} | latencia media {t.mean() * 1000:.1f} ms | p95 {np.percentile(t, 95) * 1000:.1f} ms")


if __name__ == "__main__":
    main()
