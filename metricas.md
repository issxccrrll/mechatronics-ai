# Métricas

> Plantilla. Se sobrescribe automáticamente con `python scripts/eval.py --model models/best.pt`.

Modelo: `TODO` · split: test · fecha: TODO

## Resultados por clase

| clase | precision | recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| todas | | | | |
| taza | | | | |
| botella | | | | |
| tijeras | | | | |
| destornillador | | | | |
| celular | | | | |

## Matriz de confusión

![Matriz de confusión](docs/confusion_matrix.png)

## Rendimiento (CPU)

| modelo | imgsz | FPS inferencia | FPS pipeline completo |
|---|---|---|---|
| yolo11n.pt (PyTorch) | 640 | ~22.5 | |
| yolo11n OpenVINO | 640 | ~62 | |
