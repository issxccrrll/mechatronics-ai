# Métricas

Modelo: `models/best.pt` · split: **test** (separado de train/val) · fecha: 2026-10-10

Reproducir: `python scripts/eval.py --model models/best.pt`

## Resultados por clase (protocolo estándar, conf 0.001)

| clase | precision | recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| todas | 0.7799 | 0.6951 | 0.7404 | 0.5362 |
| taza | 0.6485 | 0.7566 | 0.7467 | 0.6113 |
| botella | 0.7543 | 0.438 | 0.5032 | 0.2051 |
| tijeras | 0.7041 | 0.7083 | 0.7415 | 0.4667 |
| destornillador | 0.8815 | 0.676 | 0.7839 | 0.5454 |
| celular | 0.9109 | 0.8966 | 0.9267 | 0.8526 |

## Matriz de confusión (conf 0.3)

Filas = predicción, columnas = clase real; "fondo" = sin objeto.

| pred \ real | taza | botella | tijeras | destornillador | celular | fondo |
|---|---|---|---|---|---|---|
| taza | 79 | 16 | 0 | 0 | 1 | 28 |
| botella | 0 | 61 | 0 | 0 | 1 | 26 |
| tijeras | 0 | 0 | 35 | 0 | 0 | 16 |
| destornillador | 0 | 0 | 3 | 108 | 0 | 12 |
| celular | 0 | 0 | 0 | 0 | 78 | 13 |
| fondo | 21 | 44 | 10 | 35 | 7 | 0 |

![Matriz de confusión](docs/confusion_matrix.png)
