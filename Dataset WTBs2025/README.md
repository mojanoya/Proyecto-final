# WTBs2025 limpio a 2 clases (damage / dirt)

Dataset WTBs2025 (Zhang et al., Scientific Data 2026) preparado para combinar con el
dataset DTU de Kaggle. La guía completa está en `WTBs2025_2cls_guia.pdf`.

| Conjunto | Imágenes | Cajas damage | Cajas dirt |
|---|---:|---:|---:|
| train | 5 385 | 7 786 | 0 |
| val   | 1 086 | 1 581 | 0 |
| test  | 1 073 | 1 566 | 0 |

- Imágenes 640×480 (aspecto 4:3 corregido), labels YOLO sin duplicados.
- Split 70/15/15 por grupo de casi-duplicados (pHash + ORB), sin fuga de datos.
- Las 9 clases originales pasan a `0: damage`; `manifest.csv` guarda la clase original.

## Archivos

- `WTBs2025_2cls_guia.pdf`: cómo se preparó el dataset y cómo combinarlo con el DTU.
- `audit_dataset.py`: auditoría del dataset original (clases, duplicados, tamaño de cajas).
- `split_wtbs2025.py`: aspecto 4:3, duplicados y split train/val/test.
- `make_2class.py`: re-etiquetado a 2 clases y ZIP para Kaggle.

## Dónde está el dataset

El ZIP (`WTBs2025_2cls.zip`) no va en el repositorio por su tamaño: https://drive.google.com/drive/u/0/folders/12zPgn9EhGEW9EmUbVqNaAOpePBMj37Eb
