"""
Auditoría de un dataset YOLO (detección) organizado como:
    DATASET/
        data.yaml
        <clase>/images/*.jpg
        <clase>/labels/*.txt

Uso:
    pip install pyyaml pillow numpy
    python audit_dataset.py "D:/Proyecto Final/DRON/DATASET"

Reporta:
  - nombres de clase de data.yaml y qué IDs aparecen en cada carpeta
  - líneas mal formadas (≠ 5 valores -> ¿polígonos de segmentación?) o fuera de [0, 1]
  - cajas duplicadas dentro del mismo archivo
  - imágenes sin label y labels sin imagen
  - tamaño de las cajas en píxeles (lado menor) y cuántas son chicas
  - grupos de augmentation de Roboflow (mismo origen, varios archivos)
"""
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
SMALL_PX = 8  # umbral de "caja chica" en px (lado menor), ajustable

root = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
cfg = yaml.safe_load((root / "data.yaml").read_text(encoding="utf-8"))
names = cfg.get("names", [])
if isinstance(names, dict):
    names = [names[k] for k in sorted(names)]
print(f"data.yaml -> nc={cfg.get('nc')}  names:")
for i, n in enumerate(names):
    print(f"  {i}: {n}")
print()

rf_re = re.compile(r"^(.*?)_(jpg|jpeg|png)\.rf\.[0-9a-f]+$", re.I)
groups = defaultdict(set)          # origen -> archivos
min_side_px = []
dup_files = dup_lines = bad_lines = 0
sizes = Counter()

for cls_dir in sorted(p for p in root.iterdir() if (p / "labels").is_dir()):
    lbl_dir, img_dir = cls_dir / "labels", cls_dir / "images"
    imgs = {p.stem: p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXT} if img_dir.is_dir() else {}
    lbls = {p.stem: p for p in lbl_dir.glob("*.txt")}
    ids = Counter()

    for stem, txt in lbls.items():
        rows = [l.split() for l in txt.read_text().splitlines() if l.strip()]
        uniq = set(map(tuple, rows))
        if len(uniq) < len(rows):
            dup_files += 1
            dup_lines += len(rows) - len(uniq)

        W = H = None
        if stem in imgs:
            with Image.open(imgs[stem]) as im:
                W, H = im.size
            sizes[(W, H)] += 1

        for r in uniq:
            if len(r) != 5:
                bad_lines += 1
                continue
            c, x, y, w, h = int(r[0]), *map(float, r[1:])
            if not all(0.0 <= v <= 1.0 for v in (x, y, w, h)):
                bad_lines += 1
            ids[c] += 1
            if W:
                min_side_px.append(min(w * W, h * H))

        m = rf_re.match(stem)
        groups[(m.group(1) if m else stem)].add(f"{cls_dir.name}/{stem}")

    print(f"[{cls_dir.name}]  labels={len(lbls)}  imágenes={len(imgs)}  "
          f"img sin label={len(imgs.keys() - lbls.keys())}  "
          f"label sin img={len(lbls.keys() - imgs.keys())}")
    for c, n in sorted(ids.items()):
        nm = names[c] if c < len(names) else "¡FUERA DE RANGO!"
        print(f"    id {c} ({nm}): {n} cajas")

print("\n--- Resumen ---")
print(f"Archivos con cajas duplicadas: {dup_files}  (líneas repetidas: {dup_lines})")
print(f"Líneas mal formadas o fuera de [0,1]: {bad_lines}")
print(f"Tamaños de imagen más comunes: {sizes.most_common(5)}")

if min_side_px:
    a = np.array(min_side_px)
    p = np.percentile(a, [5, 25, 50, 75])
    print(f"Lado menor de caja [px]: p5={p[0]:.1f}  p25={p[1]:.1f}  "
          f"mediana={p[2]:.1f}  p75={p[3]:.1f}")
    print(f"Cajas con lado menor < {SMALL_PX} px (en la imagen ORIGINAL): "
          f"{(a < SMALL_PX).mean() * 100:.1f} %")
    print("  Ojo: tras re-muestreo a 5 mm/px y tile 640->320, multiplicá por "
          "(1/s)·0.5 para estimar el tamaño que ve el modelo.")

multi = {k: v for k, v in groups.items() if len(v) > 1}
print(f"\nOrígenes con más de un archivo (augmentation): {len(multi)} de {len(groups)}")
for k, v in list(multi.items())[:5]:
    print(f"  {k}: {len(v)} archivos -> {sorted(v)[:3]}...")
print("-> Hacé el split train/val/test por ORIGEN (clave de 'groups'), no por archivo.")
