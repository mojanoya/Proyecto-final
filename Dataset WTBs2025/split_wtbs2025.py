"""
split_wtbs2025.py — split train/val/test de WTBs2025 sin fuga de datos.

Qué hace:
  1. Agrupa casi-duplicados en TODO el dataset con pHash (original, espejo
     horizontal y vertical), para que una foto y sus copias aumentadas caigan
     en el mismo conjunto.
  2. En la carpeta de la clase rara (lightning strikes) agrupa además las
     copias ROTADAS con ORB + RANSAC con transformación de similitud
     (el pHash no es invariante a rotación).
  3. Hipótesis (verificar con las hojas de contacto que genera): en lightning
     strikes, las 57 fotos reales tienen 3 copias cada una (espejo, brillo,
     rotación) -> grupos de 4; las 57 generadas por GA-DCGAN quedan solas.
     Los grupos de tamaño 1 de esa carpeta se tratan como SINTÉTICOS y van
     solo a train.
  4. Split estratificado por carpeta de clase, por grupo (no por archivo).
  5. Con --out: copia el dataset corrigiendo el aspecto 640x640 -> 640x480
     (las fotos originales eran 4000x3000, 4:3) y saca líneas duplicadas de
     los labels. Las coordenadas YOLO normalizadas no cambian con el re-escalado.

Uso:
    pip install pillow opencv-python numpy pyyaml
    python split_wtbs2025.py "C:/.../DRON/DATASET" --out "C:/.../DRON/WTBs2025_4x3"
    (sin --out solo escribe train/val/test.txt y groups.csv en la carpeta original)
    Sin ningún argumento (botón Run de VS Code), con el script DENTRO de DATASET:
    usa esa carpeta y escribe la salida en ../WTBs2025_4x3
"""
import argparse
import csv
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


# ---------------------------------------------------------------- utilidades
class UnionFind:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


_LUT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def popcount64(x):
    if hasattr(np, "bitwise_count"):          # numpy >= 2.0
        return np.bitwise_count(x)
    return _LUT[x.view(np.uint8).reshape(-1, 8)].sum(1)


def _phash(gray):
    """pHash de 64 bits: DCT de la imagen 32x32, bloque 8x8 de bajas
    frecuencias, umbral en la mediana (mismo método que imagehash.phash)."""
    small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    low = cv2.dct(small)[:8, :8]
    bits = (low > np.median(low)).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def phash_variants(path):
    with Image.open(path) as im:
        g = np.asarray(im.convert("L"))
    return [_phash(g), _phash(g[:, ::-1]), _phash(g[::-1, :])]


def has_letterbox(path, thr=8):
    a = np.asarray(Image.open(path).convert("L"), dtype=np.float32)
    k = max(2, a.shape[0] // 40)
    bands = [a[:k].mean(), a[-k:].mean(), a[:, :k].mean(), a[:, -k:].mean()]
    # dos bandas opuestas negras = letterbox
    return (bands[0] < thr and bands[1] < thr) or (bands[2] < thr and bands[3] < thr)


def contact_sheet(rows, out_path, thumb=128):
    rows = [r for r in rows if r]
    if not rows:
        return
    W = max(len(r) for r in rows) * thumb
    sheet = Image.new("RGB", (W, len(rows) * thumb), (40, 40, 40))
    for y, row in enumerate(rows):
        for x, p in enumerate(row):
            with Image.open(p) as im:
                im = im.convert("RGB")
                im.thumbnail((thumb, thumb))
                sheet.paste(im, (x * thumb, y * thumb))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, quality=90)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path, nargs="?", default=None,
                    help="carpeta del dataset (por defecto: la carpeta de este script)")
    ap.add_argument("--out", type=Path, default=None,
                    help="carpeta de salida (aspecto 4:3 + labels sin duplicados)")
    ap.add_argument("--no-aspect", action="store_true",
                    help="con --out, copiar sin corregir el aspecto")
    ap.add_argument("--ratios", type=float, nargs=3, default=(0.70, 0.15, 0.15))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hash-thr", type=int, default=6,
                    help="distancia Hamming máx. de pHash (de 64 bits) para casi-duplicado")
    ap.add_argument("--rare", default="lightning strikes",
                    help="carpeta con augmentation + GAN ('' para desactivar)")
    ap.add_argument("--orb-inliers", type=int, default=30)
    args = ap.parse_args()

    if args.root is None:                      # botón "Run" de VS Code, sin argumentos
        args.root = Path(__file__).resolve().parent
        if args.out is None:
            args.out = args.root.parent / "WTBs2025_4x3"
        print(f"Sin argumentos: dataset = {args.root}\n                salida  = {args.out}")
    root = args.root.resolve()
    cfg = yaml.safe_load((root / "data.yaml").read_text(encoding="utf-8"))
    names = cfg["names"]
    names = [names[k] for k in sorted(names)] if isinstance(names, dict) else list(names)

    # --- inventario
    items = []   # (carpeta, imagen, label)
    for cls_dir in sorted(p for p in root.iterdir() if (p / "images").is_dir()):
        for img in sorted((cls_dir / "images").iterdir()):
            if img.suffix.lower() in IMG_EXT:
                items.append((cls_dir.name, img, cls_dir / "labels" / (img.stem + ".txt")))
    missing = [it for it in items if not it[2].exists()]
    items = [it for it in items if it[2].exists()]
    n = len(items)
    print(f"Imágenes con label: {n}   sin label (excluidas): {len(missing)}")
    for it in missing[:10]:
        print("   sin label:", it[1])

    # --- pHash y agrupado global
    print("Calculando pHash...")
    H = np.array([phash_variants(it[1]) for it in items], dtype=np.uint64)  # (n, 3)
    uf = UnionFind(n)
    for i in range(n):
        for k in range(3):
            d = popcount64(H[:, 0] ^ H[i, k])
            for j in np.nonzero(d <= args.hash_thr)[0]:
                if j != i:
                    uf.union(i, int(j))

    # --- ORB en la clase rara (copias rotadas)
    rare_idx = [i for i, it in enumerate(items) if it[0] == args.rare] if args.rare else []
    if rare_idx:
        print(f"ORB + RANSAC sobre '{args.rare}' ({len(rare_idx)} imágenes)...")
        orb = cv2.ORB_create(500)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        feats = {}
        for i in rare_idx:
            g = cv2.imread(str(items[i][1]), cv2.IMREAD_GRAYSCALE)
            kp, de = orb.detectAndCompute(g, None)
            feats[i] = (kp, de)
        for a_pos, a in enumerate(rare_idx):
            ka, da = feats[a]
            if da is None or len(ka) < 10:
                continue
            for b in rare_idx[a_pos + 1:]:
                if uf.find(a) == uf.find(b):
                    continue
                kb, db = feats[b]
                if db is None or len(kb) < 10:
                    continue
                good = [m[0] for m in bf.knnMatch(da, db, k=2)
                        if len(m) == 2 and m[0].distance < 0.75 * m[1].distance]
                if len(good) < args.orb_inliers:
                    continue
                src = np.float32([ka[m.queryIdx].pt for m in good])
                dst = np.float32([kb[m.trainIdx].pt for m in good])
                # rotación + escala uniforme + traslación: una copia rotada
                # tiene escala ~1; un falso match da escala degenerada
                M, mask = cv2.estimateAffinePartial2D(
                    src, dst, method=cv2.RANSAC, ransacReprojThreshold=5.0)
                if M is None or mask is None:
                    continue
                scale = float(np.hypot(M[0, 0], M[1, 0]))
                if int(mask.sum()) >= args.orb_inliers and 0.9 <= scale <= 1.1:
                    uf.union(a, b)

    groups = defaultdict(list)
    for i in range(n):
        groups[uf.find(i)].append(i)
    sizes = Counter(len(g) for g in groups.values())
    print(f"Grupos: {len(groups)}  (archivos con copias: {n - sizes[1]})  "
          f"grupo más grande: {max(sizes)}")
    if max(sizes) > 20:
        print("  ATENCIÓN: hay un grupo muy grande; probá bajar --hash-thr.")

    # --- sintéticos (hipótesis: grupos de 1 en la clase rara)
    synthetic = set()
    if rare_idx:
        rare_groups = {uf.find(i) for i in rare_idx}
        hist = Counter(len(groups[g]) for g in rare_groups)
        print(f"'{args.rare}': tamaños de grupo {dict(sorted(hist.items()))} "
              f"(esperado aprox. {{1: 57, 4: 57}})")
        synthetic = {g for g in rare_groups if len(groups[g]) == 1}
        review = (args.out or root) / "review"
        sing = [items[groups[g][0]][1] for g in sorted(synthetic)]
        contact_sheet([sing[k:k + 10] for k in range(0, len(sing), 10)],
                      review / "rare_singletons_supuestos_GAN.jpg")
        multi = [[items[i][1] for i in groups[g]] for g in sorted(rare_groups)
                 if len(groups[g]) > 1]
        contact_sheet(multi, review / "rare_grupos_reales.jpg")
        print(f"  Hojas de contacto para revisar a ojo en: {review}")

    # --- split estratificado por carpeta, por grupo
    rng = random.Random(args.seed)
    folder_groups = defaultdict(list)
    for g, members in groups.items():
        folder_groups[items[min(members)][0]].append(g)
    split_of = {}
    print(f"\n{'carpeta':26s} {'grupos':>7s} {'train':>6s} {'val':>5s} {'test':>5s} {'sint.':>6s}")
    for folder in sorted(folder_groups):
        gs = sorted(folder_groups[folder])
        syn = [g for g in gs if g in synthetic]
        real = [g for g in gs if g not in synthetic]
        rng.shuffle(real)
        m = len(real)
        n_val = max(1, round(m * args.ratios[1])) if m >= 3 else 0
        n_test = max(1, round(m * args.ratios[2])) if m >= 3 else 0
        for g in real[:n_val]:
            split_of[g] = "val"
        for g in real[n_val:n_val + n_test]:
            split_of[g] = "test"
        for g in real[n_val + n_test:] + syn:
            split_of[g] = "train"
        print(f"{folder:26s} {len(gs):7d} {m - n_val - n_test:6d} {n_val:5d} {n_test:5d} {len(syn):6d}")

    # --- copia con aspecto 4:3 y labels sin duplicados
    dest = args.out.resolve() if args.out else root
    img_path = {}
    if args.out:
        fix = not args.no_aspect
        if fix:
            sample = random.Random(1).sample(range(n), min(200, n))
            lb = sum(has_letterbox(items[i][1]) for i in sample) / len(sample)
            print(f"\nImágenes con bandas negras (letterbox) en la muestra: {lb * 100:.0f} %")
            if lb > 0.10:
                print("  Hay letterbox: NO se corrige el aspecto (usá otra estrategia).")
                fix = False
        dup_lines = 0
        for i, (folder, img, lbl) in enumerate(items):
            di = dest / folder / "images" / img.name
            dl = dest / folder / "labels" / lbl.name
            di.parent.mkdir(parents=True, exist_ok=True)
            dl.parent.mkdir(parents=True, exist_ok=True)
            with Image.open(img) as im:
                if fix and im.size[0] == im.size[1]:
                    w = im.size[0]
                    im.convert("RGB").resize((w, w * 3 // 4), Image.LANCZOS).save(di, quality=95)
                else:
                    shutil.copy2(img, di)
            seen, rows = set(), []
            for line in lbl.read_text().splitlines():
                key = tuple(line.split())
                if not key:
                    continue
                if key in seen:
                    dup_lines += 1
                    continue
                seen.add(key)
                rows.append(" ".join(key))
            dl.write_text("\n".join(rows) + ("\n" if rows else ""))
            img_path[i] = di
        print(f"Copiado a {dest}  (aspecto 4:3: {'sí' if fix else 'no'}; "
              f"líneas duplicadas eliminadas: {dup_lines})")
    else:
        img_path = {i: it[1] for i, it in enumerate(items)}

    # --- salidas
    lists = defaultdict(list)
    boxes = defaultdict(Counter)
    rows_csv = []
    for g, members in groups.items():
        s = split_of[g]
        for i in members:
            lists[s].append(img_path[i].as_posix())
            lbl = (dest / items[i][0] / "labels" / items[i][2].name) if args.out else items[i][2]
            for key in {tuple(l.split()) for l in lbl.read_text().splitlines() if l.strip()}:
                boxes[s][int(key[0])] += 1
            rows_csv.append([img_path[i].as_posix(), items[i][0], g, len(members), s,
                             int(g in synthetic)])
    for s in ("train", "val", "test"):
        (dest / f"{s}.txt").write_text("\n".join(sorted(lists[s])) + "\n", encoding="utf-8")
    with open(dest / "groups.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image", "folder", "group", "group_size", "split", "synthetic_candidate"])
        w.writerows(rows_csv)

    print(f"\nImágenes: " + "  ".join(f"{s}={len(lists[s])}" for s in ("train", "val", "test")))
    print(f"{'clase':26s} {'train':>6s} {'val':>5s} {'test':>5s}   (cajas únicas)")
    for c, nm in enumerate(names):
        print(f"{nm:26s} {boxes['train'][c]:6d} {boxes['val'][c]:5d} {boxes['test'][c]:5d}")

    yaml_out = {"path": dest.as_posix(), "train": "train.txt", "val": "val.txt",
                "test": "test.txt", "names": {i: nm for i, nm in enumerate(names)}}
    target = dest / ("data.yaml" if args.out else "data_split.yaml")
    target.write_text(yaml.safe_dump(yaml_out, allow_unicode=True, sort_keys=False),
                      encoding="utf-8")
    print(f"\nYAML listo para entrenar: {target}")


if __name__ == "__main__":
    main()
