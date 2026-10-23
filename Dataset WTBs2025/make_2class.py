"""
make_2class.py — copia de WTBs2025 (ya limpio con split_wtbs2025.py) con 2 clases:
    0: damage
    1: dirt

Entrada: la carpeta --out de split_wtbs2025.py (tiene data.yaml y groups.csv).
Salida, en la estructura estándar de Ultralytics (portable, sin rutas absolutas):

    WTBs2025_2cls/
        data.yaml
        images/{train,val,test}/*.jpg
        labels/{train,val,test}/*.txt
        manifest.csv   # imagen, split, carpeta y clases ORIGINALES, sintético o no

Por defecto las 9 clases de WTBs2025 pasan a damage (surface stains incluido).
Con --dirt "nombre de clase" ... podés mandar alguna a dirt.

ATENCIÓN para combinar con el DTU de Kaggle: verificar que en ese dataset
damage=0 y dirt=1. Si el orden es otro, re-mapear uno de los dos (--names).

Uso:
    pip install pyyaml
    python make_2class.py "C:/.../DRON/WTBs2025_4x3" --out "C:/.../DRON/WTBs2025_2cls" --zip
    Sin argumentos (botón Run de VS Code), con el script en DATASET o en DRON:
    lee DRON/WTBs2025_4x3 y escribe DRON/WTBs2025_2cls + .zip
"""
import argparse
import csv
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import yaml

SPLITS = ("train", "val", "test")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path, nargs="?", default=None,
                    help="salida de split_wtbs2025.py")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--names", nargs=2, default=("damage", "dirt"),
                    help="nombres de las clases 0 y 1")
    ap.add_argument("--dirt", nargs="*", default=[],
                    help="clases originales que van a dirt (por defecto ninguna)")
    ap.add_argument("--zip", action="store_true", help="además genera <out>.zip")
    args = ap.parse_args()

    if args.src is None:                       # botón "Run" de VS Code, sin argumentos
        here = Path(__file__).resolve().parent
        base = here.parent if (here.parent / "WTBs2025_4x3").is_dir() else here
        args.src = base / "WTBs2025_4x3"
        args.out = args.out or base / "WTBs2025_2cls"
        args.zip = True
        print(f"Sin argumentos: entrada = {args.src}\n                salida  = {args.out} (+ .zip)")
    if args.out is None:
        raise SystemExit("Falta --out")
    src, out = args.src.resolve(), args.out.resolve()
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} ya existe y no está vacía; elegí otra carpeta.")

    cfg = yaml.safe_load((src / "data.yaml").read_text(encoding="utf-8"))
    names = cfg["names"]
    names = [names[k] for k in sorted(names)] if isinstance(names, dict) else list(names)
    unknown = set(args.dirt) - set(names)
    if unknown:
        raise SystemExit(f"Clases desconocidas en --dirt: {unknown}. Disponibles: {names}")
    to_new = {i: (1 if nm in args.dirt else 0) for i, nm in enumerate(names)}
    print("Mapeo de clases:")
    for i, nm in enumerate(names):
        print(f"  {i}: {nm:24s} -> {to_new[i]} ({args.names[to_new[i]]})")

    rows = list(csv.DictReader(open(src / "groups.csv", encoding="utf-8")))

    # Una imagen con el mismo nombre en dos carpetas de clase = la misma foto:
    # se unen sus cajas en un solo label. Si el split no coincide, se avisa.
    by_name = defaultdict(list)
    for r in rows:
        by_name[Path(r["image"]).name].append(r)

    for s in SPLITS:
        (out / "images" / s).mkdir(parents=True, exist_ok=True)
        (out / "labels" / s).mkdir(parents=True, exist_ok=True)

    stats = {s: Counter() for s in SPLITS}
    manifest = []
    conflicts = 0
    for name, rs in sorted(by_name.items()):
        splits = {r["split"] for r in rs}
        if len(splits) > 1:
            conflicts += 1
        split = "train" if "train" in splits and len(splits) > 1 else rs[0]["split"]

        img_src = src / rs[0]["folder"] / "images" / name
        shutil.copy2(img_src, out / "images" / split / name)

        boxes, orig = [], []
        for r in rs:
            lbl = src / r["folder"] / "labels" / (Path(name).stem + ".txt")
            for line in lbl.read_text().splitlines():
                p = line.split()
                if len(p) != 5:
                    continue
                c = int(p[0])
                boxes.append((to_new[c], *p[1:]))
                orig.append(names[c])
        uniq = list(dict.fromkeys(boxes))          # sin duplicados, en orden
        (out / "labels" / split / (Path(name).stem + ".txt")).write_text(
            "".join(f"{c} {x} {y} {w} {h}\n" for c, x, y, w, h in uniq))

        st = stats[split]
        st["images"] += 1
        for c, *_ in uniq:
            st[args.names[c]] += 1
        syn = any(r["synthetic_candidate"] == "1" for r in rs)
        st["sintéticas"] += syn
        manifest.append([f"{split}/{name}", split, ";".join(sorted({r['folder'] for r in rs})),
                         ";".join(sorted(set(orig))), int(syn)])

    with open(out / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image", "split", "orig_folder", "orig_classes", "synthetic_candidate"])
        w.writerows(manifest)

    # Sin 'path': Ultralytics usa la carpeta del propio data.yaml, así funciona
    # igual en tu PC y en Kaggle (/kaggle/input/...).
    (out / "data.yaml").write_text(
        "# WTBs2025 re-etiquetado a 2 clases (todas las fallas -> damage)\n"
        "# Sin 'path': se toma la carpeta de este archivo.\n"
        "train: images/train\nval: images/val\ntest: images/test\n\n"
        f"names:\n  0: {args.names[0]}\n  1: {args.names[1]}\n",
        encoding="utf-8")

    print(f"\n{'split':6s} {'imágenes':>9s} {args.names[0]:>8s} {args.names[1]:>6s} {'sintét.':>8s}")
    for s in SPLITS:
        st = stats[s]
        print(f"{s:6s} {st['images']:9d} {st[args.names[0]]:8d} {st[args.names[1]]:6d} "
              f"{st['sintéticas']:8d}")
    if conflicts:
        print(f"\nAVISO: {conflicts} imágenes estaban en carpetas con distinto split; "
              f"se mandaron a train.")
    bad = sum(stats[s]["sintéticas"] for s in ("val", "test"))
    print("Sintéticas fuera de train:", bad)

    if args.zip:
        z = shutil.make_archive(str(out), "zip", root_dir=out)
        print(f"ZIP para subir a Kaggle: {z}")
    print(f"Listo: {out}")


if __name__ == "__main__":
    main()
