#!/usr/bin/env python3
"""
bench_fps.py - prueba de velocidad de modelos YOLOv8 *_edgetpu.tflite en la
Coral Dev Board (compatible con Python 3.7 de Mendel).

Uso (en la Coral, con yolo_raw.py en la misma carpeta):
    python3 bench_fps.py modelo.tflite [otro_modelo.tflite ...] [--n 100] [--img foto.jpg]

Mide, para cada modelo:
  - inferencia pura (invoke) en la Edge TPU
  - ciclo completo: preparar imagen + invoke + decodificar cajas + NMS
Con --img además imprime las detecciones, para verificar que el modelo anda.

Acepta los dos formatos de salida:
  - Ultralytics estándar: 1 salida [1, 4+nc, N] (modelos v2 y anteriores)
  - cabeza cruda: 6 salidas (train-model-v3.ipynb), decodificada con yolo_raw.py
"""
import argparse
import io
import json
import time
import zipfile

import numpy as np

import yolo_raw

try:
    from tflite_runtime.interpreter import Interpreter, load_delegate
except ImportError:                                      # si solo está TensorFlow completo
    import tensorflow as tf
    Interpreter, load_delegate = tf.lite.Interpreter, tf.lite.experimental.load_delegate


def load(model_path):
    try:
        dlg = load_delegate("libedgetpu.so.1")
    except (ValueError, OSError) as e:
        raise SystemExit("No se pudo abrir la Edge TPU (libedgetpu.so.1): %s\n"
                         "¿Hay otro programa usando la TPU? Cerralo y probá de nuevo." % e)
    it = Interpreter(model_path=model_path, experimental_delegates=[dlg])
    it.allocate_tensors()
    return it


def class_names(model_path):
    """Lee los nombres de clase que Ultralytics guarda dentro del .tflite."""
    buf = open(model_path, "rb").read()
    k = buf.find(b"PK\x03\x04")
    if k < 0:
        return None
    try:
        meta = json.loads(zipfile.ZipFile(io.BytesIO(buf[k:])).read("metadata.json"))
        names = meta.get("names", {})
        return [names[str(i)] for i in range(len(names))]
    except Exception:
        return None


def decode(out, outd, conf=0.25, iou=0.45):
    """Salida YOLOv8 [1, 4+nc, N] -> lista (x1, y1, x2, y2, score, clase), normalizado 0-1."""
    scale, zp = outd["quantization"]
    y = (out.astype(np.float32) - zp) * scale if scale else out.astype(np.float32)
    y = y[0]                                            # (4+nc, N)
    boxes, scores = y[:4].T, y[4:].T                    # xywh, (N, nc)
    cls = scores.argmax(1)
    sc = scores[np.arange(len(cls)), cls]
    keep = sc > conf
    boxes, sc, cls = boxes[keep], sc[keep], cls[keep]
    if boxes.max(initial=0) > 2:                        # coordenadas en píxeles -> normalizar
        boxes = boxes / max(outd["shape"][-1], 1)
    xyxy = np.concatenate([boxes[:, :2] - boxes[:, 2:] / 2,
                           boxes[:, :2] + boxes[:, 2:] / 2], 1)
    order = sc.argsort()[::-1]
    res = []
    while order.size:                                   # NMS por clase, simple
        i = order[0]
        res.append((*xyxy[i], sc[i], cls[i]))
        xx1 = np.maximum(xyxy[i, 0], xyxy[order[1:], 0])
        yy1 = np.maximum(xyxy[i, 1], xyxy[order[1:], 1])
        xx2 = np.minimum(xyxy[i, 2], xyxy[order[1:], 2])
        yy2 = np.minimum(xyxy[i, 3], xyxy[order[1:], 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        area = lambda b: (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
        ov = inter / (area(xyxy[[i]]) + area(xyxy[order[1:]]) - inter + 1e-9)
        order = order[1:][(ov < iou) | (cls[order[1:]] != cls[i])]
    return res


def read_image(path, w, h):
    try:
        from PIL import Image
        return np.asarray(Image.open(path).convert("RGB").resize((w, h)))
    except ImportError:
        import cv2
        return cv2.cvtColor(cv2.resize(cv2.imread(path), (w, h)), cv2.COLOR_BGR2RGB)


def stats(ms):
    ms = np.array(ms)
    return "media %6.1f ms | p50 %6.1f | p90 %6.1f | %5.1f fps" % (
        ms.mean(), np.percentile(ms, 50), np.percentile(ms, 90), 1000.0 / ms.mean())


def bench(model_path, n, img_path):
    print("=" * 72)
    print("Modelo:", model_path)
    t0 = time.perf_counter()
    it = load(model_path)
    print("  carga + delegate Edge TPU: %.0f ms" % ((time.perf_counter() - t0) * 1e3))
    inp, outs = it.get_input_details()[0], it.get_output_details()
    _, h, w, _ = inp["shape"]
    names = class_names(model_path)
    crudo = len(outs) > 1
    print("  entrada %s %s | salidas %s | clases %s" % (
        inp["shape"].tolist(), inp["dtype"].__name__, [o["shape"].tolist() for o in outs], names))

    if crudo:                                           # cabeza cruda: decode DFL + NMS con yolo_raw
        niveles = yolo_raw.preparar_salidas(outs)

        def run_decode():
            d = yolo_raw.decodificar(it, niveles, h, w, conf=0.25, iou=0.45)
            d[:, [0, 2]] /= w                           # normalizado 0-1, igual que decode()
            d[:, [1, 3]] /= h
            return [tuple(r) for r in d]
    else:
        outd = outs[0]

        def run_decode():
            return decode(it.get_tensor(outd["index"]), outd)

    if img_path:
        img = read_image(img_path, w, h)
    else:
        img = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)

    x = yolo_raw.cuantizar_entrada(img, inp)
    for _ in range(5):                                  # calentamiento (la 1a es lenta)
        it.set_tensor(inp["index"], x)
        it.invoke()

    t_inv, t_full = [], []
    for _ in range(n):
        t0 = time.perf_counter()
        x = yolo_raw.cuantizar_entrada(img, inp)
        it.set_tensor(inp["index"], x)
        t1 = time.perf_counter()
        it.invoke()
        t2 = time.perf_counter()
        dets = run_decode()
        t3 = time.perf_counter()
        t_inv.append((t2 - t1) * 1e3)
        t_full.append((t3 - t0) * 1e3)

    print("  solo inferencia : " + stats(t_inv))
    print("  ciclo completo  : " + stats(t_full))

    if img_path:
        print("  detecciones (conf > 0.25):", len(dets))
        for x1, y1, x2, y2, s, c in sorted(dets, key=lambda d: -d[4])[:10]:
            nm = names[int(c)] if names and int(c) < len(names) else str(int(c))
            print("    %-8s %.2f  caja x1=%.0f y1=%.0f x2=%.0f y2=%.0f px" % (
                nm, s, x1 * w, y1 * h, x2 * w, y2 * h))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+")
    ap.add_argument("--n", type=int, default=100, help="iteraciones medidas")
    ap.add_argument("--img", default=None, help="imagen de prueba (opcional)")
    a = ap.parse_args()
    for m in a.models:
        bench(m, a.n, a.img)
    print("=" * 72)
    print("Nota: 'solo inferencia' es el techo de fps del modelo; la cámara, el")
    print("tiling 2x2 (4 inferencias por frame) y el dibujo restan más.")


if __name__ == "__main__":
    main()
