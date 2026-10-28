"""
yolo_raw.py - decodificado en numpy de un YOLOv8 exportado con la cabeza "cruda" (train-model-v3.ipynb).

El modelo tiene 6 salidas int8, dos por nivel (stride 8, 16 y 32), todas NHWC:
  - caja : [1, h, w, 64]  logits DFL (4 lados x 16 bins), sin decodificar
  - clase: [1, h, w, nc]  score de cada clase, ya con sigmoid (0-1)
El orden de las salidas en el .tflite no es fijo: se identifican por forma.

Así la Edge TPU ejecuta TODAS las convoluciones, y en la CPU queda solo esto:
filtrar por score en int8 (sin decuantizar), decodificar las pocas anclas que pasan y NMS.
Compatible con Python 3.7 (Coral / Mendel). Solo necesita numpy.
"""
import numpy as np

REG_MAX = 16


def preparar_salidas(out_details):
    """Agrupa las salidas por nivel -> [(det_caja, det_clase, h, w)], de mayor a menor resolución."""
    niveles = {}
    for d in out_details:
        _, h, w, c = d["shape"]
        niveles.setdefault((h, w), {})["box" if c == 4 * REG_MAX else "cls"] = d
    return [(v["box"], v["cls"], h, w) for (h, w), v in sorted(niveles.items(), reverse=True)]


def decodificar(interp, niveles, H, W, conf=0.25, iou=0.45, max_det=300, nms_fn=None):
    """Lee las salidas de `interp` (después de invoke) -> array (K, 6): x1, y1, x2, y2, score, clase.
    Coordenadas en píxeles de la entrada del modelo (HxW). nms_fn(cajas, scores, clases, iou) opcional."""
    cajas, scores, clases = [], [], []
    for bd, cd, h, w in niveles:
        sc, zp = cd["quantization"]
        q = interp.get_tensor(cd["index"])[0].reshape(h * w, -1)        # (anclas, nc) int8
        qmax = q.max(1)
        idx = np.nonzero(qmax >= int(np.ceil(conf / sc + zp)))[0]     # umbral comparado en int8
        if idx.size == 0:
            continue
        bs, bz = bd["quantization"]
        b = interp.get_tensor(bd["index"])[0].reshape(h * w, 4, REG_MAX)[idx]
        b = (b.astype(np.float32) - bz) * bs
        b = np.exp(b - b.max(2, keepdims=True))                         # softmax DFL por lado
        stride = float(H) / h
        dist = (b / b.sum(2, keepdims=True)).dot(np.arange(REG_MAX, dtype=np.float32)) * stride
        cx = (idx % w + 0.5) * stride
        cy = (idx // w + 0.5) * stride
        cajas.append(np.stack([cx - dist[:, 0], cy - dist[:, 1], cx + dist[:, 2], cy + dist[:, 3]], 1))
        scores.append((qmax[idx].astype(np.float32) - zp) * sc)
        clases.append(q[idx].argmax(1))
    if not cajas:
        return np.zeros((0, 6), np.float32)
    cajas = np.concatenate(cajas).astype(np.float32)        # la cuantización viene en float64: todo a float32
    scores = np.concatenate(scores).astype(np.float32)
    clases = np.concatenate(clases)
    keep = (nms_fn or nms)(cajas, scores, clases, iou)[:max_det]
    return np.concatenate([cajas[keep], scores[keep, None], clases[keep, None].astype(np.float32)], 1)


def nms(cajas, scores, clases, iou):
    """NMS por clase (desplaza las cajas según la clase para resolverlo en una sola pasada)."""
    off = cajas + clases[:, None].astype(np.float32) * 4096.0
    area = (off[:, 2] - off[:, 0]) * (off[:, 3] - off[:, 1])
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        r = order[1:]
        xx1 = np.maximum(off[i, 0], off[r, 0])
        yy1 = np.maximum(off[i, 1], off[r, 1])
        xx2 = np.minimum(off[i, 2], off[r, 2])
        yy2 = np.minimum(off[i, 3], off[r, 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        order = r[inter / (area[i] + area[r] - inter + 1e-9) < iou]
    return np.array(keep, dtype=np.int64)


def cuantizar_entrada(img_u8, inp):
    """img_u8: HxWx3 uint8 RGB, ya del tamaño de entrada. Con scale=1/255 y zp=-128 el int8 es
    pixel-128, o sea invertir el bit alto: no hace falta pasar a float."""
    sc, zp = inp["quantization"]
    img_u8 = np.ascontiguousarray(img_u8)
    if sc == 0:                                         # entrada sin cuantizar
        return img_u8.astype(inp["dtype"])[None]
    if inp["dtype"] == np.int8 and abs(sc - 1 / 255.0) < 1e-6 and zp == -128:
        return (img_u8 ^ 0x80).view(np.int8)[None]
    if inp["dtype"] == np.uint8 and abs(sc - 1 / 255.0) < 1e-6 and zp == 0:
        return img_u8[None]
    q = np.round(img_u8.astype(np.float32) / 255.0 / sc + zp)
    info = np.iinfo(inp["dtype"])
    return np.clip(q, info.min, info.max).astype(inp["dtype"])[None]
