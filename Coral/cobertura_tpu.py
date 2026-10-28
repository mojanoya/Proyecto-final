"""
cobertura_tpu.py — qué parte de un modelo *_edgetpu.tflite corre en la Edge TPU
y qué parte cae en la CPU, SIN necesitar la Coral ni el log del compilador.

Uso:
    pip install flatbuffers
    python cobertura_tpu.py best_full_integer_quant_edgetpu.tflite

Cómo funciona: lee el grafo del .tflite. Todo lo que la Edge TPU ejecuta queda
empaquetado en un único operador 'edgetpu-custom-op'; cualquier otro operador
del grafo corre en la CPU. El compilador mapea en orden y corta en el primer
operador que no soporta: todo lo que viene después va a la CPU.

Objetivo: 0 CONV_2D en CPU. Que el final (decodificado de cajas: RESHAPE,
SOFTMAX, CONCAT, etc.) quede en CPU es aceptable: es poco cómputo.
"""
import io
import json
import struct
import sys
import zipfile
from collections import Counter

import flatbuffers
from flatbuffers import number_types as N
from flatbuffers.table import Table

OPN = {0: "ADD", 1: "AVERAGE_POOL_2D", 2: "CONCATENATION", 3: "CONV_2D",
       4: "DEPTHWISE_CONV_2D", 6: "DEQUANTIZE", 9: "FULLY_CONNECTED", 14: "LOGISTIC",
       17: "MAX_POOL_2D", 18: "MUL", 22: "RESHAPE", 23: "RESIZE_BILINEAR", 25: "SOFTMAX",
       34: "PAD", 39: "TRANSPOSE", 41: "SUB", 45: "STRIDED_SLICE", 49: "SPLIT",
       97: "RESIZE_NEAREST_NEIGHBOR", 114: "QUANTIZE"}
TYPES = {0: "float32", 1: "float16", 2: "int32", 3: "uint8", 4: "int64", 7: "int16", 9: "int8"}


def _root(b):
    return Table(b, flatbuffers.encode.Get(N.UOffsetTFlags.packer_type, b, 0))


def _off(t, i):
    return t.Offset(4 + 2 * i)


def _tab(t, i):
    o = _off(t, i)
    return Table(t.Bytes, t.Indirect(o + t.Pos)) if o else None


def _tabs(t, i):
    o = _off(t, i)
    if not o:
        return []
    n, s = t.VectorLen(o), t.Vector(o)
    return [Table(t.Bytes, t.Indirect(s + 4 * k)) for k in range(n)]


def _nums(t, i, fmt):
    o = _off(t, i)
    if not o:
        return []
    n, s = t.VectorLen(o), t.Vector(o)
    return list(struct.unpack_from("<%d%s" % (n, fmt), t.Bytes, s))


def _str(t, i):
    o = _off(t, i)
    return t.String(o + t.Pos).decode() if o else None


def _scal(t, i, typ):
    o = _off(t, i)
    return t.Get(typ, o + t.Pos) if o else 0


def analyze(path):
    buf = open(path, "rb").read()
    m = _root(buf)
    codes = []
    for oc in _tabs(m, 1):
        b = max(_scal(oc, 0, N.Int8Flags), _scal(oc, 3, N.Int32Flags))
        codes.append(_str(oc, 1) or OPN.get(b, f"builtin_{b}"))
    sg = _tabs(m, 2)[0]
    tensors = _tabs(sg, 0)
    ops = [codes[_scal(op, 0, N.Uint32Flags)] for op in _tabs(sg, 3)]

    def tinfo(k):
        t = tensors[k]
        q = _tab(t, 4)
        return {"shape": _nums(t, 0, "i"), "dtype": TYPES.get(_scal(t, 1, N.Int8Flags), "?"),
                "scale": (_nums(q, 2, "f") or [None])[0] if q else None,
                "zero_point": (_nums(q, 3, "q") or [None])[0] if q else None}

    meta = {}
    if b"PK\x03\x04" in buf:                   # metadatos de Ultralytics
        try:
            z = zipfile.ZipFile(io.BytesIO(buf[buf.index(b"PK\x03\x04"):]))
            meta = json.loads(z.read("metadata.json"))
        except Exception:
            pass
    return ops, [tinfo(k) for k in _nums(sg, 1, "i")], [tinfo(k) for k in _nums(sg, 2, "i")], meta


def main(path):
    ops, ins, outs, meta = analyze(path)
    n_tpu = ops.count("edgetpu-custom-op")
    cpu = [o for o in ops if o != "edgetpu-custom-op"]
    print(f"Archivo: {path}")
    if meta:
        print(f"  Ultralytics {meta.get('version')}  imgsz={meta.get('imgsz')}  "
              f"clases={meta.get('names')}")
        print(f"  datos: {meta.get('description', '').split(' on ')[-1]}")
    for t in ins:
        print(f"  entrada: {t['shape']} {t['dtype']}  scale={t['scale']}  zp={t['zero_point']}")
    for t in outs:
        print(f"  salida : {t['shape']} {t['dtype']}  scale={t['scale']}  zp={t['zero_point']}")

    if n_tpu == 0:
        print("\nNo hay edgetpu-custom-op: este archivo NO está compilado para la Edge TPU.")
        return
    first = ops.index("edgetpu-custom-op")
    print(f"\nBloques en Edge TPU: {n_tpu}   operadores en CPU: {len(cpu)}")
    print(f"  antes del bloque TPU : {dict(Counter(ops[:first]))}")
    print(f"  después del bloque TPU: {dict(Counter(ops[first + 1:]))}")
    if n_tpu > 1:
        print("  (más de un bloque TPU: hay idas y vueltas TPU <-> CPU, también es lento)")
    if cpu:
        corte = next((o for o in ops[first + 1:] if o != "edgetpu-custom-op"), None)
        print(f"  primer operador en CPU después de la TPU: {corte}")

    n_conv = sum(o in ("CONV_2D", "DEPTHWISE_CONV_2D") for o in cpu)
    print()
    if n_conv == 0:
        print("OK: todas las convoluciones corren en la Edge TPU. "
              "Lo que queda en CPU es el postproceso.")
    else:
        print(f"MAL: {n_conv} convoluciones corren en la CPU. "
              f"El modelo va a ser lento en la Coral.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("Uso: python cobertura_tpu.py modelo_edgetpu.tflite")
    for p in sys.argv[1:]:
        main(p)
        print("-" * 70)
