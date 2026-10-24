# Proyecto: Detección de Daños en Palas de Aerogeneradores

> Archivo de contexto para que una IA (o una persona nueva) entienda el proyecto sin leer todo el repositorio.
> Última actualización: 24/10/2026.

## 🎯 Objetivo
Detectar daños en la superficie de palas de aerogeneradores a partir del video de una cámara montada en un dron, con inferencia en el borde (Edge AI) sobre una Google Coral Dev Board. El sistema marca dónde hay un posible daño; la clasificación final del daño la hace una persona revisando los frames guardados.

Prioridad: **no perder daños** (recall alto). Una falsa alarma cuesta poco porque se descarta en la revisión humana.

## ⚙️ Hardware y stack
- **Placa:** Google Coral Dev Board (Edge TPU). Solo ejecuta modelos **int8** compilados con `edgetpu_compiler`; las operaciones que no mapean a la TPU corren en la CPU y bajan los fps.
- **Cámara:** módulo USB **OV9281** (placa DECXIN-1M-2012V1), **monocromática** (escala de grises) y **global shutter**.
  - Sensor 1280×800 px (1 MP), píxel de 3 µm; 120 fps a 720p según el vendedor.
  - Lente de 3 mm (dato del vendedor, FOV horizontal ≈ 65°), aparentemente rosca M12 intercambiable: **a verificar**.
  - Resolución sobre la pala: GSD = d · p / f ≈ 1 mm/px por metro de distancia con la lente de 3 mm. A 5 m (distancia de vuelo supuesta) ≈ **5 mm/px**. Con lente de 6 mm a 5 m ≈ 2,5 mm/px.
  - Exposición manual ≤ 1 ms para evitar blur por vibración (verificar con `v4l2-ctl -l` al recibirla).
- **Modelo:** YOLOv8n de Ultralytics (versión fija 8.4.171). Candidato a comparar: YOLO26n (sin DFL ni NMS, pensado para int8).
- **Lenguaje / herramientas:** Python, OpenCV, Ultralytics. Entrenamiento en Kaggle (GPU T4).

## 🏷️ Clases
- Orden final (el que usa `train-model-v2.ipynb`): **`0 = damage`, `1 = dirt`**.
- Se entrena con *dirt* como clase aparte para que el modelo aprenda que la suciedad no es daño (menos falsos positivos). En vuelo **solo se alerta por `damage`**. Opción `SOLO_DANIO = True` para entrenar un único detector de daño.
- Ojo: el modelo viejo (`Modelo/`, entrenado el 2/10/2026) usa el orden inverso: `0 = dirt`, `1 = damage`.

## 📊 Datasets
Se entrena con la suma de dos datasets públicos. Ninguno fue tomado con la cámara del proyecto, así que ambos se **convierten para parecerse al OV9281** (gris + ruido + blur de sensor).

### 1. DTU / Nordtank (Kaggle)
- **Nombre:** *YOLO Annotated Wind Turbine Surface Damage* — https://www.kaggle.com/datasets/ajifoster3/yolo-annotated-wind-turbines-586x371 (Foster et al., CC BY-NC 4.0). Basado en *DTU – Drone inspection images of wind turbine* (Shihavuddin & Chen, 2018).
- **Origen:** fotos de dron de **una sola turbina** (Nordtank NTK 500/41, DTU, Roskilde, Dinamarca), años 2017 y 2018. Cámara DJI Zenmuse X5S (f = 15 mm), fotos color de 5280×2970 px.
- **Composición:** cada foto se cortó en una grilla de 9 × 8 recortes (tiles) de **586×371 px**; se quitaron los tiles sin superficie de pala.
  - 13 470 tiles en total; **2 995 con etiquetas** y 10 475 sin etiquetas (pala limpia).
  - *damage*: 8 770 cajas, chicas (alto medio 29 px, 0,73 % del tile; hay cajas de 1 px).
  - *dirt*: 581 cajas, grandes (alto medio 238 px, ~30 % del tile).
  - Nombre de archivo `DJI_0201_02_06` = foto `DJI_0201`, posición fila/columna en la grilla.
  - En las etiquetas originales `0 = dirt`, `1 = damage` (deducido de la cantidad de cajas); el notebook lo re-mapea.
- **GSD estimado:** ≈ 1,1–1,8 mm/px en las fotos de detalle (no hay dato de distancia). Hay también fotos generales de toda la turbina.
- **Procesamiento en `train-model-v2.ipynb`:** se reconstruye cada foto con sus tiles → se re-muestrea a 5 mm/px → gris + ruido → tiles de **640×400** (mitad del frame del OV9281) con 15 % de solape → negativos (pala limpia) hasta el 20 % de train; en val y test se conservan todos (tasa de falsas alarmas más estable).
- **Split:** 70/15/15 **por foto original** (prefijo `DJI_XXXX`), nunca por tile, para evitar fuga de datos.
- **Uso:** su test es el **test de decisión** (el que define si la cámara sirve).

### 2. WTBs2025_2cls (repo, carpeta `Dataset WTBs2025`)
- **Nombre:** *WTBs2025* (Zhang et al., Scientific Data 2026; figshare doi:10.6084/m9.figshare.28876391; uso académico). Versión limpia preparada por el equipo: `WTBs2025_2cls.zip`, en la carpeta de Google Drive https://drive.google.com/drive/u/0/folders/12zPgn9EhGEW9EmUbVqNaAOpePBMj37Eb (no está en el repo por tamaño).
- **Origen:** inspecciones de rutina en varios parques eólicos de China (Shanxi, Gansu, Shaanxi), 2022–2024, cámaras RGB de 4000×3000. Muchas turbinas y fondos: complementa al DTU, que es una sola turbina.
- **Composición:** 7 544 imágenes de **640×480** (aspecto 4:3 corregido), todas con al menos una falla; **sin imágenes de pala limpia y sin dirt**.

  | Split | Imágenes | Cajas damage |
  |---|---:|---:|
  | train | 5 385 | 7 786 |
  | val | 1 086 | 1 581 |
  | test | 1 073 | 1 566 |

  - Las 9 clases originales (erosion, coating detachment, localized damage, pinholes, protective film damage, oil leakage, paint cracks, lightning strikes, surface stains) pasan todas a `0 = damage`. `manifest.csv` guarda la clase original de cada imagen.
  - Split 70/15/15 por grupo de casi-duplicados (pHash + ORB) hecho con `split_wtbs2025.py`; re-etiquetado con `make_2class.py`. Guía completa: `Dataset WTBs2025/WTBs2025_2cls_guia.pdf`.
- **GSD:** desconocido y variable → **no se re-muestrea ni se corta en tiles**; solo gris + ruido. No entra al test de decisión: se reporta aparte como referencia, con recall por clase original.

### Dataset final (lo arma `train-model-v2.ipynb` en Kaggle)
- `train` y `val` = DTU + WTBs2025; `test` = **solo DTU** a 5 mm/px.
- Negativos solo del DTU. Imágenes/tiles con daños demasiado chicos para etiquetar se descartan (no se usan como negativos).

## 📁 Archivos principales
- `train-model-v2.ipynb`: notebook de Kaggle actual (necesita como inputs el dataset DTU de Kaggle y `WTBs2025_2cls` subido como dataset privado; `USAR_WTBS = False` entrena solo con DTU) (merge de datasets, simulación del OV9281, entrenamiento, evaluación por tamaño de daño, export int8 a Edge TPU calibrado con datos propios). Genera `entregables.zip` con `best.pt`, `.tflite` y `metricas.json`.
- `train-model.ipynb`: notebook original (YOLOv8n, color, split aleatorio por tile, 50 epochs). Se conserva como referencia.
- `Prueba modelo.ipynb`: prueba en la PC sobre `Fotos de prueba/` (fotos DTU completas), con tiling a la escala de entrenamiento. `MODELO = "actual"` (modelo viejo, modo nativo color) o `"nuevo"` (modo `ov9281`: gris a 5 mm/px).
- `Modelo/`: modelo viejo (`Modelo nnc/best.zip` = `best.pt` renombrado; export NCNN; `Modelo edgeTPU/` con el `.tflite` viejo). El modelo nuevo va en `Modelo/Modelo nuevo/best.pt`.
- `Dataset WTBs2025/`: README, guía PDF y scripts de preparación del segundo dataset.

## 📈 Estado actual (23/10/2026)
- **Fase:** mejora del modelo y validación de la cámara antes de comprarla, simulando su imagen con los datasets.
- **Modelo viejo (referencia, optimista por fuga de datos en el split):** damage P = 0,73, R = 0,64, mAP50 = 0,71; dirt mAP50 = 0,93. Entrenado en color, a la escala original, sin negativos, 50 epochs.
- **Pendiente:** correr `train-model-v2.ipynb` con DTU + WTBs (nueva línea de base honesta), comparar con solo DTU, barrer GSD 2,5 / 5 / 10 mm/px, medir la pérdida int8 y los fps en la Coral.

## ⚠️ Decisiones y supuestos clave
- **La escala manda:** a 10 mm/px un daño promedio del DTU queda en ~4 px (indetectable). Trabajar a ≥ 5 mm/px (tiles de 640×400 sin achicar, entrada del modelo 640) y, si se puede, a ~2,5 mm/px (lente de 6 mm o volar a 2,5–3 m). La guía de WTBs menciona entrada de 320 px: quedó descartada por esto.
- **La velocidad no es el cuello de botella:** para cubrir la pala alcanza con 1–3 frames por segundo; en vuelo se eligen pocos frames nítidos, se procesan en 4 tiles y se confirma cada daño en ≥ 2 frames.
- **Se evalúa el modelo int8** (el que corre en la Coral), con recall de damage por tamaño y falsos positivos cada 100 imágenes limpias; el umbral de confianza se elige en **val** (parte DTU, máximo recall con P ≥ 0,6) y se aplica fijo en test, para no medir con fuga.
- Supuestos a verificar: distancia de vuelo 5 m, lente de 3 mm intercambiable, GSD original del DTU ≈ 1,5 mm/px.
- Informe con el detalle de las mejoras: "Informe de mejoras: detección de daños en palas" (doc en claude.ai).
