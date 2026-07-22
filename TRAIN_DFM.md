# Crear un modelo `.dfm` para una Cara (identidad de máxima fidelidad)

Un **`.dfm`** es un modelo de **DeepFaceLive/DeepFaceLab** entrenado para **UNA persona**.
A diferencia del swap one-shot (que infiere la cara de tus fotos), el `.dfm` **aprende la
cara completa** — incluida la **geometría de cráneo/nariz** — y por eso es el techo de calidad
para "que se parezca de verdad" y sea estable en movimiento.

Fuser **usa** el `.dfm` en tu GPU (DirectML). Fuser **no lo entrena**: entrenar es CUDA/DeepFaceLab,
en un **entorno separado**. Regla de oro: **entrenás con CUDA una vez, usás con DirectML siempre.**

> ⚠️ **NUNCA** hagas `pip install` de `tensorflow`, `tensorflow-directml`, `onnxruntime-gpu`,
> CUDA, etc. **dentro del `.venv` de Fuser** (`C:\Users\Usuario\Fuser\fuser\.venv`). Rompería
> el `onnxruntime-directml` y la inferencia de Fuser dejaría de andar. El entrenamiento va en
> su **propia** carpeta/venv o en la **nube**.

---

## 0) Validá el pipeline ANTES de invertir horas (recomendado)

Probá que el Deep Swapper corre en tu GPU con un `.dfm` **público** (de un famoso), sin entrenar nada:

1. Bajá un `.dfm` de HuggingFace: `dimanchkek/Deepfacelive-DFM-Models` (elegí uno chico, p.ej. un 224).
2. En Fuser → Biblioteca de Caras → creá una Cara cualquiera (con 1 foto) → **🧬 Asociar .dfm**
   subí ese `.dfm` a esa Cara. (O copialo a mano a
   `fuser\vendor\facefusion\.assets\models\custom\<slug>.dfm`.)
3. Reiniciá Fuser, elegí esa Cara, subí un video, procesá. Si swapea al famoso → **el pipeline anda**.

Si esto funciona, ya sabés que tu `.dfm` propio va a funcionar. Si no, arreglamos eso antes de entrenar.

---

## 1) Elegí dónde entrenar

| | **Nube (recomendado)** | **Local (plan B)** |
|---|---|---|
| GPU | RTX 4090/3090 alquilada (Vast.ai / RunPod) | Tu RTX 4060 Ti 8GB |
| Costo | ~USD 3-15 por modelo | $0 (electricidad) |
| Tiempo | medio día – 2 días | 2-4+ días (más lento) |
| Riesgo | ninguno de hardware | **freeze documentado** de SAEHD en 40-series con el build CUDA → hay que usar el build **DirectX12** (legacy, ~2.8× más lento) |
| VRAM | 24GB → res 320-384, batch grande | 8GB → res 192-224, batch 4-6, optimizer en RAM |

**Por qué nube:** tu 4060 Ti (Ada) tiene un *freeze* reportado entrenando SAEHD con el build CUDA, y
8GB limita fuerte la resolución. Alquilar una 4090/3090 esquiva ambos y sale barato.

> **Colab gratis está prohibido** para entrenar deepfakes (Google lo banea).

---

## 2) Curá el faceset SRC (esto define el parecido — es lo que más importa)

- **~500–2000 imágenes** de la persona, **variadas**: muchos ángulos (yaw/pitch), expresiones,
  luces/sombras, distancias. De un **período corto**, sin mezclar edad/maquillaje/estructura.
- **Stills alcanzan** (no hace falta video). Sacá fotos nítidas, bien iluminadas, sin oclusiones.
- Descartá borrosas, muy oscuras, tapadas y casi-duplicadas.
- (Las 6 fotos de una Cara de la Biblioteca sirven para el one-shot, **NO** para entrenar un `.dfm`.)

El **curado manual pesa MÁS que las horas de GPU**. Dedicale 1-3 h.

---

## 3) Entrenar (workflow RTM con arranque en caliente)

No entrenes de cero: **fine-tuneá el preentrenado**. Baja de días a horas.

1. Instalá **DeepFaceLab** (nube: build Linux o *DeepFaceLab-MVE*; local: build **DeepFaceLab_DirectX12**).
2. Bajá del **FAQ oficial de DeepFaceLive** (`github.com/iperov/DeepFaceLive`):
   - **`RTT model 224 v2`** (modelo preentrenado, ~10M+ iteraciones ya hechas).
   - **`RTM WF Faceset V2`** (~63k caras genéricas = el DST que hace el modelo "universal").
   - Verificá origen/hash; **ignorá redirectores dudosos**.
3. Extraé/alineá tu SRC: `extract faces` **Whole Face (WF)** → ordenar/borrar basura → **XSeg**
   (Generic XSeg alcanza) → opcional `faceset.pak`.
4. Colocá: tu SRC en `workspace/data_src/aligned`; el **RTM WF Faceset V2** en `workspace/data_dst/aligned`;
   cargá el **RTT 224 v2** como modelo base.
5. **Arquitectura — decidí ANTES** (no se cambia después):
   - **SAEHD** (`liae-udt`): identidad más nítida, **sin** slider morph. ← recomendado para tu caso.
   - **AMP**: tiene `morph_value`, pero morphea **expresión**, no identidad.
6. Entrenar SAEHD por etapas (en 8GB: `models/optimizer on GPU = NO` → optimizer a RAM, `AdaBelief`,
   `uniform_yaw:Y`, `blur_out_mask:Y`, batch tan alto como entre — empezá en 4):
   - `+500k` con defaults →
   - `+500k` (borrando `inter_AB.npy` cada 500k para subir el parecido al SRC) →
   - `+700k` final: `random_warp:OFF`, GAN power `0.1`, patch `28`, `gan_dims:32`.

---

## 4) Exportar a `.dfm`

- Corré **`6) export SAEHD as dfm.bat`**, elegí el índice del modelo y marcá **`quantized`**
  (archivo más chico + inferencia más rápida en 8GB).
- Sale un `<nombre>.dfm` (~30–700 MB). Bajalo a tu máquina local.

---

## 5) Usarlo en Fuser

1. **Biblioteca de Caras → 🧬 Asociar .dfm**: elegí la Cara, subí el `.dfm`, importá.
   (Equivale a copiarlo a `fuser\vendor\facefusion\.assets\models\custom\<slug>.dfm`.)
2. **Reiniciá Fuser.**
3. Elegí esa Cara arriba, subí el video y procesá. Como la Cara tiene `.dfm`, Fuser usa el
   **Deep Swapper** automáticamente (ignora las fotos; la identidad viene del modelo).

---

## Caveats honestos

- DeepFaceLab está **archivado** (read-only desde nov-2024): funciona, nadie lo arregla.
- `.dfm` = **1 identidad por modelo**; no es "subí una foto y listo".
- SAEHD **no** tiene morph; AMP morphea expresión, no identidad → elegí antes de entrenar.
- Modelos `custom/` **no** llevan hash → si el `.dfm` se copió truncado, sólo falla al cargar.
  Verificá el tamaño tras importar.
- Si tu objetivo es "esta persona en videos" y no querés el costo de entrenar, primero exprimí
  el one-shot que ya tenés (modo **🎯 Máxima Identidad** / **PRO**). El `.dfm` compensa cuando
  necesitás identidad+geometría de altísima fidelidad y estabilidad temporal.
