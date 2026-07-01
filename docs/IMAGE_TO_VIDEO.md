# 🎞️ Imagen → Vídeo (Wan 2.2 14B I2V) — guía completa

Esta función **nueva e independiente** del face swap convierte **una imagen + un
prompt** en un **vídeo corto (~480p, ~6 s, con audio)**. Está pensada para correr
**100% local** en **8 GB de VRAM NVIDIA + 40 GB de RAM** usando **cuantización GGUF**
y **offloading agresivo a la RAM**.

> **TL;DR**
> 1. Instala **ComfyUI** + 3 custom nodes (GGUF, VideoHelperSuite, MultiGPU).
> 2. Descarga los modelos de Wan 2.2 I2V (GGUF Q4_K_M) y de Stable Audio.
> 3. Arranca ComfyUI con `--lowvram`.
> 4. En Fuser, abre la pestaña **🎞️ Imagen → Vídeo**, pulsa **🔌 Comprobar ComfyUI**, sube imagen + prompt y **Generar**.
>
> Diagnóstico en un comando: `python scripts/setup_i2v.py`

---

## 1. Arquitectura: cómo habla la app con ComfyUI

Wan 2.2 14B con offloading pesado es **muy complejo** de ejecutar "a mano". En vez
de reimplementar esa inferencia, Fuser usa **ComfyUI como motor** y habla con él
por su **API HTTP + WebSocket**. Así separamos responsabilidades y no tocamos el
stack ONNX del face swap.

```
┌─────────────────────────────┐         HTTP / WebSocket          ┌──────────────────────────────┐
│  Fuser (Gradio, esta app)   │  ───────────────────────────────► │  ComfyUI (proceso aparte :8188)│
│                             │   POST /upload/image              │                                │
│  pestaña "Imagen → Vídeo"   │   POST /prompt   (workflow JSON)  │  • Wan 2.2 I2V GGUF (vídeo)    │
│        │                    │   WS   /ws       (progreso)       │  • Stable Audio Open (audio)  │
│        ▼                    │   GET  /history/{id}              │  • GGUF + offload a RAM        │
│  I2VService                 │   GET  /view     (descarga)       │                                │
│   ├─ ComfyUIClient (urllib) │ ◄───────────────────────────────  │                                │
│   ├─ workflow.patch_*()     │                                   └──────────────────────────────┘
│   └─ ffmpeg (mezcla A/V)    │
└─────────────────────────────┘
```

**Flujo de una generación** (lo implementa [`fuser/i2v/service.py`](../fuser/i2v/service.py)):

1. **Sube** la imagen a ComfyUI → `POST /upload/image`.
2. **Carga** una plantilla de workflow ([`fuser/i2v/workflows/`](../fuser/i2v/workflows/)) y la
   **parchea** (imagen, prompt, tamaño, duración, semilla, modelos) → `POST /prompt`.
3. **Sigue el progreso** por WebSocket (`/ws`), con *fallback* a *polling* de `/history`.
4. **Descarga** el `.mp4` resultante → `GET /view`.
5. **Genera el audio** (segundo workflow, Stable Audio Open) y lo descarga.
6. **Mezcla** vídeo + audio con `ffmpeg` → `.mp4` final en `outputs/i2v/`.

**Por qué este diseño:**
- El parcheo es **por `class_type`**, no por id de nodo → el mismo código funciona
  con nuestras plantillas **y con cualquier workflow que exportes tú** desde ComfyUI
  ("Save (API Format)"). Si ComfyUI cambia su plantilla oficial, exportas la nueva y
  Fuser la sigue manejando.
- El cliente usa **solo `urllib`** (librería estándar): la app no añade dependencias
  obligatorias. `websocket-client` es **opcional** (solo mejora el progreso).

---

## 2. Instalación paso a paso (ComfyUI + Wan 2.2 para 8 GB)

> Hazlo **una vez** en tu máquina con GPU. ComfyUI vive **fuera** del repo de Fuser
> (es un programa aparte). No metas Wan dentro de Fuser.

### 2.1 Instala ComfyUI

```bash
git clone https://github.com/comfyanonymous/ComfyUI
cd ComfyUI
python -m venv venv && source venv/bin/activate     # Windows: venv\Scripts\activate
# PyTorch con CUDA (ajusta cu121/cu124 a tu CUDA):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

> Lo más fácil en Windows es el **ComfyUI Desktop / portable** (trae todo). Da igual
> cómo lo instales mientras quede escuchando en `http://127.0.0.1:8188`.

### 2.2 Instala los custom nodes (con ComfyUI-Manager)

Instala **[ComfyUI-Manager](https://github.com/Comfy-Org/ComfyUI-Manager)** y, desde
"Install Custom Nodes", añade:

| Custom node | Para qué | Repo |
|---|---|---|
| **ComfyUI-GGUF** | Cargar los modelos cuantizados GGUF (`UnetLoaderGGUF`) | https://github.com/city96/ComfyUI-GGUF |
| **ComfyUI-VideoHelperSuite** | Exportar los frames a `.mp4` (`VHS_VideoCombine`) | https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite |
| **ComfyUI-MultiGPU** *(opcional)* | Offload de capas GGUF a la RAM (DisTorch2) — preset "Offload máximo" | https://github.com/pollockjj/ComfyUI-MultiGPU |

Reinicia ComfyUI tras instalarlos.

### 2.3 Descarga los modelos

Coloca cada fichero en la subcarpeta indicada de `ComfyUI/models/`. Puedes hacerlo
automáticamente:

```bash
# desde el repo de Fuser:
python scripts/setup_i2v.py --download --comfy-path /ruta/a/ComfyUI
# (para Stable Audio, antes: export HF_TOKEN=hf_...  tras aceptar su licencia en HF)
```

O a mano (lista + URLs):

```bash
python scripts/setup_i2v.py --list
```

| Fichero | Carpeta (`ComfyUI/models/…`) | ~Tamaño | Fuente |
|---|---|---|---|
| `Wan2.2-I2V-A14B-HighNoise-Q4_K_M.gguf` | `unet/` | 9.6 GB | [QuantStack/Wan2.2-I2V-A14B-GGUF](https://huggingface.co/QuantStack/Wan2.2-I2V-A14B-GGUF) |
| `Wan2.2-I2V-A14B-LowNoise-Q4_K_M.gguf` | `unet/` | 9.6 GB | (mismo repo, carpeta `LowNoise/`) |
| `umt5_xxl_fp8_e4m3fn_scaled.safetensors` | `text_encoders/` | 5 GB | [Comfy-Org/Wan_2.2_ComfyUI_Repackaged](https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged) |
| `wan_2.1_vae.safetensors` | `vae/` | 0.25 GB | (mismo repo) |
| `stable_audio_open_1.0.safetensors` | `checkpoints/` | 4.9 GB | [stabilityai/stable-audio-open-1.0](https://huggingface.co/stabilityai/stable-audio-open-1.0) **(gated)** |
| `t5_base.safetensors` | `text_encoders/` | 0.9 GB | [google-t5/t5-base](https://huggingface.co/google-t5/t5-base) (`model.safetensors` → renómbralo) |

> **Nota:** el I2V de 14B usa el **VAE de Wan 2.1** (`wan_2.1_vae`) y **NO usa CLIP
> Vision** (eso era de Wan 2.1 I2V). No descargues clip_vision.

### 2.4 Arranca ComfyUI con offloading

```bash
# Recomendado para 8 GB (preset "Equilibrado"):
python main.py --listen 127.0.0.1 --port 8188 --lowvram --reserve-vram 0.6
```

Comprueba que Fuser lo ve:

```bash
python scripts/setup_i2v.py        # ✅ servidor, nodos y modelos
```

---

## 3. Mejor cuantización y offloading para 8 GB + 40 GB

### 3.1 Por qué Wan 2.2 14B "cabe" en 8 GB

Wan 2.2 I2V-A14B es **Mixture-of-Experts**: dos modelos de ~14B (**alto ruido** y
**bajo ruido**) que actúan en distintas fases del *denoising*. **No están los dos en
VRAM a la vez**: el workflow usa el experto de alto ruido en los primeros pasos y el
de bajo ruido en los últimos (dos `KSamplerAdvanced` encadenados). Aun así, **cada
experto Q4_K_M pesa ~9.6 GB**, más que 8 GB → hay que **descargar parte a la RAM**.

### 3.2 Cuantización (GGUF)

| Quant | ~Tamaño/experto | 8 GB | Calidad | Velocidad |
|---|---|---|---|---|
| `Q3_K_M` | ~7.5 GB | ✅ el que menos peta | aceptable | la más rápida |
| **`Q4_K_M`** ⭐ | **~9.6 GB** | ✅ con offload | **el mejor equilibrio** | media |
| `Q5_K_M` | ~11 GB | ⚠️ más offload/lento | mejor | más lento |
| `fp8_e4m3fn` | ~14 GB | ⚠️ casi todo en RAM | alta | lento en 8 GB |

**Recomendación: `Q4_K_M`.** Si te da *out of memory* incluso con offload, baja a
`Q3_K_M` (cambia los nombres en *Avanzado* o en el preset).

### 3.3 Offloading: dos palancas

**(a) Flags de arranque de ComfyUI** — la "smart memory" mueve a RAM lo que no cabe:

| Flag | Qué hace | Cuándo |
|---|---|---|
| `--lowvram` | Reparte el modelo entre VRAM y RAM por capas | **Siempre en 8 GB** |
| `--novram` | Mantiene casi todo en RAM, mínimo en VRAM | Si `--lowvram` aún peta |
| `--reserve-vram 0.6` | Deja ~0.6 GB libres (escritorio/otros) | Recomendado |
| `--use-sage-attention` | Atención más eficiente en VRAM | Si tienes SageAttention instalado |
| `--cache-none` | No cachea modelos entre ejecuciones (ahorra RAM, recarga) | Solo si te falta RAM |

**(b) Cargador con offload explícito (DisTorch2, ComfyUI-MultiGPU)** — para el caso
más justo. El nodo `UnetLoaderGGUFDisTorch2MultiGPU` te deja fijar **cuántos GB
"tomar prestados" de la RAM** por modelo con `virtual_vram_gb` (donante = `cpu`).

Fuser expone esto como **presets** (selector "🧠 Offload / VRAM"):

| Preset | Workflow | Flags | `virtual_vram_gb` | Para |
|---|---|---|---|---|
| **⚖️ Equilibrado** ⭐ | GGUF normal | `--lowvram --reserve-vram 0.6` | — | 8 GB, lo más estable |
| **🧠 Offload máximo** | GGUF + DisTorch2 | `--lowvram --reserve-vram 0.8` | **6.0** | 8 GB muy justo / OOM |
| **⚡ Rendimiento** | GGUF normal | `--reserve-vram 0.5` | — | 10–12 GB |

> **Con 40 GB de RAM vas sobrado**: dos expertos Q4 (~19 GB) + codificador fp8
> (~5 GB) + buffers caben con holgura. La RAM es justo lo que hace viable el 14B en
> una tarjeta de 8 GB.

### 3.4 (Opcional) Acelerar con LoRAs "lightning"

Existen LoRAs de pocos pasos (p. ej. *Wan2.2 Lightning 4-step*) que recortan el
tiempo a costa de algo de calidad. Colócalos en `ComfyUI/models/loras`, añade un
nodo `LoraLoaderModelOnly` por experto en tu workflow, **exporta el API JSON** y
úsalo como plantilla en *Avanzado* (Fuser sigue parcheando por `class_type`). Con un
LoRA de 4 pasos, baja `steps` a 4–8 y `cfg` a 1.0.

---

## 4. El workflow de Wan 2.2 I2V (qué nodos y cómo se conectan)

La plantilla por defecto es
[`wan22_i2v_14b_gguf.json`](../fuser/i2v/workflows/wan22_i2v_14b_gguf.json):

```
LoadImage ─────────────────────────────────┐
CLIPLoader(type=wan) → CLIPTextEncode(+) ───┤
                     └ CLIPTextEncode(−) ───┤
VAELoader ──────────────────────────────────┤→ WanImageToVideo (width,height,length)
                                             │      ├─(positive)─┐
UnetLoaderGGUF(HIGH) → ModelSamplingSD3(8.0) │      ├─(negative)─┤
UnetLoaderGGUF(LOW)  → ModelSamplingSD3(8.0) │      └─(latent)───┤
                                             │                   ▼
   KSamplerAdvanced(ALTO ruido: add_noise=enable, steps 0→10) ───┐
   KSamplerAdvanced(BAJO ruido: add_noise=disable, steps 10→fin) ◄┘
                         ▼
                    VAEDecode → VHS_VideoCombine (mp4, 16 fps)
```

- **No hay CLIP Vision** (Wan 2.2 I2V no lo usa).
- Los **dos expertos** se encadenan: el de **alto ruido** hace la primera mitad de
  pasos (`add_noise=enable`, `return_with_leftover_noise=enable`) y el de **bajo
  ruido** la segunda (`add_noise=disable`). Fuser ajusta el punto de corte a la mitad
  de `steps` automáticamente.
- `VHS_VideoCombine` exporta a **`.mp4`** (`format=video/h264-mp4`); el resultado
  aparece en `/history` y Fuser lo descarga por `/view`.

### Parámetros recomendados (~480p, ~6 s)

| Parámetro | Valor | Nota |
|---|---|---|
| `width × height` | **832 × 480** | 480p 16:9 (lados múltiplos de 16) |
| `length` (frames) | **97** | a 16 fps ≈ 6.06 s (Wan usa frames `4n+1`) |
| `fps` | **16** | fps nativo de Wan 2.2 |
| `steps` | **20** | 10 alto ruido + 10 bajo ruido |
| `cfg` | **3.5** | 3–4 va bien en I2V |
| `sampler` / `scheduler` | **euler** / **simple** | combinación robusta |
| `shift` (ModelSamplingSD3) | **8.0** | 8.0 va bien a 480p |

---

## 5. Audio local + mezcla

Wan **no genera audio**. Fuser lo crea aparte con **Stable Audio Open** (texto →
audio, ~6 s) usando la plantilla
[`stable_audio.json`](../fuser/i2v/workflows/stable_audio.json):

```
CheckpointLoaderSimple(stable_audio_open_1.0)
   ├─ CLIPTextEncode(+ "prompt de audio")
   ├─ CLIPTextEncode(−)
EmptyLatentAudio(seconds≈6) → KSampler(steps 50, cfg 4.5, dpmpp_3m_sde_gpu/exponential)
                                    ▼
                              VAEDecodeAudio → SaveAudio (.flac)
```

- Stable Audio Open ocupa **poca VRAM (~1–2 GB)** y corre **después** de que Wan
  libere la suya, así que cabe sin problema en 8 GB.
- La duración del audio se **iguala a la del vídeo** por defecto.

**Mezcla A/V** (la hace Fuser con `ffmpeg`, en
[`utils/video.mux_external_audio`](../fuser/utils/video.py)):

```bash
ffmpeg -y -i video.mp4 -i audio.flac \
  -map 0:v:0 -map 1:a:0 -c:v copy -c:a aac -b:a 192k -shortest salida.mp4
```

> **Alternativa (mejor sincronía):** *foley* vídeo→audio con **MMAudio**
> ([ComfyUI-MMAudio](https://github.com/kijai/ComfyUI-MMAudio)): genera audio que
> **encaja con el movimiento** del vídeo (pasos, golpes…). Es un custom node aparte;
> si lo prefieres, móntalo en ComfyUI, exporta su API JSON y úsalo como plantilla de
> audio. Para "música de fondo a partir del prompt", **Stable Audio Open** (el de por
> defecto) es lo más simple y estable.

---

## 6. Llamar a la API de ComfyUI desde código (referencia)

La app ya hace todo esto en [`fuser/i2v/comfy_client.py`](../fuser/i2v/comfy_client.py)
y [`fuser/i2v/service.py`](../fuser/i2v/service.py). Ejemplo mínimo y autocontenido:

```python
from fuser.i2v.config import I2VSettings
from fuser.i2v.service import I2VService

settings = I2VSettings(
    comfy_url="http://127.0.0.1:8188",
    offload_preset="balanced_8gb",     # o "max_offload"
    width=832, height=480, length_frames=97,   # ~6 s a 16 fps
    audio_enabled=True,
)
service = I2VService(settings)

# Diagnóstico (¿ComfyUI vivo? ¿nodos? ¿modelos?)
print(service.validate_setup()["markdown"])

# Generar (callback de progreso opcional)
result = service.generate(
    image_path="entrada.png",
    prompt="a woman singing on a neon stage, slow camera push-in, cinematic",
    progress=lambda f, m="": print(f"{int(f*100):3d}% {m}"),
)
print("Vídeo final:", result["video"], "· con audio:", result["has_audio"])
```

Y el cliente "a pelo" (subir imagen + encolar + esperar + descargar):

```python
from fuser.i2v.comfy_client import ComfyUIClient
from fuser.i2v import workflow as wf

cli = ComfyUIClient("http://127.0.0.1:8188")
assert cli.is_available()

image_ref = cli.upload_image("entrada.png")             # POST /upload/image
graph = wf.load_workflow("wan22_i2v_14b_gguf")
graph = wf.patch_i2v(graph, image=image_ref, positive="...", negative="",
                     width=832, height=480, length=97, fps=16, steps=20,
                     cfg=3.5, seed=-1, sampler="euler", scheduler="simple", shift=8.0,
                     high_model="Wan2.2-I2V-A14B-HighNoise-Q4_K_M.gguf",
                     low_model="Wan2.2-I2V-A14B-LowNoise-Q4_K_M.gguf")
prompt_id = cli.queue_prompt(graph)                      # POST /prompt
outputs = cli.wait(prompt_id, progress=lambda f, m="": None)  # WS /ws + /history
video = cli.pick_output(outputs, "video")
cli.download(video, "resultado.mp4")                     # GET /view
```

---

## 7. Limitaciones y rendimiento esperado (8 GB + 40 GB)

- **Velocidad.** Un clip de **~6 s a 480p** tarda **varios minutos** (orientativo:
  **~5–15 min**) según quant, pasos y offload. La **primera** generación es la más
  lenta (carga y cuantiza modelos). Con offload máximo (DisTorch2) es **más estable
  pero más lento** (más tráfico VRAM↔RAM).
- **Memoria.** Q4_K_M + `--lowvram` debería caber en 8 GB. Si ves `CUDA out of
  memory`: usa **Offload máximo a RAM**, baja la **resolución/duración**, o cambia a
  **Q3_K_M**. Cierra otras apps que usen VRAM.
- **Calidad.** Es generación, no edición: la imagen guía la **escena/identidad**, no
  es un calco exacto. Prompts **en inglés**, claros y centrados en el **movimiento**,
  rinden mejor. Movimientos extremos pueden deformar.
- **Audio.** Stable Audio Open hace **música/ambiente** a partir de texto; **no es
  voz/lip-sync**. Para sincronía con el vídeo, mira MMAudio (§5).
- **Duración.** Pensado para **clips cortos** (4–7 s). Vídeos largos multiplican
  tiempo y memoria.

### Fallbacks si el 14B es demasiado en tu equipo

| Opción | Cómo | Tradeoff |
|---|---|---|
| **GGUF más pequeño** | `Q3_K_M` en *Avanzado* | menos VRAM/tiempo, algo menos de calidad |
| **Wan 2.2 TI2V-5B** | Modelo único (usa `wan2.2_vae`), mucho más ligero; monta su workflow en ComfyUI, expórtalo y úsalo como plantilla | más rápido y ligero; calidad algo menor que 14B |
| **Wan 2.1 I2V 1.3B/14B** | Versiones previas; el 1.3B es muy ligero (ojo: Wan 2.1 I2V **sí** usa CLIP Vision) | el 1.3B vuela en 8 GB; menos detalle |

> **Sugerencia:** si el 14B se te hace muy lento, **TI2V-5B** es el mejor compromiso
> para 8 GB. El parcheador de Fuser ya soporta esos grafos: solo exporta el workflow
> en API format y selecciónalo.

---

## 8. Solución de problemas

| Síntoma | Causa probable | Solución |
|---|---|---|
| "ComfyUI no responde" | No está arrancado / otra URL | Arráncalo con `--lowvram`; revisa la URL en *Avanzado* |
| "Faltan custom nodes" | GGUF/VHS/MultiGPU sin instalar | Instálalos con ComfyUI-Manager y reinicia |
| "Faltan modelos" | No descargados / mal ubicados | `python scripts/setup_i2v.py --download --comfy-path …` o colócalos a mano (§2.3) |
| `CUDA out of memory` | Quant/resolución alta | Preset **Offload máximo**, baja resolución/duración, `Q3_K_M`, `--novram` |
| Vídeo negro | Resolución no soportada | Baja la resolución en el nodo `WanImageToVideo` (probada 832×480) |
| Sin audio en la salida | Stable Audio no instalado/gated | Descarga sus modelos (token HF) o desactiva el audio |
| Progreso "a saltos" | Sin `websocket-client` | `pip install -r requirements-i2v.txt` (opcional) |
| El nodo DisTorch2 da error | Versión del custom node distinta | Arrástralo en ComfyUI, ajusta sus campos y **re-exporta** el API JSON |

---

## Fuentes

- ComfyUI Wan 2.2 (workflow oficial): https://docs.comfy.org/tutorials/video/wan/wan2_2 ·
  https://comfyanonymous.github.io/ComfyUI_examples/wan22/
- GGUF de Wan 2.2 I2V: https://huggingface.co/QuantStack/Wan2.2-I2V-A14B-GGUF
- Repackaged (encoder/VAE): https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged
- ComfyUI-GGUF: https://github.com/city96/ComfyUI-GGUF
- ComfyUI-MultiGPU (DisTorch2): https://github.com/pollockjj/ComfyUI-MultiGPU
- VideoHelperSuite: https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite
- Stable Audio Open en ComfyUI: https://comfyanonymous.github.io/ComfyUI_examples/audio/
- Guía low-VRAM Wan 2.2: https://comfyui-wiki.com/en/tutorial/advanced/video/wan2.2/wan2-2
