---
title: Fuser Video Face Swap
emoji: 🎭
colorFrom: purple
colorTo: indigo
sdk: gradio
sdk_version: 5.50.0
app_file: app.py
pinned: false
license: mit
---

# 🎭 Fuser — Face swap de vídeo local (optimizado para 8 GB VRAM + 40 GB RAM)

**Fuser** es una aplicación web local cuya **única** función es hacer **face swap en vídeo** con
la mejor calidad posible en hardware modesto. Está diseñada y afinada específicamente para una
configuración de **8 GB de VRAM + 40 GB de RAM**, exprimiendo la RAM del sistema para buffering y
solapamiento de I/O, y controlando con precisión el uso de VRAM.

UI moderna en **Gradio** → fácil de probar **localmente** o en **Hugging Face Spaces** antes de
usarla en tu máquina.

> ⚠️ **Uso responsable.** Esta herramienta es para usos legítimos y con **consentimiento** (efectos,
> doblaje, investigación, arte). No la uses para suplantar identidades, acosar, ni crear
> desinformación o contenido engañoso. Cumple las leyes de tu país.

---

## 📑 Tabla de contenidos
1. [Investigación: ¿qué backend y por qué?](#-investigación-qué-backend-y-por-qué)
2. [🎤 Optimización para videos musicales](#-optimización-para-videos-musicales-v11)
3. [Cómo aprovecha los 8 GB VRAM + 40 GB RAM](#-cómo-aprovecha-los-8-gb-vram--40-gb-ram)
3. [Características](#-características)
4. [Requisitos de hardware](#-requisitos-de-hardware)
5. [Instalación (local con GPU)](#-instalación-local-con-gpu)
6. [Probar en Hugging Face Spaces](#-probar-la-ui-en-hugging-face-spaces)
7. [Uso paso a paso](#-uso-paso-a-paso)
8. [Modos de memoria explicados](#-modos-de-memoria-explicados)
9. [Estructura del repositorio](#-estructura-del-repositorio)
10. [Solución de problemas](#-solución-de-problemas)
11. [Licencias y ética](#-licencias-y-ética)

---

## 🔬 Investigación: ¿qué backend y por qué?

Comparativa del estado del arte (2025–2026) de soluciones **locales** de face swap de vídeo,
con foco en **baja VRAM (8 GB)**:

| Solución | Calidad | VRAM | Facilidad | Multiplataforma | Notas |
|---|---|---|---|---|---|
| **DeepFaceLab** | ⭐⭐⭐⭐⭐ (con entrenamiento) | Alta | ❌ Muy difícil | Limitada | Calidad "cinematográfica" pero requiere **entrenar días por identidad**. Poco mantenido. No es *plug & play*. |
| **FaceFusion 3.x** | ⭐⭐⭐⭐ | Baja-media (4 GB+) | ✅ Buena | ✅ | El estándar abierto más mantenido. One-shot (1 foto), sin entrenar. Usa `inswapper`/`hyperswap` + enhancers. |
| **InsightFace + ReActor / Roop** | ⭐⭐⭐⭐ | Baja | ✅ | ✅ | El mismo motor `inswapper_128` + restaurador. Probado y ubicuo. |
| **VisoMaster / Rope** | ⭐⭐⭐⭐ | Media | ✅ | ⚠️ (Windows/RTX) | Muy rápidos con TensorRT, pero setup pesado y centrado en NVIDIA/Windows. |
| **Modelos GAN entrenados (SimSwap, GHOST, etc.)** | ⭐⭐⭐⭐ | Media | ⚠️ | ⚠️ | Buenos pero menos *plug & play* y con licencias/dependencias variables. |

**Conclusión y decisión de backend:**

> Fuser implementa un **pipeline propio y ligero sobre el motor de InsightFace** —el mismo que hace
> grande a FaceFusion/ReActor— ejecutado **íntegramente como ONNX vía onnxruntime**:
>
> - **Detección + reconocimiento:** `buffalo_l` (SCRFD para detectar + ArcFace para el embedding).
> - **Swap:** `inswapper_128` (one-shot, una sola foto de referencia, sin entrenamiento).
> - **Realce:** `GFPGAN 1.4`, `CodeFormer`, `GPEN`, `RestoreFormer++` (**todos como ONNX**).

**¿Por qué esta elección para 8 GB de VRAM y no envolver FaceFusion o usar DeepFaceLab?**

1. **One-shot, cero entrenamiento.** A diferencia de DeepFaceLab (días de GPU por identidad),
   `inswapper` swapea con **una sola imagen**. Es la única vía realista para una app *plug & play*.
2. **Huella de VRAM diminuta y controlable.** `inswapper_128` (~250 MB), `buffalo_l` (~300 MB) y un
   enhancer 512 (~350 MB) **caben de sobra** en 8 GB **todos a la vez**, dejando margen para buffers.
   El cuello de botella real no es la VRAM sino el *throughput* y el I/O → que es justo lo que
   optimizamos con la RAM.
3. **ONNX puro = ligero y sin "infierno de dependencias".** Evitamos PyTorch/basicsr. La única
   dependencia pesada es `onnxruntime`, lo que hace la instalación mucho más robusta y permite
   **mover el enhancer entre GPU y CPU** cambiando un parámetro (clave para el modo *VRAM mínima*).
4. **Control total de la calidad y la memoria.** Al no envolver FaceFusion como caja negra,
   controlamos el *paste-back* (máscara suavizada, opacidad, realce a 512 px antes de pegar),
   la **gestión explícita de VRAM** (`gpu_mem_limit` por sesión) y el **pipeline de buffers en RAM**.
   La calidad fina (lo que evita el típico look "de plástico") sale del **enhancer + máscara + color
   match**, exactamente las palancas que aquí exponemos al usuario.
5. **El mismo techo de calidad práctico que FaceFusion**, pero con una app a medida, UI propia y
   estrategia de memoria diseñada para *este* hardware.

En resumen: **mejor relación calidad/VRAM/usabilidad** para 8 GB es el stack **InsightFace
(inswapper_128) + enhancer ONNX**, orquestado con una gestión de memoria inteligente.

---

## 🎤 Optimización para videos musicales (v1.1)

Fuser está afinado para el caso exigente de **caras de mujeres cantando en videos
musicales**: múltiples ángulos (frontal/3-4/perfil), **boca muy abierta**,
expresiones intensas y mucho movimiento de cabeza.

### Problemas de la v1.0 (y cómo se corrigen)

| Problema v1.0 | Causa | Solución v1.1 |
|---|---|---|
| **Ojos "muertos"/planos** | El swap a 128 px tiene ojos minúsculos; el enhancer los aplana | **Realce dirigido de ojos** (región derivada de los kps) que devuelve nitidez y vida sin tocar el resto |
| **Dientes borrosos al cantar** | Boca abierta a 128 px = pocos píxeles; el enhancer alucina dientes | **Detalle de boca** localizado + región de boca **alargada** para boca abierta + CodeFormer (mejor en dientes) |
| **Deformación en perfiles** (mandíbula/oreja/nariz) | La máscara **rectangular** pegaba piel sobre oreja/pelo/fondo | **Máscara de contorno** (casco convexo de los 106 landmarks) que **sigue la cara real**; opción de **segmentación BiSeNet** |
| **"Lag"/fantasmas en la boca** | El suavizado temporal EMA arrastra los movimientos rápidos | **Suavizado adaptativo al movimiento**: responde al instante cuando hay movimiento, suaviza solo el temblor |
| **Identidad inestable con la cabeza en movimiento** | Una sola foto de referencia | **Multi-referencia robusta**: varias fotos, ponderadas por frontalidad y con rechazo de outliers |

> Medido en pruebas sintéticas del repo: la máscara de contorno **elimina** el
> sangrado del swap fuera de la cara (R≈98 vs 219 del rectángulo en la esquina del
> recuadro), y el suavizado adaptativo reacciona **~2.4× más rápido** a la boca que
> el EMA clásico, reduciendo a la vez el temblor de los landmarks ~25×.

### Controles específicos
- **🎤 Modo "Videos musicales"**: con un clic activa CodeFormer, máscara de contorno,
  preservación alta de ojos/boca, color match, suavizado adaptativo, **2 pasadas** y
  recomienda **5 referencias**.
- **👁️ Preservación de ojos** y **👄 Detalle de boca/dientes**: sliders dedicados.
- **Tipo de máscara**: contorno (recomendado) · segmentación BiSeNet · elipse · rectángulo.
- **Nº de referencias**: 1 / 3 / 5 / 8 / auto.

### Cómo subir buenas referencias (lo más importante)
Sube **3–5 fotos de la MISMA persona** que cubran lo que aparece en el vídeo:
- **Ángulos**: frontal, 3/4 **y** perfil.
- **Expresiones**: boca cerrada **y** abierta/sonriendo (para que los dientes salgan bien).
- **Calidad**: nítidas, bien iluminadas, sin gafas de sol, sin manos/micrófono tapando la cara,
  sin filtros agresivos.
- Más ángulos = más consistencia con la cabeza en movimiento. **No cuesta VRAM extra**
  (los embeddings se promedian en un único vector de identidad).

### Flujo recomendado para un videoclip
1. Modo **🎤 Videos musicales**. 2. Sube **5 fotos** variadas + el vídeo.
3. **Previsualiza** 6 frames (incluye uno con boca abierta y uno de perfil).
4. Afina **Preservación de ojos** / **Detalle de boca** y, si hay bordes, *Suavizado del borde*.
5. Activa **2 pasadas** si te sobra RAM. 6. **Procesa** y descarga.

---

## 🧠 Cómo aprovecha los 8 GB VRAM + 40 GB RAM

La idea central: **la VRAM es escasa pero suficiente para los modelos; la RAM es abundante y se usa
como buffer elástico para que la GPU nunca espere por el disco.**

**Estrategias implementadas** (ver `fuser/core/memory_manager.py` y `fuser/core/pipeline.py`):

- **Pipeline de 3 etapas solapadas** (productor/consumidor con colas en RAM):
  `decodificar (CPU) → swap+realce (GPU) → codificar (CPU)`. El I/O de disco se solapa con el cómputo
  de GPU. Los **buffers de frames viven en RAM** (decenas/cientos de frames según el modo).
- **Backpressure acotado:** las colas tienen tamaño máximo (`prefetch_frames`, `writer_queue`) para
  usar mucha RAM **pero de forma controlada**, sin desbordarla.
- **Dimensionado dinámico por RAM (`ram_boost`):** los buffers se ajustan a la **RAM libre real**
  (reservando ~30%), de modo que en una máquina con 40 GB la GPU casi nunca espera por el disco.
- **2 pasadas por tramos en RAM (`two_pass_temporal`):** opción de máxima estabilidad. Carga un
  **tramo de frames en RAM** (tamaño calculado según la RAM libre), suaviza los landmarks con una
  **ventana centrada** (no causal → sin lag) y luego renderiza. Aprovecha la RAM para una calidad
  temporal imposible en una sola pasada. **No aumenta la VRAM** (los modelos siguen siendo los mismos).
- **Techo de VRAM por sesión** (`gpu_mem_limit`): cada modelo ONNX tiene un límite de arena, evitando
  que uno acapare la tarjeta. `arena_extend_strategy=kSameAsRequested` y
  `cudnn_conv_algo_search=HEURISTIC` minimizan los picos de VRAM.
- **Offloading GPU↔CPU del enhancer:** en *Bajo/Mínimo VRAM*, el restaurador corre en **CPU usando la
  RAM y todos los núcleos**, liberando VRAM para detección+swap.
- **Procesamiento frame a frame** (nunca se carga el vídeo entero en VRAM) y **resolución de
  procesamiento configurable** para acotar memoria/tiempo.
- **Fallback automático a CPU** por operación si algo no cabe en GPU (provider CPU de respaldo).

---

## ✨ Características

- ✅ **Solo face swap de vídeo** (sin distracciones).
- ✅ **Modo "🎤 Videos musicales"** que activa la mejor configuración para caras cantando.
- ✅ **Multi-referencia robusta**: varias fotos (ángulos/expresiones), ponderadas por frontalidad
  y con **rechazo de outliers**, en un único vector de identidad (no cuesta VRAM).
- ✅ **Una o varias caras**: a todas, a la más grande, por **referencia** (una persona) o por índice.
- ✅ **Selector de enhancer** (GFPGAN, CodeFormer, GPEN, RestoreFormer++) con **control de fuerza**.
- ✅ **Preservación dirigida de ojos y boca/dientes** (sliders dedicados).
- ✅ **Máscara que sigue el contorno** (casco de landmarks) o **segmentación BiSeNet** → perfiles limpios.
- ✅ **Estabilidad temporal**: suavizado **adaptativo al movimiento** (sin lag) y **2 pasadas** (usa RAM).
- ✅ **Calidad**: resolución de procesamiento, **fuerza del swap (opacidad)**, máscara (suavizado y
  recorte) e **igualar color** (iluminación cambiante).
- ✅ **Previsualización de frames clave** antes de procesar todo el vídeo.
- ✅ **Barra de progreso real + FPS + ETA**.
- ✅ **Descarga** del vídeo resultado (con audio original re-multiplexado).
- ✅ **Configuración avanzada de memoria** (modos *Calidad máxima / Equilibrado / Bajo VRAM /
  VRAM mínima*, límite de VRAM, forzar CPU).
- ✅ **Manejo robusto de errores** con mensajes claros (incl. descarga manual de modelos si falla la red).
- ✅ **Plug & play**: FFmpeg incluido vía `imageio-ffmpeg`; modelos auto-descargados al primer uso.

---

## 🖥️ Requisitos de hardware

| | Mínimo (probar UI) | Recomendado (objetivo) | Ideal |
|---|---|---|---|
| GPU | — (CPU, muy lento) | **8 GB VRAM NVIDIA** | 12 GB+ |
| RAM | 8 GB | **40 GB** | 32–64 GB |
| Disco | 3 GB libres | 5 GB+ | SSD |
| SO | Linux / Windows / macOS* | Linux o Windows | Linux |

\* En macOS no hay CUDA; funciona por CPU (lento) o, opcionalmente, con el provider CoreML.

---

## 🚀 Instalación (local con GPU)

> 📘 **Guía completa y solución de problemas en [`INSTALL.md`](INSTALL.md)** (incluye CUDA/onnxruntime,
> notas de Windows y un **prompt listo para instalar con Claude Code**). Contexto para asistentes en
> [`CLAUDE.md`](CLAUDE.md).

**Instalación en un comando** (crea `.venv`, instala, baja modelos y diagnostica):

```bash
git clone https://github.com/marianomanto-cmd/fuser.git
cd fuser
bash scripts/setup.sh        # Linux/macOS   (Windows: scripts\setup.bat)
python app.py                # abre http://127.0.0.1:7860
```

**Instalación manual:**

```bash
# 1) Clonar
git clone https://github.com/marianomanto-cmd/fuser.git
cd fuser

# 2) Entorno virtual (Python 3.10–3.11 recomendado)
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3) Dependencias (GPU NVIDIA / CUDA)
pip install -r requirements.txt

# 4) Diagnóstico de entorno (¿GPU? ¿qué falta?)
python scripts/check_env.py

# 5) (Opcional) Pre-descargar modelos
python scripts/download_models.py

# 6) Lanzar
python app.py
# Abre http://127.0.0.1:7860
```

> **CUDA / onnxruntime-gpu:** asegúrate de tener drivers NVIDIA recientes. `onnxruntime-gpu>=1.17`
> usa CUDA 11.8/12.x según la build. Si ves `CUDAExecutionProvider` ausente en el panel de estado,
> revisa la [matriz de compatibilidad de onnxruntime](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html).
> **No** instales `onnxruntime` y `onnxruntime-gpu` a la vez.

---

## 🤗 Probar la UI en Hugging Face Spaces

Este repo ya incluye la cabecera YAML de Spaces (arriba). Para un Space **gratuito (CPU)**, que sirve
para validar la interfaz y el flujo:

1. Crea un Space → SDK **Gradio**.
2. Sube este repositorio (o conéctalo a GitHub).
3. **Importante (CPU):** sustituye el contenido de `requirements.txt` por el de
   `requirements-cpu.txt` (cambia `onnxruntime-gpu` por `onnxruntime`). En un Space de CPU el
   procesamiento es lento: úsalo solo para **probar la UI**, no para producir vídeos.

> Para velocidad real, ejecuta Fuser en **tu máquina local con la GPU de 8 GB**.

---

## 📋 Uso paso a paso

1. **Sube la cara fuente** (1+ fotos nítidas y de frente). Varias fotos → embedding promediado, más fiel.
2. **Sube el vídeo** objetivo.
3. Elige **enhancer** (GFPGAN para algo natural y rápido; CodeFormer para más nitidez) y su **fuerza**.
4. (Opcional) Ajusta **selección de caras**, **opacidad**, **máscara**, **color** y **suavizado temporal**.
5. Pulsa **👁️ Previsualizar frames clave** para ver el resultado en varios momentos del vídeo.
6. Afina y, cuando estés conforme, **🚀 Procesa el vídeo completo** (verás progreso, FPS y ETA).
7. **Descarga** el resultado.

**Consejos de calidad:**
- Empieza con `enhancer = GFPGAN 1.4`, fuerza `0.8`.
- Si hay bordes visibles, sube *Suavizado del borde* o aplica *Recorte interior* de máscara.
- Si el tono no encaja, activa *Igualar color al original*.
- Si la cara "vibra" entre frames, sube el *Suavizado temporal*.

---

## 🎛️ Modos de memoria explicados

| Modo | VRAM objetivo | `gpu_mem_limit` | Detector | Enhancer | Buffer RAM | Cuándo usarlo |
|---|---|---|---|---|---|---|
| **Calidad máxima** | ~8 GB | 7.0 GB | 640 | GPU | 96 frames | GPU dedicada de 8 GB sin nada más abierto |
| **Equilibrado** ⭐ | ~6 GB | 5.5 GB | 640 | GPU | 64 frames | **Recomendado** para 8 GB de uso diario |
| **Bajo VRAM** | ~4 GB | 3.8 GB | 512 | GPU | 32 frames | 6 GB, o 8 GB con el escritorio cargado |
| **VRAM mínima / + RAM** | ~2.5 GB | 2.4 GB | 320 | **CPU/RAM** | 16 frames | 4 GB, o si te quedas sin VRAM (offload a RAM) |

- **Límite de VRAM por sesión (GB):** `0 = automático` (usa el del modo). Súbelo/bájalo manualmente
  para afinar.
- **Forzar CPU:** todo en CPU (sin GPU). Muy lento; solo para probar.
- **Usar más RAM (`ram_boost`):** dimensiona los buffers según la RAM libre (ideal con 40 GB).
- **2 pasadas (`two_pass_temporal`):** máxima estabilidad temporal usando la RAM; ideal para
  videoclips con cabeza en movimiento. No aumenta la VRAM.
- **Resolución de procesamiento:** *Nativa* = máxima calidad; baja a 1080p/720p para ahorrar
  memoria/tiempo en vídeos grandes.

---

## 📂 Estructura del repositorio

```
fuser/
├── app.py                     # Entrypoint Gradio (local + Hugging Face Spaces)
├── requirements.txt           # Dependencias GPU (CUDA)
├── requirements-cpu.txt       # Dependencias CPU / Spaces
├── README.md
├── INSTALL.md                 # Guía de instalación local (CUDA, Windows, Claude Code)
├── CLAUDE.md                  # Contexto del proyecto para Claude Code
├── .gitignore
├── LICENSE
├── scripts/
│   ├── setup.sh / setup.bat   # Instalación automática (Linux/macOS · Windows)
│   ├── check_env.py           # Doctor de entorno (GPU/RAM/modelos)
│   └── download_models.py     # Pre-descarga de modelos
├── models/                    # Modelos ONNX (auto-descarga; ignorado por git)
└── fuser/
    ├── config.py              # Settings, presets de memoria, registro de modelos
    ├── models/                # Envoltorios ONNX
    │   ├── downloader.py       # Descarga perezosa, verificable y con fallback manual
    │   ├── face_analyser.py    # InsightFace buffalo_l (detección + embeddings + yaw)
    │   ├── face_swapper.py     # InSwapper (inswapper_128)
    │   ├── face_enhancer.py    # GFPGAN / CodeFormer / GPEN / RestoreFormer++ (ONNX)
    │   └── face_parser.py      # Segmentación BiSeNet (máscaras por región, opcional)
    ├── core/                  # Lógica optimizada
    │   ├── memory_manager.py   # VRAM/RAM, providers, offloading, buffers por RAM
    │   ├── face_store.py       # Multi-referencia robusta + selección de objetivos
    │   ├── temporal.py         # Suavizado adaptativo + 2 pasadas (bilateral centrado)
    │   └── pipeline.py         # Orquestación (1/2 pasadas, compositing por regiones, ETA)
    ├── utils/
    │   ├── system.py           # Detección de GPU/VRAM/RAM/FFmpeg
    │   ├── video.py            # Lectura/escritura de vídeo + audio (FFmpeg)
    │   ├── image.py            # Máscaras, paste-back afín, color transfer
    │   └── logging.py
    └── ui/
        └── interface.py        # Interfaz Gradio
```

---

## 🛠️ Solución de problemas

| Síntoma | Causa probable | Solución |
|---|---|---|
| `CUDAExecutionProvider` no aparece | onnxruntime-gpu/CUDA mal emparejados | Reinstala `onnxruntime-gpu` acorde a tu CUDA; revisa drivers NVIDIA |
| `CUDA out of memory` | Modo demasiado agresivo | Baja a *Bajo VRAM* o *VRAM mínima*; reduce *Resolución de procesamiento* |
| Falla la descarga de un modelo | Red/URL caída | El error indica la ruta exacta donde colocar el `.onnx` manualmente |
| "No se detectó ninguna cara en la fuente" | Foto de baja calidad | Usa una foto nítida, de frente y bien iluminada |
| Bordes visibles del swap | Máscara dura | Sube *Suavizado del borde*; prueba *Recorte interior* |
| La cara "tiembla" | Jitter de detección | Activa/sube *Suavizado temporal* |
| Sin audio en la salida | FFmpeg no disponible / sin audio original | Verifica el estado de FFmpeg en el panel; `imageio-ffmpeg` debería bastar |
| Muy lento | Estás en CPU | Revisa que el panel muestre **GPU (CUDA)**; instala `onnxruntime-gpu` |

---

## ⚖️ Licencias y ética

- **Código de Fuser:** licencia MIT (ver `LICENSE`).
- **Modelos de terceros:** `inswapper_128` y los packs de InsightFace se publican para **investigación
  / uso no comercial**. Los enhancers (GFPGAN, CodeFormer, etc.) tienen sus propias licencias. **Revisa
  y respeta cada licencia** según tu caso de uso. Fuser no incluye los pesos: se descargan de sus
  fuentes en el primer uso.
- **Ética:** haz face swap **solo con consentimiento**. No suplantes identidades ni generes
  desinformación. La responsabilidad del uso es de quien ejecuta la herramienta.

---

<div align="center">
Hecho con ❤️ para correr rápido y bonito en 8 GB de VRAM.
</div>
