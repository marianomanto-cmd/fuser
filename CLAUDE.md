# CLAUDE.md — Guía del proyecto para Claude Code

Contexto para asistentes que trabajen en este repo **en la máquina local**.
Lee también `INSTALL.md` (instalación) y `README.md` (uso y diseño).

## Qué es
**Fuser**: app web local (Gradio) cuya **única** función es **face swap de vídeo** de alta calidad,
optimizada para **8 GB de VRAM NVIDIA + 40 GB de RAM** y afinada para **caras cantando en videos
musicales**. Tres pestañas:
- **🎭 Face Swap** (one-shot): Cara guardada/fotos + video + preset (🎯 Máxima Identidad, 🎯➕ PRO,
  🔥 MÁXIMO…). Motor FaceFusion vendorizado.
- **🧬 Deep Swap**: SOLO crea el **modelo `.dfm` por-persona** con un botón (videos de la persona →
  curado → síntesis si falta material → **entrenamiento local** DeepFaceLab DirectX12 en segundo
  plano → autopiloto exporta/registra) + seguimiento/pausa/export/import.
  Código: `fuser/core/dfm_trainer.py`, `faceset.py`, `faceset_synth.py`, `face_library.py`.
- **🎬 Montador** (`fuser/ui/montador.py`, reescrito de cero 2026-07-26): monta un `.dfm` en 1..N
  videos (misma cola), por partes con preview en vivo, reanudable, con Detener-y-guardar. Presets:
  Máxima Identidad (final) / Estándar (test). Plumbing compartido con interface.py en
  `fuser/ui/shared.py` (caché única del pipeline, guarda GPU-ocupada, semáforo de fondo).
  La pestaña CUDA fue ELIMINADA (callejón sin salida en esta GPU; el backend `cuda` sigue en
  `dfm_trainer` por si aparece un build compatible). REGLA: montar con un entrenamiento DFL vivo
  mata el proceso (VRAM llena, DirectML no lanza excepción) — por eso `ensure_gpu_libre` corta
  ANTES en todas las acciones de GPU.

## Hardware objetivo (y VERDADES de esta máquina — leer antes de tocar nada)
- GPU NVIDIA 8 GB VRAM, 40 GB RAM. En ESTA máquina: **DirectML** (onnxruntime-directml 1.24.4;
  1.17.3 diverge aquí). El build CUDA de DeepFaceLab CONGELA en la 4060 Ti → se usa el build
  **DirectX12** (entrena ~3s/iter, res 224).
- **LEY: un modelo ONNX por subproceso** — DirectML retiene la VRAM entre modelos en un mismo
  proceso (colgado silencioso). Vale para benchmarks Y para la síntesis (`faceset_synth._run_pass`
  spawnea `python -m fuser.core.faceset_synth passN`).
- **Principio de memoria**: la RAM asiste a la VRAM en TODO — buffers/2 pasadas en swap,
  optimizer del entrenamiento en RAM (batch alto, `FUSER_DFM_BATCH`), `MODE_MAX_QUALITY` en
  síntesis y montaje.
- Los modelos de FaceFusion viven en `E:\modelos\facefusion` (junction desde
  `vendor/facefusion/.assets/models`); el entrenador en `E:\modelos\deepfacelab`. Al borrar
  recursivo: **desmontar junctions primero** (protege E:).
- `Settings.output_quality` ES el CRF de x264 (menor = mejor). hyperswap está roto en DML aquí.

## Reglas con el usuario (INQUEBRANTABLES)
- **JAMÁS abrir/leer/tocar los inputs u outputs del usuario** (sus fotos/videos/caras guardadas).
- Todas las pruebas con **material de stock** (Pexels/Mixkit) en `agent_tests/`, con
  `FUSER_FACES_DIR`/`FUSER_OUTPUT_DIR` apuntando a dirs de test aislados.
- No parar/relanzar la app sin permiso.

## Comandos
```bash
# Docker (todo incluido: CUDA + ambos motores + modelos)
docker compose up --build        # http://localhost:7860

# Instalación nativa (crea .venv, instala, baja modelos, instala FaceFusion, diagnostica)
bash scripts/setup.sh            # Linux/macOS  (--cpu = CPU; --no-facefusion = sin FaceFusion)
scripts\setup.bat                # Windows

# Motor FaceFusion (alta calidad) — se auto-instala al elegir el toggle; o a mano:
python scripts/install_facefusion.py     # clona en vendor/facefusion + instala deps

# Manual
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # GPU  (requirements-cpu.txt = CPU)

python scripts/check_env.py              # DOCTOR: qué falta, GPU, RAM, modelos
python scripts/download_models.py [--all]# pre-descargar modelos
python scripts/run_demo.py               # PRIMERA PRUEBA: baja stock + prueba features -> carpeta prueba/
python app.py                            # lanza la UI -> http://127.0.0.1:7860
python app.py --share                    # enlace público temporal
```

## Verificación rápida de GPU
```bash
python -c "import onnxruntime as ort; print(ort.get_available_providers())"  # debe incluir CUDAExecutionProvider
```
Si no aparece CUDA: revisar driver NVIDIA y **emparejar `onnxruntime-gpu` con la versión de CUDA**
(ver `INSTALL.md` §3). **Nunca** tener `onnxruntime` y `onnxruntime-gpu` instalados a la vez.

## Kit offline (pendrive)
Normalmente el usuario trae **solo `facefusion/`** en el pendrive (lo pesado): **pídele la ruta** y copia
`facefusion → vendor/facefusion/`. Los `.onnx` de Fuser y `buffalo_l` se bajan solos con
`python scripts/download_models.py` (o en el primer uso). Si el usuario tampoco tendrá internet, también
puede traer `fuser_models/*.onnx → models/` e `insightface/buffalo_l → models/models/buffalo_l/`.
Detalle en `PENDRIVE.md`. La ruta del pendrive la indica el usuario (cambia por máquina).

## Arquitectura (resumen)
```
app.py                      # entrypoint Gradio (también HF Spaces); expone `demo`
fuser/
  config.py                 # Settings, presets de memoria, registro de modelos, motores, expresión
  engines/                  # MOTORES intercambiables (selector en la UI)
    base.py                 #   BaseFaceSwapper (interfaz) + fábrica create_engine
    insightface_engine.py   #   InsightFaceSwapper (pipeline propio: compositing por regiones)
    facefusion_engine.py    #   FaceFusionSwapper (adaptador a módulos internos de FaceFusion)
  models/                   # envoltorios ONNX (import perezoso de onnxruntime/insightface)
    downloader.py           #   descarga perezosa con fallback manual
    face_analyser.py        #   InsightFace buffalo_l (detección + embeddings + yaw)
    face_swapper.py         #   InSwapper inswapper_128 (swap_raw -> cara alineada + matriz afín)
    face_enhancer.py        #   GFPGAN/CodeFormer/GPEN/RestoreFormer++ como ONNX
    face_parser.py          #   BiSeNet face parsing (máscaras por región, OPCIONAL)
  core/
    memory_manager.py       #   providers CUDA/CPU, gpu_mem_limit, buffers según RAM, offloading
    face_store.py           #   multi-referencia robusta + selección de caras objetivo
    temporal.py             #   suavizado adaptativo (1 pasada) + bilateral centrado (2 pasadas)
    pipeline.py             #   orquestación AGNÓSTICA AL MOTOR (habla con BaseFaceSwapper): 1/2 pasadas, ETA, RAM
  utils/                    #   system (GPU/RAM/ffmpeg), video (ffmpeg), image (máscaras/paste), logging
  ui/interface.py           #   UI Gradio (modo Videos musicales, controles de ojos/boca/máscara)
scripts/                    # setup.sh/.bat, check_env.py, download_models.py
models/                     # .onnx descargados (ignorado por git salvo .gitkeep)
```

## Convenciones / decisiones clave
- **Dos motores tras `BaseFaceSwapper`** (`fuser/engines/`): el `pipeline` llama a la interfaz, nunca
  a una implementación. InsightFace es el motor por defecto; **FaceFusion es opcional** (si no está
  instalado, `FaceFusionSwapper.load()` lanza `FaceFusionNotAvailable` con instrucciones). No metas
  dependencia dura de `facefusion` en `requirements.txt`.
- **Todo ONNX vía onnxruntime** (sin PyTorch/basicsr): instalación ligera y robusta.
- **Imports perezosos**: `onnxruntime`/`insightface` se importan dentro de funciones/métodos para que
  la UI arranque sin ellos (clave para probar la UI). No los subas a nivel de módulo.
- **Modelos no versionados**: se descargan en `models/` en el primer uso. `.gitignore` usa `/models/*`
  (ancla a la raíz) para NO ignorar el paquete `fuser/models/`.
- **Gradio 5**. La build de la UI debe pasar `demo.get_api_info()` sin error.
- **Calidad**: el swap base es 128 px; la calidad fina sale del **compositing por regiones** (realce
  dirigido de ojos/boca + máscara de contorno) y del **enhancer**. La matriz afín del swapper se
  escala ×4 para pegar la cara realzada a 512 px (`utils/image.scale_affine`).
- **Memoria**: la RAM ayuda a la VRAM (buffers `ram_boost`, modo 2 pasadas por tramos). La VRAM por
  sesión se acota con `gpu_mem_limit`. El enhancer puede ir a CPU en modos de baja VRAM.

## Pruebas
No hay suite formal aún. Para validar cambios sin GPU:
```bash
python -m compileall -q fuser app.py scripts        # sintaxis
python -c "import app"                               # construye la UI (Gradio)
python scripts/check_env.py                          # entorno
```
La lógica pura (máscaras, suavizado temporal, multi-ref, gestión de memoria) es testeable sin GPU/modelos.

## Qué NO hacer
- No conviertas esto en algo más que face swap de vídeo.
- No añadas dependencias pesadas innecesarias (mantén el stack ONNX).
- No hagas commits/push salvo que el usuario lo pida explícitamente.
- No subas pesos de modelos al repo.
