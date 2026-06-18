# CLAUDE.md — Guía del proyecto para Claude Code

Contexto para asistentes que trabajen en este repo **en la máquina local**.
Lee también `INSTALL.md` (instalación) y `README.md` (uso y diseño).

## Qué es
**Fuser**: app web local (Gradio) cuya **única** función es **face swap de vídeo** de alta calidad,
optimizada para **8 GB de VRAM NVIDIA + 40 GB de RAM** y afinada para **caras cantando en videos
musicales** (múltiples ángulos, boca abierta, perfiles, mucho movimiento).

## Hardware objetivo
- GPU NVIDIA 8 GB VRAM (CUDA), 40 GB RAM. Sin GPU funciona en CPU (lento, solo para ver la UI).

## Comandos
```bash
# Instalación (crea .venv, instala, baja modelos, diagnostica)
bash scripts/setup.sh            # Linux/macOS  (--cpu para versión CPU)
scripts\setup.bat                # Windows

# Manual
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # GPU  (requirements-cpu.txt = CPU)

python scripts/check_env.py              # DOCTOR: qué falta, GPU, RAM, modelos
python scripts/download_models.py [--all]# pre-descargar modelos
python app.py                            # lanza la UI -> http://127.0.0.1:7860
python app.py --share                    # enlace público temporal
```

## Verificación rápida de GPU
```bash
python -c "import onnxruntime as ort; print(ort.get_available_providers())"  # debe incluir CUDAExecutionProvider
```
Si no aparece CUDA: revisar driver NVIDIA y **emparejar `onnxruntime-gpu` con la versión de CUDA**
(ver `INSTALL.md` §3). **Nunca** tener `onnxruntime` y `onnxruntime-gpu` instalados a la vez.

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
