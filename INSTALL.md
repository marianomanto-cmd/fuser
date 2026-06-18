# 🛠️ Instalación local de Fuser (paso a paso)

Guía para instalar y correr Fuser **en tu PC** (objetivo: **8 GB de VRAM NVIDIA + 40 GB de RAM**).
Pensada para hacerlo a mano **o con ayuda de Claude Code** (ver el [prompt listo para pegar](#-instalar-con-claude-code) al final).

> TL;DR:
> - **Docker (todo incluido):** `docker compose up --build` → http://localhost:7860
> - **Nativo (Linux/macOS):** `bash scripts/setup.sh` → `python app.py` (Windows: `scripts\setup.bat`)
>
> En ambos, los **dos motores** y los modelos quedan listos; en la UI eliges con el **toggle
> "🧠 Motor de Face Swap"** entre InsightFace (rápido) y FaceFusion (alta calidad).

---

## 🐳 Opción rápida: Docker (todo en una caja)

La forma con **menos pasos**: la imagen incluye CUDA, **ambos motores** (InsightFace + FaceFusion) y
los modelos. En la otra PC solo necesitas:

1. **Driver NVIDIA** reciente + **NVIDIA Container Toolkit** (para pasar la GPU al contenedor):
   ```bash
   # Ubuntu (ejemplo): instalar el toolkit y reiniciar Docker
   # https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
   ```
2. Construir y arrancar:
   ```bash
   git clone https://github.com/marianomanto-cmd/fuser.git
   cd fuser
   docker compose up --build        # http://localhost:7860
   ```
   (sin compose:  `docker build -t fuser . && docker run --gpus all -p 7860:7860 -v "$PWD/outputs:/app/outputs" fuser`)

Los vídeos resultantes quedan en `./outputs`. Si tu Docker no acepta la clave `deploy.devices` del
compose, usa el `docker run --gpus all ...` de arriba.

---

## 0) Requisitos previos

| | Recomendado |
|---|---|
| **Python** | 3.10 o 3.11 (evita 3.12/3.13: algunas wheels aún fallan) |
| **GPU** | NVIDIA con **8 GB VRAM** + **drivers recientes** |
| **CUDA** | Lo aporta `onnxruntime-gpu` vía pip; solo necesitas el **driver NVIDIA** actualizado |
| **RAM** | 40 GB (la app la aprovecha para buffers y la opción de 2 pasadas) |
| **Disco** | ~5 GB libres (modelos + dependencias) |
| **Git** | Para clonar el repo |

Comprueba el driver con: `nvidia-smi` (debe listar tu GPU y una versión de CUDA).
**FFmpeg no hace falta instalarlo**: viene incluido vía `imageio-ffmpeg`.

---

## 1) Clonar el repo

```bash
git clone https://github.com/marianomanto-cmd/fuser.git
cd fuser
```

---

## 2) Instalación

### Opción A — Script automático (recomendada)

**Linux / macOS:**
```bash
bash scripts/setup.sh          # GPU (CUDA)
# bash scripts/setup.sh --cpu  # solo CPU (probar la UI, sin GPU)
```

**Windows (CMD o PowerShell):**
```bat
scripts\setup.bat
:: scripts\setup.bat --cpu
```

El script crea `.venv`, instala dependencias, **descarga los modelos recomendados** y corre el diagnóstico.

### Opción B — Manual

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows:
# .venv\Scripts\activate

python -m pip install --upgrade pip
pip install -r requirements.txt          # GPU (CUDA)  ·  o requirements-cpu.txt para CPU
```

---

## 3) ⚠️ CUDA y onnxruntime-gpu (el punto que más falla)

`onnxruntime-gpu` debe coincidir con tu versión de **CUDA**. Reglas prácticas:

- **No instales `onnxruntime` y `onnxruntime-gpu` a la vez** → quítalos y deja solo el GPU:
  ```bash
  pip uninstall -y onnxruntime onnxruntime-gpu
  pip install onnxruntime-gpu
  ```
- Las builds recientes de `onnxruntime-gpu` en PyPI apuntan a **CUDA 12.x**. Si tu sistema usa **CUDA 11.8**, instala la variante para CUDA 11:
  ```bash
  pip install onnxruntime-gpu --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-11/pypi/simple/
  ```
- Necesitas **cuDNN** compatible con tu CUDA. Si ves errores tipo `libcudnn... not found`, instala el cuDNN correspondiente (o usa una imagen/entorno con CUDA+cuDNN ya provistos).
- Referencia oficial: <https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html> (matriz onnxruntime ↔ CUDA ↔ cuDNN).

**Verifica que la GPU está disponible:**
```bash
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
# Debe incluir 'CUDAExecutionProvider'
```

### Nota Windows · insightface
`pip install insightface` a veces compila código nativo. Si falla:
- Instala **Microsoft C++ Build Tools** (Visual Studio Build Tools, workload "Desktop development with C++"), o
- Usa una wheel precompilada de insightface para tu versión de Python.

---

## 4) Diagnóstico de entorno (doctor)

```bash
python scripts/check_env.py
```
Te dice qué falta, si hay GPU/CUDA, RAM, FFmpeg, providers ONNX y qué modelos están descargados.
**Apunta a ver `CUDAExecutionProvider` y `✅ Entorno listo`.**

---

## 5) Modelos

Se **descargan solos en el primer uso**. Para bajarlos por adelantado:
```bash
python scripts/download_models.py            # recomendados (inswapper_128 + gfpgan_1.4)
python scripts/download_models.py --all      # todos (incluye CodeFormer, GPEN, parser, etc.)
```
Si una descarga falla (red/URL), el error te dice **la ruta exacta** donde dejar el `.onnx` a mano (carpeta `models/`).
El detector `buffalo_l` de InsightFace se descarga automáticamente la primera vez.

---

## 6) Ejecutar

```bash
# con el entorno activado:
python app.py
# abre http://127.0.0.1:7860
```

Opciones útiles:
```bash
python app.py --share            # enlace público temporal (probar desde el móvil)
python app.py --listen           # escuchar en la LAN (0.0.0.0)
python app.py --port 7861
```

### Primer uso para videos musicales
1. Modo **🎤 Videos musicales**. 2. Sube **3–5 fotos** (frontal/3-4/perfil, boca cerrada y abierta).
3. **Previsualiza** un frame con boca abierta y uno de perfil. 4. Afina *Preservación de ojos* / *Detalle de boca*.
5. Activa **2 pasadas** si te sobra RAM. 6. **Procesa** y descarga. (Más detalle en el README.)

---

## 📦 Instalar desde un pendrive

En el pendrive **solo hace falta llevar `facefusion/`** (la carpeta pesada). Los `.onnx` de Fuser y el
detector `buffalo_l` **la app los descarga sola** al instalar. ⚠️ **Indicá la ruta real del pendrive**
(cambia en cada PC). Guía completa en **[`PENDRIVE.md`](PENDRIVE.md)**.

| Del pendrive | Va a | ¿Necesario? |
|---|---|---|
| `facefusion/` | `fuser/vendor/facefusion/` | ✅ sí (es la pesada) |
| `fuser_models/*.onnx` | `fuser/models/` | ❌ se descarga sola |
| `insightface/buffalo_l/` | `fuser/models/models/buffalo_l/` | ❌ se descarga sola |

**A mano** (Linux/macOS; `KIT` = ruta real del pendrive):
```bash
bash scripts/setup.sh --no-facefusion              # crea venv, instala deps y BAJA SOLO los modelos + buffalo_l
mkdir -p vendor && cp -r "$KIT"/facefusion vendor/facefusion
python scripts/check_env.py                         # modelos "descargado" + FaceFusion "disponible"
```

### Prompt para Claude Code (instalar desde el pendrive)
> Instala esta app en local. Traigo **FaceFusion** en un pendrive; el resto que se **descargue solo**.
> La carpeta del pendrive está en: `<RUTA_REAL>` (reemplaza por la ruta real; si no la sabes, búscala).
> Pasos: 1) crea `.venv` e instala `requirements.txt`; 2) copia `<RUTA_REAL>/facefusion` a `vendor/facefusion`;
> 3) corre `python scripts/download_models.py` (baja solos los .onnx de Fuser + buffalo_l);
> 4) corre `python scripts/check_env.py` y arregla lo que falte (CUDA incluido); 5) lanza `python app.py`.
> No hagas commits.

---

## ➕ Motor opcional: FaceFusion (Alta Calidad)

El motor **FaceFusion** mejora **boca abierta, dientes y perfiles**, a cambio de **más VRAM y tiempo**.

**Normalmente no tienes que hacer nada:** Docker y `scripts/setup.sh` ya lo instalan, y si lo activas
en el **toggle** de la UI y aún no está, Fuser lo **auto-instala la primera vez** (lo clona en
`vendor/facefusion` e instala sus dependencias). Las siguientes veces es instantáneo.

### Instalarlo a mano (si lo saltaste)
```bash
# con el .venv de Fuser ACTIVO:
python scripts/install_facefusion.py    # cross-platform
# o:  bash scripts/install_facefusion.sh   (Windows: scripts\install_facefusion.bat)
```
Clona FaceFusion en `vendor/facefusion` (Fuser lo **auto-detecta** ahí), instala sus dependencias en
el mismo entorno y **restaura los pines de Fuser** (gradio 5, numpy<2).

### Instalación manual
```bash
git clone https://github.com/facefusion/facefusion vendor/facefusion
pip install -r vendor/facefusion/requirements.txt
# Si FaceFusion trae su propio instalador y lo anterior no basta:
#   cd vendor/facefusion && python install.py --onnxruntime cuda
# Restaura los pines de Fuser por si algo cambió:
pip install -U "gradio>=5,<6" "numpy<2"
```
(Alternativa: instala FaceFusion donde quieras y exporta `PYTHONPATH` a su carpeta.)

### Verificar
```bash
python scripts/check_env.py        # debe mostrar: FaceFusion (Alta Calidad) — disponible
python -c "from fuser.engines.facefusion_engine import is_available; print(is_available())"   # True
```

### Notas importantes
- **VRAM (8 GB):** empieza con *pixel boost* **256x256**; si te quedas sin VRAM, baja a **128x128** o
  usa un modo de memoria más estricto (Fuser lo mapea a `video_memory_strategy` de FaceFusion).
- **Conflictos de dependencias:** FaceFusion instala muchas libs; si cambia la versión de gradio/numpy
  y la UI de Fuser deja de arrancar, reejecuta `pip install -U "gradio>=5,<6" "numpy<2"`.
- **Versión:** el adaptador apunta a **FaceFusion 3.x**. Si tu versión difiere y el motor falla al
  cargar, verás un error claro: usa InsightFace mientras tanto.
- Fuser solo usa los **módulos internos** de FaceFusion (swapper/enhancer/analizador), no su UI.

---

## 7) Problemas frecuentes

| Síntoma | Solución |
|---|---|
| `CUDAExecutionProvider` no aparece | Revisa driver NVIDIA; reinstala `onnxruntime-gpu` acorde a tu CUDA (sección 3) |
| `CUDA out of memory` | UI → *Memoria* → **Bajo VRAM** o **VRAM mínima**; baja *Resolución de procesamiento* |
| `libcudnn...not found` | Instala cuDNN compatible con tu CUDA |
| `insightface` no instala (Windows) | Instala C++ Build Tools o usa wheel precompilada |
| Falla descargar un modelo | Coloca el `.onnx` manualmente en `models/` (la ruta sale en el error) |
| Va lentísimo | Estás en CPU; revisa `check_env.py` que muestre **GPU (CUDA)** |
| "No se detectó cara en la fuente" | Usa fotos nítidas, de frente y bien iluminadas |

---

## 🤖 Instalar con Claude Code

En la PC con la GPU, clona el repo, abre **Claude Code** dentro de la carpeta `fuser/` y pega este prompt:

> Instala esta app en local. Es una app Gradio de face swap de vídeo (lee `CLAUDE.md` e `INSTALL.md`).
> Pasos: 1) crea un venv `.venv` con Python 3.10/3.11; 2) instala `requirements.txt`;
> 3) asegúrate de que `onnxruntime-gpu` coincide con mi CUDA (ejecuta `nvidia-smi` y, si hace falta,
> reinstala con el índice de CUDA 11.8); 4) corre `python scripts/check_env.py` y arrégla lo que falte
> hasta que aparezca `CUDAExecutionProvider`; 5) descarga los modelos con `python scripts/download_models.py`;
> 6) lanza `python app.py` y dime la URL.
> Opcional (máxima calidad): instala el motor FaceFusion con `bash scripts/install_facefusion.sh` y
> verifica con `python scripts/check_env.py` que aparezca "FaceFusion — disponible".
> Si traigo **FaceFusion** en un **pendrive**, te indico la ruta y copias `facefusion` a `vendor/facefusion`
> (el resto se descarga solo con `download_models.py`). Ver `PENDRIVE.md`.
> No subas nada a git ni hagas commits.

Claude Code tiene todo el contexto en `CLAUDE.md` para resolver los detalles (CUDA, modelos, troubleshooting).
