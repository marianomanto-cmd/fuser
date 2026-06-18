# 📦 Instalar Fuser desde un pendrive (kit offline)

Guía para cuando **traes los modelos y FaceFusion en un pendrive** y querés instalar Fuser en casa
**sin volver a descargar nada pesado de internet**.

> ⚠️ **Lo más importante:** la ruta del pendrive **cambia en cada PC** (ej. `D:\modelos`,
> `/media/usb/modelos`, `/Volumes/USB/modelos`). **Tenés que indicársela a Claude Code** (o usarla en
> los comandos). En esta guía esa ruta se llama `KIT`.

---

## 1) Cómo debe verse la carpeta del pendrive

```
modelos/                          ← esta carpeta es "KIT"
├── fuser_models/                 # modelos .onnx de Fuser
│   ├── inswapper_128.onnx
│   ├── gfpgan_1.4.onnx
│   ├── codeformer.onnx
│   ├── gpen_bfr_512.onnx
│   ├── restoreformer_plus_plus.onnx
│   └── face_parser.onnx
├── insightface/
│   └── buffalo_l/                # detector (varios .onnx dentro)
└── facefusion/                   # repo de FaceFusion (clonado)
```

Y cada parte va a un sitio del repo de Fuser:

| Del pendrive | Va a (dentro de `fuser/`) |
|---|---|
| `fuser_models/*.onnx` | `models/` |
| `insightface/buffalo_l/` | `models/models/buffalo_l/` |
| `facefusion/` | `vendor/facefusion/` |

*(¿Aún no armaste el kit? Mira el [Anexo A](#anexo-a--prompt-para-armar-el-kit-pc-con-internet).)*

---

## 2) Instalar en casa — la forma fácil (Claude Code)

Abre **Claude Code** dentro de la carpeta `fuser/` (cloná el repo primero) y pega esto,
**reemplazando la ruta** por la real de tu pendrive:

```text
Instala esta app de face swap en local SIN descargar de internet lo que ya traigo en un pendrive.
La carpeta del kit (KIT) está en:  <PON AQUÍ LA RUTA REAL, p. ej. /media/usb/modelos o D:\modelos>

Haz:
1) Crea un venv .venv (Python 3.10/3.11) e instala requirements.txt.
2) Copia los modelos del kit a su sitio:
   - KIT/fuser_models/*.onnx        -> models/
   - KIT/insightface/buffalo_l/     -> models/models/buffalo_l/
   - KIT/facefusion/                -> vendor/facefusion/
3) Asegúrate de que onnxruntime-gpu coincide con mi CUDA (corre nvidia-smi; si hace falta, reinstala).
4) Corre  python scripts/check_env.py  y arregla lo que falte hasta ver:
   - CUDAExecutionProvider disponible
   - los modelos como "descargado"
   - FaceFusion como "disponible"
5) Lanza  python app.py  y dime la URL.
No subas nada a git ni hagas commits.
```

Claude Code tiene todo el contexto en `CLAUDE.md` e `INSTALL.md` para resolver detalles (CUDA, rutas, etc.).

---

## 3) Instalar en casa — a mano (sin Claude Code)

**Linux / macOS** (reemplazá `KIT` por tu ruta real):
```bash
cd fuser
KIT=/media/usb/modelos        # ⚠️ tu ruta real del pendrive

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp "$KIT"/fuser_models/*.onnx models/
mkdir -p models/models && cp -r "$KIT"/insightface/buffalo_l models/models/
mkdir -p vendor && cp -r "$KIT"/facefusion vendor/facefusion

python scripts/check_env.py    # verifica GPU + modelos + FaceFusion
python app.py                  # http://127.0.0.1:7860
```

**Windows (PowerShell)** (reemplazá `$KIT`):
```powershell
cd fuser
$KIT = "D:\modelos"            # ⚠️ tu ruta real del pendrive

python -m venv .venv ; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

Copy-Item "$KIT\fuser_models\*.onnx" models\
New-Item -ItemType Directory -Force models\models | Out-Null
Copy-Item -Recurse "$KIT\insightface\buffalo_l" models\models\
New-Item -ItemType Directory -Force vendor | Out-Null
Copy-Item -Recurse "$KIT\facefusion" vendor\facefusion

python scripts\check_env.py
python app.py
```

---

## 4) Verificar
```bash
python scripts/check_env.py
```
Debe mostrar: **CUDAExecutionProvider**, los modelos como **descargado** y **FaceFusion — disponible**.
Luego, en la UI, usa el **toggle "🧠 Motor de Face Swap"** y/o el **Modo 🎤 Videos musicales**.

> Nota: lo único que NO va en el pendrive es el **driver NVIDIA** (instálalo en la PC de casa) y los
> *wheels* de pip (los baja `pip install -r requirements.txt`). Si querés también offline total de pip,
> pídeme el kit de wheels indicando tu SO + versión de Python.

---

## Anexo A · Prompt para armar el kit (PC con internet)

Pega esto en **Claude Code** en una PC con buena internet para descargar todo a `Documentos/modelos`
y luego zipearlo al pendrive:

```text
Tarea: armar un "kit offline" de modelos para una app de face swap, para llevarlo en un pendrive.
NO clones ningún repo de la app ni hagas commits: solo DESCARGA archivos a una carpeta.

1) Crea la carpeta `modelos` dentro de mi carpeta de Documentos (Documents/Documentos). KIT = esa carpeta.
   Crea dentro: KIT/fuser_models, KIT/insightface, KIT/facefusion.

2) Descarga en KIT/fuser_models (verifica que cada archivo pese > 1 MB; si pesa poco es error, reintenta):
   - https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx
   - https://huggingface.co/facefusion/models-3.0.0/resolve/main/gfpgan_1.4.onnx
   - https://huggingface.co/facefusion/models-3.0.0/resolve/main/codeformer.onnx
   - https://huggingface.co/facefusion/models-3.0.0/resolve/main/gpen_bfr_512.onnx
   - https://huggingface.co/facefusion/models-3.0.0/resolve/main/restoreformer_plus_plus.onnx
   - https://huggingface.co/facefusion/models-3.0.0/resolve/main/face_parser.onnx

3) En KIT/insightface descarga y descomprime:
   - https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip
   (debe quedar KIT/insightface/buffalo_l/ con varios .onnx)

4) Clona FaceFusion (versión fija) en KIT/facefusion:
   git clone --depth 1 --branch 3.1.1 https://github.com/facefusion/facefusion KIT/facefusion
   (si el tag falla: git clone --depth 1 https://github.com/facefusion/facefusion KIT/facefusion)

5) (OPCIONAL, deja FaceFusion 100% offline pero pesa varios GB) dentro de KIT/facefusion crea un venv,
   instala sus requirements y baja sus modelos:  python facefusion.py force-download
   (si falla o es demasiado, sáltalo: los bajará en casa la primera vez.)

6) Muéstrame el árbol de KIT con el tamaño de cada archivo y el TAMAÑO TOTAL, para zipearlo al pendrive.
```

> El **paso 5 es opcional**. Sin él, FaceFusion descargará sus modelos la primera vez en casa
> (necesitás internet ese rato). Con él, queda todo offline.
