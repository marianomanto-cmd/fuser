# 📦 Instalar Fuser con FaceFusion desde un pendrive

En el pendrive **solo hace falta llevar la carpeta `facefusion/`** (la pesada). Los modelos de Fuser
(`fuser_models`) y el detector de InsightFace (`insightface/buffalo_l`) **la app los descarga sola** al
instalar/usar, así que **no necesitás llevarlos** en el pendrive.

> ⚠️ **Lo único que tenés que recordar:** indicarle a Claude Code (o usar en los comandos) la **ruta
> real del pendrive**, que cambia en cada PC (ej. `D:\`, `/media/usb`, `/Volumes/USB`). Aquí la
> llamamos `KIT` y apunta a la carpeta que contiene `facefusion/`.

---

## 1) Qué llevás en el pendrive vs. qué se baja solo

| Carpeta | ¿Va en el pendrive? | Dónde va / cómo se obtiene |
|---|---|---|
| **`facefusion/`** | ✅ **Sí** (es la pesada) | copiar a `fuser/vendor/facefusion/` |
| `fuser_models/` (`.onnx`) | ❌ No hace falta | **se descarga sola** → `fuser/models/` |
| `insightface/buffalo_l/` | ❌ No hace falta | **se descarga sola** → `fuser/models/models/buffalo_l/` |

> Si igual los llevaste, podés copiarlos a su sitio y ahorrás esa descarga (ver más abajo). Pero **no
> es necesario**: con internet en casa, la app los baja en la instalación/primer uso.

---

## 2) Instalar en casa — la forma fácil (Claude Code)

Cloná el repo, abrí **Claude Code** dentro de `fuser/` y pegá esto **reemplazando la ruta real**:

```text
Instala esta app de face swap en local. Traigo FaceFusion en un pendrive; el resto que se descargue solo.
La carpeta del pendrive (KIT) está en:  <PON AQUÍ LA RUTA REAL, p. ej. /media/usb  o  D:\>

Haz:
1) Crea un venv .venv (Python 3.10/3.11) e instala requirements.txt.
2) Copia FaceFusion del pendrive:   KIT/facefusion  ->  vendor/facefusion   (créala si no existe).
3) Asegúrate de que onnxruntime-gpu coincide con mi CUDA (corre nvidia-smi; si hace falta, reinstala).
4) Corre  python scripts/download_models.py   (descarga solos los .onnx de Fuser + el detector buffalo_l).
5) Corre  python scripts/check_env.py  y arregla lo que falte hasta ver:
   - CUDAExecutionProvider disponible
   - modelos como "descargado"
   - FaceFusion como "disponible"
6) Primera prueba:  python scripts/run_demo.py  (baja una foto fuente de stock y prueba varias
   configuraciones). Si te pide el clip objetivo, baja uno corto de una mujer cantando que se pase la
   mano por la cara, guárdalo como prueba/target.mp4 y reejecuta. Deja los resultados en prueba/.
7) Lanza  python app.py  y dime la URL.
No subas nada a git ni hagas commits.
```

Claude Code tiene el contexto en `CLAUDE.md` e `INSTALL.md` para resolver detalles (CUDA, rutas, etc.).

---

## 3) Instalar en casa — a mano (sin Claude Code)

**Linux / macOS** (reemplazá `KIT`):
```bash
cd fuser
KIT=/media/usb                 # ⚠️ ruta real del pendrive (la que contiene facefusion/)

bash scripts/setup.sh --no-facefusion   # crea venv, instala deps y BAJA SOLO los modelos + buffalo_l
mkdir -p vendor && cp -r "$KIT"/facefusion vendor/facefusion   # FaceFusion del pendrive

python scripts/check_env.py    # verifica GPU + modelos + FaceFusion
python scripts/run_demo.py     # PRIMERA PRUEBA: baja stock y prueba features -> carpeta prueba/
python app.py                  # http://127.0.0.1:7860
```

**Windows (PowerShell)** (reemplazá `$KIT`):
```powershell
cd fuser
$KIT = "D:\"                    # ⚠️ ruta real del pendrive

scripts\setup.bat --no-facefusion
New-Item -ItemType Directory -Force vendor | Out-Null
Copy-Item -Recurse "$KIT\facefusion" vendor\facefusion

python scripts\check_env.py
python scripts\run_demo.py
python app.py
```

> Usamos `--no-facefusion` porque FaceFusion ya lo traés en el pendrive (no hace falta clonarlo).
> `setup` igual **descarga solo** los `.onnx` de Fuser y el detector buffalo_l.

---

## 4) Verificar
```bash
python scripts/check_env.py
```
Debe mostrar: **CUDAExecutionProvider**, los modelos como **descargado** y **FaceFusion — disponible**.
Luego, en la UI, usá el **toggle "🧠 Motor de Face Swap"** y/o el **Modo 🎤 Videos musicales**.

> Lo único que NO va en el pendrive ni lo baja la app es el **driver NVIDIA** (instálalo en la PC de
> casa). Los *wheels* de pip los instala `pip install -r requirements.txt`.

---

## 5) (Opcional) Si NO vas a tener internet en casa

Si querés **todo 100% offline**, llevá también `fuser_models/` e `insightface/buffalo_l/` en el pendrive
y copialos a mano (no hará falta descargar nada):
```bash
cp "$KIT"/fuser_models/*.onnx models/
mkdir -p models/models && cp -r "$KIT"/insightface/buffalo_l models/models/
```

---

## Anexo A · Prompt para armar el kit (PC con internet)

Para volver a generar el contenido del pendrive. **Para el caso normal alcanza con el paso 4
(FaceFusion)**; los pasos 2–3 solo si querés el modo 100% offline.

```text
Tarea: armar un kit en una carpeta para llevar en un pendrive. NO clones el repo de la app ni hagas commits.

1) Crea la carpeta KIT (p. ej. Documentos/modelos).

2) (OPCIONAL, solo si NO tendré internet en casa) En KIT/fuser_models descarga (verifica > 1 MB c/u):
   - https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx
   - https://huggingface.co/facefusion/models-3.0.0/resolve/main/gfpgan_1.4.onnx
   - https://huggingface.co/facefusion/models-3.0.0/resolve/main/codeformer.onnx
   - https://huggingface.co/facefusion/models-3.0.0/resolve/main/gpen_bfr_512.onnx
   - https://huggingface.co/facefusion/models-3.0.0/resolve/main/restoreformer_plus_plus.onnx
   - https://huggingface.co/facefusion/models-3.0.0/resolve/main/face_parser.onnx

3) (OPCIONAL) En KIT/insightface descarga y descomprime:
   - https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip
   (debe quedar KIT/insightface/buffalo_l/ con varios .onnx)

4) (NECESARIO) Clona FaceFusion en KIT/facefusion:
   git clone --depth 1 --branch 3.1.1 https://github.com/facefusion/facefusion KIT/facefusion
   (si el tag falla: git clone --depth 1 https://github.com/facefusion/facefusion KIT/facefusion)
   Para dejar FaceFusion 100% offline, dentro de KIT/facefusion: instala sus requirements y corre
   "python facefusion.py force-download" (pesa varios GB; si no, sus modelos se bajan en casa).

5) Muéstrame el árbol de KIT con el tamaño de cada parte y el TOTAL, para zipearlo al pendrive.
```

> En el pendrive **lo único imprescindible es `facefusion/`**. El resto se baja solo en casa.
