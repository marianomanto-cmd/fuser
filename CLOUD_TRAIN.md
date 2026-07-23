# Guía: entrenar tu modelo `.dfm` en la nube y usarlo en Fuser

**Para quién es:** vos, que querés un modelo `.dfm` de **una** persona para el **Deep Swapper** de Fuser (FaceFusion 3.1.1), y NO sos experto en Linux/GPU.

**Por qué la nube:** tu RTX 4060 Ti (Ada, 8 GB) se congela con los builds CUDA de DeepFaceLab y 8 GB es poco para entrenar. Entonces **entrenás en una GPU alquilada** (Linux, CUDA) y el `.dfm` terminado lo usás **localmente por inferencia en DirectML** sobre tu 4060 Ti (eso sí funciona, no necesita CUDA).

**Plan técnico (ya decidido, no lo cambies):** fine-tune del preentrenado **RTT 224 v2** usando **RTM WF Faceset V2** como DST + tu faceset SRC → **SAEHD (liae-udt)** → **export SAEHD as dfm (quantized)** → el `.dfm` corre en Fuser.

> **Aviso de honestidad.** Marco con **[NO VERIFICADO]** cada cosa que no pude confirmar al 100 % (URLs de descarga, existencia exacta de un script, precios que cambian mes a mes). No es relleno: es dónde tenés que mirar con ojo. Y una realidad: el procedimiento **RTM completo** son ~1.2 millones de iteraciones (decenas de horas de GPU). Podés parar antes y obtener un parecido "bueno", pero el `.dfm` que **generaliza a cualquier video** sale del recorrido completo.

---

## 0) Qué vas a necesitar (antes de gastar un peso)

- [ ] **Fotos de la persona a clonar (SRC):** 500–2000 imágenes idealmente, con **variedad de ángulos** (frontal, 3/4, perfiles), **expresiones** (boca abierta/cerrada, ojos), y buena luz/nitidez. Más variedad = mejor identidad. Fotos pobres = `.dfm` pobre, y eso **no lo arregla** Fuser.
- [ ] **Consentimiento y uso legítimo.** Face-swap solo con permiso de la persona. Los TOS de RunPod/Vast prohíben contenido no consentido y adulto. **[NO VERIFICADO]** que mencionen explícitamente "face-swap SFW" como permitido; ante duda, es tu responsabilidad.
- [ ] **Cuenta + saldo en la nube.** Con **US$15–25** te sobra para una primera prueba. Un fine-tune real puede ser de pocas horas (US$2–8) hasta 1–2 días (US$10–40) según cuánto entrenes.
- [ ] **Una clave SSH** (te la creo en el paso 2).
- [ ] **Fuser ya instalado y funcionando** en tu PC (este repo).
- [ ] **Tiempo.** No es un botón mágico: contá una tarde para montar todo + horas/días de entreno desatendido.

---

## 1) Preparar el faceset localmente (curar tus fotos)

**Fuser trae un curador de facesets:** `scripts/prep_faceset.py`. Le pasás una carpeta con las fotos candidatas de la persona y **deja solo las útiles**, listas para que DeepFaceLab las procese.

Qué hace (verificado en el código local):
- Descarta ilegibles, sin cara, cara muy chica, borrosas o muy oscuras/quemadas.
- **Deduplica** casi-idénticas (embedding ArcFace).
- Avisa si **se coló otra persona** (consistencia de identidad).
- Reporta la **cobertura de ángulos** (frontal/perfiles) para que sepas qué te falta fotografiar.
- **NO recorta caras** (eso lo hace DeepFaceLab): **copia las fotos BUENAS completas** a una carpeta de salida, renumeradas.

En tu PC (PowerShell, desde la raíz del repo `C:\Users\Usuario\Fuser\fuser`):

```powershell
# activa el venv de Fuser si aplica, luego:
python scripts/prep_faceset.py --input "C:\Users\Usuario\Fuser\fotos_persona" --output "C:\Users\Usuario\Fuser\faceset_listo"
# opcionales de rigor: --min-face 128  --min-sharpness 60  --dedup 0.96
```

Leé el reporte final: si dice que faltan perfiles o expresiones, **conseguí más fotos de esos ángulos** antes de gastar GPU. Cuando la carpeta `faceset_listo` te convenza, comprimila para subirla:

```powershell
Compress-Archive -Path "C:\Users\Usuario\Fuser\faceset_listo\*" -DestinationPath "C:\Users\Usuario\Fuser\faceset_src.zip"
```

> Estas siguen siendo **fotos completas**, no caras recortadas. El recorte/alineado lo hace DeepFaceLab en el **paso 5**.

---

## 2) Alquilar la GPU en RunPod (camino principal)

### 2.1 Elegí la GPU correcta
**Recomendación fuerte: RTX 3090 (Ampere, 24 GB).** El stack TensorFlow 2.4 de DeepFaceLab está **mejor probado en Ampere**. La RTX 4090 es **Ada** — la misma familia que tu 4060 Ti que te congela — y arrastra los mismos dolores de versiones CUDA. La 3090 tiene los mismos 24 GB, es más barata y evita ese lío. (La 4090 entrena más rápido, pero solo si te bancás forzar CUDA 11.8; es zona gris.)

### 2.2 Creá la clave SSH (una sola vez, en tu PC)
```powershell
ssh-keygen -t ed25519 -C "marianomanto@gmail.com"    # Enter a todo (sin passphrase está bien)
Get-Content $env:USERPROFILE\.ssh\id_ed25519.pub      # copiá TODA la salida
```
Pegá esa clave pública en **https://www.console.runpod.io/user/settings** → campo **SSH Public Keys**.

### 2.3 Creá almacenamiento persistente (Network Volume) — CLAVE
El entreno es largo; **no querés perder el modelo** si el pod se cae.

- Panel RunPod → **Storage → Network Volumes → New Network Volume**.
- Elegí un **datacenter que tenga stock de RTX 3090/4090**, tamaño **50–100 GB** (el RTT 1.84 GB + el faceset RTM + tu SRC + checkpoints ocupan varios GB).
- Costo ~**US$0.07/GB/mes** y **sigue cobrando aunque el pod esté apagado o borrado** (el volume vive aparte del pod). Todo lo que guardes en `/workspace` sobrevive a borrar el pod.

> **Importante:** el Network Volume **solo existe en Secure Cloud**, no en Community. Para DFL usá **Secure Cloud + Network Volume**. Community es la mitad de precio pero **sin persistencia y sin SLA**: el host puede desaparecer y perdés todo.

### 2.4 Deployá el pod
Pods → **Deploy** → seleccioná el **Network Volume** que creaste (esto fija la región) → elegí **RTX 3090 24 GB** → **Template**. Dos caminos:

- **(A) Template genérico "RunPod PyTorch" / Ubuntu+CUDA** (recomendado, más controlable): montás el entorno a mano con los comandos del paso 3.
- **(B) Template de escritorio "DeepFaceLab Runpod" (kasm, GUI en el browser)** basado en el fork `DaviSoEditando/DeepFaceLab-Runpod`: más amigable porque ves la **ventana de preview** y clickeás los prompts. **[NO VERIFICADO]** el estado actual del template — su README avisó que el link original fue removido; verificá el template vigente al desplegar. **Ojo de seguridad:** usa user `kasm_user` / pass `password` (default público) en el puerto 6901 — no lo dejes expuesto sin cambiarlo.

Poné **Container Disk 30 GB+** y **Volume Disk 30 GB+**. Click **Deploy On-Demand**.

### 2.5 Conectate al pod
De más simple a más robusto:
- **Web Terminal** (Connect → Open Web Terminal): rápido para comandos cortos, **NO sirve para el entreno** (se corta al cerrar la pestaña, sin scp).
- **SSH proxy** (recomendado para comandos): copiá el comando del botón Connect, del estilo:
  ```powershell
  ssh <POD_ID>-<HASH>@ssh.runpod.io -i $env:USERPROFILE\.ssh\id_ed25519
  ```
  No soporta scp/sftp, pero para correr el entreno en `tmux` va perfecto.
- **SSH sobre TCP con IP pública** (si querés scp/rsync de datasets grandes): Pod → Edit → exponé el **TCP Port 22**, luego `ssh root@<POD_IP> -p <PORT> -i ...`.
- **JupyterLab** (puerto 8888) o **escritorio kasm** (puerto 6901) desde el botón Connect.

> Para archivos, lo más simple es **`runpodctl`** (ya viene preinstalado en el pod) — ver paso 4. No hace falta pelear con puertos.

---

## 3) Instalar DeepFaceLab-Linux (el repo que HOY funciona)

**Contexto:** `iperov/DeepFaceLab` está **ARCHIVADO (read-only, nov-2024)** pero **sigue clonable**. El launcher Linux canónico y mantenido es **`nagadit/DeepFaceLab_Linux`**, que por dentro clona el core de iperov.

> **NO uses** `birdstream/DeepFaceLab_Linux` (GitHub lo deshabilitó por TOS). Google Colab tampoco: entrenar deepfakes viola sus términos y los notebooks viejos están rotos.

Conectado por SSH al pod, dentro de `/workspace` (para que quede en el volume persistente):

```bash
# dependencias de sistema (en muchos templates el driver NVIDIA ya viene)
sudo apt update && sudo apt install -y ffmpeg git wget unzip

# Miniconda si el template no la trae
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p /workspace/miniconda
source /workspace/miniconda/bin/activate

# launcher Linux + core de iperov adentro
cd /workspace
git clone --depth 1 https://github.com/nagadit/DeepFaceLab_Linux.git
cd DeepFaceLab_Linux
git clone --depth 1 https://github.com/iperov/DeepFaceLab.git
```

**Entorno conda — RECETA AMPERE (RTX 3090 / A4000 / A5000).** El README stock fija **CUDA 10.1 / cuDNN 7.6.5**, que en GPUs modernas tira `Illegal instruction (core dumped)` o "no GPU". Esta es LA trampa principal. Usá:

```bash
conda create -y -n deepfacelab -c conda-forge python=3.8 cudatoolkit=11.0 cudnn=8.0.5
conda activate deepfacelab
python -m pip install -r ./DeepFaceLab/requirements-cuda.txt   # trae tensorflow-gpu==2.4.0 contra CUDA 11.0
```

> Si te tocara una **RTX 4090 (Ada)**: misma idea pero base con **CUDA 11.8** (imagen `nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04`). Es más frágil; por eso recomendé 3090.

**Versiones verificadas del core** (por si algo choca): `numpy==1.19.3`, `opencv-python==4.1.0.25`, `scipy==1.4.1`, `h5py==2.10.0`, `tensorflow-gpu==2.4.0`, `tf2onnx==1.9.3`. Esto **exige Python ≤ 3.8** — nada de 3.10+ en este entorno.

Copiá miniconda al volume para que persista entre reinicios del pod:
```bash
cp -r /workspace/miniconda /workspace/miniconda_backup   # opcional, respaldo en el volume
```

---

## 4) Subir tu faceset + bajar el RTT 224 v2 y el RTM WF Faceset V2

**Regla de oro:** lo **tuyo y chico** (faceset SRC) lo **subís**; lo **grande y público** (RTT, RTM) lo **bajás directo dentro del pod** (mucho más rápido que subir desde tu casa).

### 4.1 Subir tu faceset SRC (con `runpodctl`, lo más simple)
En tu **PC** instalá runpodctl y enviá el zip:
```powershell
# instalar runpodctl en Windows (una vez)
wget https://github.com/runpod/runpodctl/releases/latest/download/runpodctl-windows-amd64.exe -O runpodctl.exe
.\runpodctl.exe send "C:\Users\Usuario\Fuser\faceset_src.zip"   # te da un CÓDIGO tipo 8338-galileo-collect-fidel
```
En el **pod**:
```bash
cd /workspace
runpodctl receive 8338-galileo-collect-fidel     # pegá el código que te dio tu PC
```

### 4.2 Bajar el preentrenado RTT 224 v2 y el faceset RTM (dentro del pod)
Mirror verificable en HuggingFace: **`dimanchkek/Deepfacelive-DFM-Models`**, carpeta `/Pretrained` tiene **`RTT model 224 V2.zip` (1.84 GB) CONFIRMADO**.

```bash
pip install -U "huggingface_hub[cli]"
mkdir -p /workspace/dl && cd /workspace/dl
huggingface-cli download dimanchkek/Deepfacelive-DFM-Models "Pretrained/RTT model 224 V2.zip" \
  --repo-type dataset --local-dir /workspace/dl
```

El **RTM WF Faceset V2** (el DST: decenas de miles de caras variadas) se distribuye en los mirrors de la comunidad DFL (HuggingFace / mrdeepfakes / deepfakevfx). **[NO VERIFICADO]** una URL directa estable — buscala en esos mirrors y bajala con `wget "<URL>"` o `huggingface-cli download`. **[NO VERIFICADO]** circulan IDs de Google Drive para ambos (RTT: `1auhf7Wtuwygi8rGFx4EJ4OEgVp1LtQpj`, Faceset: `1jZlh2K0YHzTccTDyk1bxWmyB9kTWyR6c`) que salieron de resúmenes de búsqueda; no los confirmé uno por uno — preferí el mirror de HuggingFace. Si usás Drive: `pip install gdown && gdown <FILE_ID>`.

### 4.3 Colocar todo en la estructura DFL
```bash
cd /workspace/DeepFaceLab_Linux
mkdir -p workspace/model workspace/data_src/aligned workspace/data_dst/aligned

# preentrenado (WARM-START) -> workspace/model/
unzip "/workspace/dl/Pretrained/RTT model 224 V2.zip" -d workspace/model/

# tus fotos SRC curadas -> a una carpeta cruda (NO a aligned todavía; se alinean en el paso 5)
mkdir -p workspace/data_src
unzip /workspace/faceset_src.zip -d workspace/data_src/

# RTM WF Faceset V2 como DST:
#  - si viene como carpeta de caras alineadas -> copialas a workspace/data_dst/aligned/
#  - si viene como faceset.pak -> copiá el .pak a data_dst/aligned/ y desempaquetá:
cp RTM_WF_Faceset_V2.pak workspace/data_dst/aligned/ 2>/dev/null || true
cd scripts && bash 5.2_data_dst_util_faceset_unpack.sh ; cd ..
```

> El **RTM V2 ya viene alineado**: NO lo re-extraigas. Solo tu SRC necesita extracción (paso 5).

---

## 5) Extraer / alinear el SRC

`prep_faceset.py` te dejó **fotos completas**. DeepFaceLab ahora **detecta y recorta las caras alineadas** (S3FD) hacia `data_src/aligned/`:

```bash
cd /workspace/DeepFaceLab_Linux/scripts
bash 4_data_src_extract_faces_S3FD.sh
# recomendado: máscaras XSeg Whole-Face genéricas sobre el SRC (identidad más limpia)
bash 5_XSeg_generic_wf_data_src_apply.sh
```

Verificá que `workspace/data_src/aligned/` tenga tus caras recortadas y decentes. Si hay caras basura o de otra persona, borralas ahora.

---

## 6) Entrenar SAEHD por etapas

Arrancá **siempre dentro de `tmux`** para que el entreno **no muera** al cerrar el SSH:

```bash
cd /workspace/DeepFaceLab_Linux/scripts
tmux new -s train           # detach = Ctrl-b luego d   |   volver = tmux attach -t train
bash 6_train_SAEHD.sh        # (en server headless: 6_train_SAEHD_no_preview.sh)
```

**Arquitectura BLOQUEADA:** como reanudás el RTT, DFL **no te pregunta** res/archi/dims (ya vienen `res:224, WF, liae-udt, ae_dims:512, e_dims:64, d_dims:64, d_mask_dims:32`). Si intentaras cambiarlas, empezarías de cero. Solo confirmás opciones entrenables. **Settings recomendados** (receta del FAQ canónico):

| Opción | Valor | Nota |
|---|---|---|
| `pretrain` | **No** | CRÍTICO: en `Yes` sigue en modo genérico y **no aprende tu cara** |
| `batch_size` | 8 (subí a 12–16) | en 24 GB podés subir para converger más rápido |
| `models_opt_on_gpu` | **Y** | dejalo ON en 24 GB; solo OFF con poca VRAM |
| `random_warp` | Y (etapa 1) | ver abajo |
| `uniform_yaw` | Y | |
| `blur_out_mask` | Y | |
| `lr_dropout` | Y | |
| `eyes_mouth_prio` | N | |

### Procedimiento RTM (hacé BACKUP del modelo antes de cada etapa)

**Etapa 1 — generalizar sobre tu cara (+500.000 iters, borrando `inter_AB.npy` cada 100k):**
El borrado periódico del `inter_AB` es lo que **fuerza al liae-udt a adaptarse a TU cara SRC** generalizando sobre el faceset RTM. Secuencia cada ~100k iters:
```bash
# 1) dejá que el modelo GUARDE (autosave), 2) parás el train, 3) borrás y 4) relanzás:
rm /workspace/DeepFaceLab_Linux/workspace/model/*inter_AB.npy
bash 6_train_SAEHD.sh
```

**Etapa 2 — acabado con GAN (+700.000 iters):** en los prompts poné:
- `random_warp` → **OFF**
- `gan_power` → **0.1**, `gan_patch_size` → **28**, `gan_dims` → **32**

> El GAN puede **degradar** la identidad; por eso el **backup pre-GAN** importa: si empeora, volvés al backup.

**¿Cuándo parar?** El objetivo es que la cara SRC se vea **nítida y estable** en el preview. Como partís de un preentrenado de 20M iters, la identidad llega **mucho antes** que entrenando de cero. Para un resultado "bueno" podés cortar bastante antes del 1.2M total; el recorrido completo es lo que da un `.dfm` que **generaliza a cualquier video**.

---

## 7) Exportar a `.dfm` quantized

Dentro del **mismo entorno conda del pod**:
```bash
cd /workspace/DeepFaceLab_Linux/scripts
bash 6_export_SAEHD_as_dfm.sh        # respondé "Export quantized?" = Yes (fp16: más chico y rápido)
```
Genera `workspace/model/<nombremodelo>.dfm`.

> **[NO VERIFICADO]** que el script `6_export_SAEHD_as_dfm.sh` esté presente en todas las versiones del fork (hay reportes de scripts de export ausentes). **Fallback directo** si falta:
> ```bash
> cd /workspace/DeepFaceLab_Linux
> python DeepFaceLab/main.py exportdfm --model-dir workspace/model --model SAEHD
> ```
> **Consejo DirectML:** si el `.dfm` quantized (fp16) diera artefactos o fallara en tu 4060 Ti (tu historial de quirks DML), re-exportá con **quantized = No (fp32)**: más pesado y lento, pero más seguro. El `.dfm` suele pesar ~600–700 MB.

---

## 8) Bajar el `.dfm` a tu PC

Lo más simple, con `runpodctl` (en el **pod**):
```bash
runpodctl send /workspace/DeepFaceLab_Linux/workspace/model/<nombremodelo>.dfm   # te da un código
```
En tu **PC**:
```powershell
.\runpodctl.exe receive <codigo>     # queda en la carpeta actual
```
Alternativa por scp (si expusiste el TCP 22):
```powershell
scp -P <PORT> -i $env:USERPROFILE\.ssh\id_ed25519 root@<POD_IP>:/workspace/DeepFaceLab_Linux/workspace/model/*.dfm "C:\Users\Usuario\Fuser\"
```

**Bajá el `.dfm` ANTES de apagar nada.** Luego **TERMINÁ el pod** (no solo Stop: un pod detenido sigue cobrando disco, y el Network Volume cobra storage aunque el pod no exista). Si vas a iterar, dejá el volume (cuesta centavos/GB/mes) y en el próximo deploy re-montás el entorno.

---

## 9) Importarlo en Fuser (Biblioteca → 🧬 Asociar `.dfm` → reiniciar)

Fuser tiene el flujo integrado (verificado en el código local): **no** copiás archivos a mano, lo hace la UI.

1. Abrí Fuser (`python app.py` → http://127.0.0.1:7860).
2. **Creá primero una Cara** en el panel **🗂️ Biblioteca de Caras — crear / borrar** (guardá la identidad con unas fotos). El `.dfm` se **asocia a una Cara existente**, así que esta Cara tiene que existir antes.
3. Abrí el acordeón **🧬 Modelo entrenado (.dfm) — máxima geometría**.
4. En **"Cara destino"** elegí la Cara que creaste; en el campo **`.dfm`** subí tu archivo.
5. Click en **🧬 Asociar .dfm a la Cara**.
   - Fuser **copia solo** el `.dfm` a `C:\Users\Usuario\Fuser\fuser\vendor\facefusion\.assets\models\custom\<slug>.dfm` y lo registra como `custom/<slug>` — **no tenés que crear carpetas ni tocar código**.
   - Valida que el `.dfm` sea real (>1 MB); si está truncado, te avisa.
6. **Reiniciá Fuser.** Al arrancar, el Deep Swapper escanea la carpeta `custom` y registra el modelo.
7. Elegí esa **Cara** como fuente y un **video objetivo** cualquiera, y procesá. El `.dfm` trae **horneada** la identidad SRC: **no necesita foto fuente** y se aplica a **cualquier** video (por eso el DST de entreno era el faceset RTM variado, no tu video final). El template es `dfl_whole_face`, que coincide con tu modelo WF. La inferencia corre en **DirectML sobre la 4060 Ti**, sin CUDA.

> Si el modelo no aparece: confirmá que reiniciaste Fuser y que el archivo quedó en `...\vendor\facefusion\.assets\models\custom\`.

---

## Costos estimados (verificá siempre en el momento)

- **GPU (RunPod, RTX 3090 on-demand):** ~US$0.22/hr Community · ~US$0.43–0.46/hr Secure. RTX 4090: US$0.34 Community · US$0.69 Secure. Cobro **por segundo**.
- **Storage RunPod:** Network Volume ~US$0.07/GB/mes (primer TB) · Container Disk US$0.10/GB/mes corriendo, US$0.20/GB/mes detenido. **Todo sigue cobrando con el pod apagado.**
- **Total realista:** prueba corta (100–300k iters, pocas horas) ≈ **US$2–8** + centavos de storage. RTM completo (~1.2M iters, 1–2 días) ≈ **US$10–40**.
- **[NO VERIFICADO]** Los precios cambian mes a mes; confirmá en **runpod.io/pricing** / **vast.ai/pricing** antes de desplegar.

---

## RunPod vs Vast.ai (tabla de decisión)

| | **RunPod** (principal) | **Vast.ai** (alternativa) |
|---|---|---|
| Facilidad no-experto | **Mejor**: UI pulida, templates, `runpodctl` preinstalado, web terminal | Más barato pero más CLI a mano |
| RTX 3090 on-demand | ~US$0.22 (Community) / ~US$0.43–0.46 (Secure) | **~US$0.10–0.25** (más barato) |
| RTX 4090 on-demand | US$0.34 / US$0.69 | ~US$0.29–0.59 |
| Persistencia | **Network Volume** (solo Secure), sobrevive al pod | Disco de la instancia; se cobra **mientras exista**, **NO se agranda** después |
| Interrumpible | Community puede desaparecer; Secure estable | Bid/spot te **pausan** (matan el proceso); usá **on-demand** (sin `--bid`) |
| Transferencia | `runpodctl send/receive`, scp | `vastai copy`, scp (¡`-P` mayúscula en puerto!) |
| Facturación | Por segundo | Por segundo (incluye disco mientras exista) |
| Recomendación | **Empezá acá** (Secure + Network Volume + RTX 3090) | Si querés exprimir el costo y no te asusta el CLI |

### Vast.ai — comandos equivalentes (resumen)
```powershell
pip install --upgrade vastai
vastai set api-key TU_API_KEY
# subí tu id_ed25519.pub en https://cloud.vast.ai/manage-keys/ (Vast SOLO acepta clave, no password)
# buscar 1x RTX 3090 on-demand, host verificado, 80GB+ disco, puerto directo:
vastai search offers 'gpu_name=RTX_3090 num_gpus=1 disk_space>=80 reliability>0.98 verified=true direct_port_count>=1 rentable=true' -o dph
# crear ON-DEMAND (SIN --bid = precio fijo, no interrumpible); pedí 80-100GB de una (no se agranda):
vastai create instance OFFER_ID --image nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04 --disk 80 --ssh --direct
vastai ssh-url INSTANCE_ID        # devuelve ssh://root@IP:PUERTO
# subir faceset (scp usa -P MAYÚSCULA):
scp -P PUERTO "C:\Users\Usuario\Fuser\faceset_src.zip" root@IP:/workspace/
# ... mismo paso 3-7 que RunPod ...
vastai destroy instance INSTANCE_ID   # DESTRUIR para dejar de pagar (Stop NO frena el cobro de disco)
```
**[NO VERIFICADO]** los nombres exactos de campos del query (`verified`, `direct_port_count`, `disk_space`) pueden variar según la versión del CLI; confirmá con `vastai search offers --help`.

---

## Resumen de trampas (leé esto una vez más)

1. **GPU Ampere (RTX 3090), no Ada (4090).** Evitás el infierno de versiones CUDA — el mismo que te congela en local.
2. **CUDA 11.0 / cuDNN 8.0.5** en el conda (no el 10.1 stock), o `Illegal instruction (core dumped)`.
3. **`pretrain = No`** al reanudar el RTT, o no aprende tu cara.
4. **No cambies res/archi/dims** — están bloqueadas por el warm-start.
5. **Entrená dentro de `tmux`** — el Web Terminal y el SSH suelto matan el proceso al desconectar.
6. **Backup antes del GAN** — puede degradar la identidad.
7. **Bajá el `.dfm` ANTES de terminar el pod.** Stop/volume **siguen cobrando**.
8. **Fuser copia el `.dfm` solo** vía "🧬 Asociar .dfm a la Cara" → **reiniciá** para verlo.
9. **Si el `.dfm` fp16 falla en DirectML**, re-exportá fp32 (`quantized = No`).
10. **[NO VERIFICADO]:** URL directa del RTM WF Faceset V2, presencia del script de export en todas las versiones del fork, template kasm vigente de RunPod, y aprobación explícita de face-swap en los TOS. Confirmá cada uno en su fuente.