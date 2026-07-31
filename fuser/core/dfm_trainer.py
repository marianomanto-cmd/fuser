"""Entrenador LOCAL de modelos .dfm (DeepFaceLab DirectX12) orquestado por Fuser.

Permite hacer TODO desde la app: instalar el entrenador (descarga automática,
una vez), preparar el workspace, entrenar en la GPU local y exportar el .dfm —
sin nube ni pasos manuales.

Diseño (restricciones duras de esta máquina):
- El build CUDA de DeepFaceLab CONGELA en la RTX 4060 Ti (Ada). Se usa el build
  **DirectX12** (tensorflow-directml), más lento (~2-3x) pero funcional.
- DeepFaceLab vive en su PROPIA carpeta con su Python embebido. JAMÁS se instala
  nada de esto en el .venv de Fuser (rompería onnxruntime-directml).
- Los procesos de DFL corren como SUBPROCESOS desacoplados (sobreviven al
  cierre de la app); estado vía train.log + train.pid + state.json.
- Mientras entrena, la GPU está ocupada: no conviene procesar videos a la vez.

Piezas que descarga `install()` (a E:\\modelos\\deepfacelab si E: existe):
- Build DeepFaceLab_DirectX12 (Windows, autoextraíble/zip).
- Preentrenado "RTT model 224 V2" (warm-start: baja el entrenamiento de semanas
  a horas/días).
Las URLs viven en constantes al tope del módulo para poder actualizarlas.
"""
from __future__ import annotations

import json
import os
import pickle
import re
import shutil
import subprocess
import time
import zipfile
from pathlib import Path
from typing import Callable, List, Optional

from .. import config
from ..utils.logging import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Ubicaciones y URLs (actualizables)
# ---------------------------------------------------------------------------
def dfl_root() -> Path:
    env = os.environ.get("FUSER_DFL_DIR")
    if env:
        return Path(env)
    e_drive = Path("E:/modelos")
    if e_drive.is_dir():
        return e_drive / "deepfacelab"
    return config.PROJECT_ROOT / "dfl"


# URLs VERIFICADAS (2026-07, mirror HF dimanchkek/Deepfacelive-DFM-Models, GPL-3.0;
# HEAD 200, CDN con soporte Range para reanudar). Si cambian de mirror, actualizá
# SOLO estas líneas (o las env vars).
BUILD_URL = os.environ.get(
    "FUSER_DFL_BUILD_URL",
    "https://huggingface.co/datasets/dimanchkek/Deepfacelive-DFM-Models/resolve/main/Pre-builds/DeepFaceLab_DirectX12_build_05_04_2022.exe",
)
BUILD_SHA256 = "dd666c196e5053a57c6aad08caa870a5e85207c12dd4ed95f5b5718235febeda"  # 2.783.049.668 bytes

# ---------------------------------------------------------------------------
# MOTORES de entrenamiento (conviven; el usuario elige cuál usar)
# ---------------------------------------------------------------------------
#  dml  : DirectX12 — LENTO pero FUNCIONA seguro en esta GPU (el de siempre).
#  cuda : build NVIDIA — ~3x más rápido SI arranca; es de 2021 (CUDA 11.x) y la
#         4060 Ti es Ada, arquitectura posterior: hay freezes reportados en este
#         mismo modelo de placa. Por eso NO reemplaza al dml: se prueba primero.
BACKENDS = {
    "dml": {
        "label": "DirectX12 (estable)",
        "url": BUILD_URL,
        "sha256": BUILD_SHA256,
        "exe": "DeepFaceLab_DirectX12_build_05_04_2022.exe",
        "dir_glob": "DeepFaceLab_DirectX12*",
    },
    "cuda": {
        "label": "NVIDIA CUDA (rápido, experimental)",
        "url": os.environ.get(
            "FUSER_DFL_CUDA_URL",
            "https://huggingface.co/datasets/dimanchkek/Deepfacelive-DFM-Models/resolve/main/"
            "Pre-builds/DeepFaceLab_NVIDIA_RTX3000_series_build_11_20_2021.exe"),
        "sha256": "4ca31c30ca8f683a825a643e7090811d750c1250775537dcdb5c80d5f3b7f722",
        "exe": "DeepFaceLab_NVIDIA_RTX3000_series_build_11_20_2021.exe",
        "dir_glob": "DeepFaceLab_NVIDIA*",
    },
}
DEFAULT_BACKEND = "dml"


def _backend_file() -> Path:
    return dfl_root() / "backend.txt"


def active_backend() -> str:
    """Motor elegido para los PRÓXIMOS entrenamientos (default: el estable)."""
    try:
        b = _backend_file().read_text(encoding="utf-8").strip()
        return b if b in BACKENDS else DEFAULT_BACKEND
    except OSError:
        return DEFAULT_BACKEND


def set_active_backend(backend: str) -> str:
    if backend not in BACKENDS:
        raise ValueError(f"Motor desconocido: {backend}")
    dfl_root().mkdir(parents=True, exist_ok=True)
    _backend_file().write_text(backend, encoding="utf-8")
    log.info("Motor de entrenamiento activo: %s", backend)
    return BACKENDS[backend]["label"]
RTT_URL = os.environ.get(
    "FUSER_DFL_RTT_URL",
    "https://huggingface.co/datasets/dimanchkek/Deepfacelive-DFM-Models/resolve/main/Pretrained/RTT%20model%20224%20V2.zip",
)
RTT_SHA256 = "0f5f4a4b5bfc48df1fa2c4be8be89dae8fd6a664e4d81eae5eaf4fbb3e84d227"  # 1.842.844.041 bytes
# Faceset genérico RTM (DST "universal"): solo se baja si el usuario NO aporta
# videos destino. ~8.8 GB, caras YA alineadas (no se re-extraen).
RTM_URL = os.environ.get(
    "FUSER_DFL_RTM_URL",
    "https://huggingface.co/datasets/dimanchkek/Deepfacelive-DFM-Models/resolve/main/Facesets/RTM%20WF%20Faceset.zip",
)
RTM_SIZE = 9_494_305_175  # bytes (verificación de integridad por tamaño)

PROGRESS_RE = re.compile(r"\[?#?(\d{4,9})\]?\[(\d+)ms\]\[([\d.]+)\]\[([\d.]+)\]")


# ---------------------------------------------------------------------------
# Rutas derivadas
# ---------------------------------------------------------------------------
def _paths(backend: Optional[str] = None) -> dict:
    """Rutas del motor pedido (default: el activo). Los builds CONVIVEN."""
    backend = backend if backend in BACKENDS else active_backend()
    root = dfl_root()
    build = root / "build"
    glob_pat = BACKENDS[backend]["dir_glob"]
    inner = next((p for p in build.glob(glob_pat) if p.is_dir()), build / glob_pat.rstrip("*"))
    internal = inner / "_internal"
    return {
        "root": root,
        "backend": backend,
        "downloads": root / "downloads",
        "build": build,
        "internal": internal,
        "python": internal / "python-3.6.8" / "python.exe",
        "main": internal / "DeepFaceLab" / "main.py",
        "rtt": root / "assets" / "rtt_224_v2",
        "workspaces": root / "workspaces",
    }


def _find_python(paths: dict) -> Optional[Path]:
    """El Python embebido cambia de nombre según el build: buscalo."""
    if paths["python"].is_file():
        return paths["python"]
    internal = paths["internal"]
    if internal.is_dir():
        for cand in internal.glob("python*/python.exe"):
            return cand
    return None


def workspace_of(slug: str) -> Path:
    return _paths()["workspaces"] / slug / "workspace"


def _state_file(slug: str) -> Path:
    return _paths()["workspaces"] / slug / "state.json"


import threading

_STATE_LOCK = threading.Lock()   # autopiloto (hilo) vs handlers de la UI
_EXPORT_LOCK = threading.Lock()  # nunca dos exportdfm sobre el mismo model-dir


def _read_state(slug: str) -> dict:
    try:
        return json.loads(_state_file(slug).read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_state(slug: str, **kw) -> None:
    with _STATE_LOCK:
        st = _read_state(slug)
        st.update(kw)
        f = _state_file(slug)
        f.parent.mkdir(parents=True, exist_ok=True)
        tmp = f.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(st, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, f)  # escritura atómica


# ---------------------------------------------------------------------------
# Estado global del entrenador
# ---------------------------------------------------------------------------
def rtm_ready() -> bool:
    """¿El faceset genérico (DST universal + donantes de síntesis) está listo?"""
    d = dfl_root() / "assets" / "rtm_faceset"
    if not d.is_dir():
        return False
    return any(d.rglob("faceset.pak")) or next(d.rglob("*.jpg"), None) is not None


def backend_ready(backend: str) -> bool:
    """¿Está instalado (descargado y descomprimido) ese motor?"""
    p = _paths(backend)
    return bool(_find_python(p) and _find_main(p))


def backends_status() -> dict:
    """Estado de TODOS los motores + cuál está activo (para la pestaña CUDA)."""
    return {
        "active": active_backend(),
        "backends": {b: {"label": BACKENDS[b]["label"], "ready": backend_ready(b)}
                     for b in BACKENDS},
    }


def status() -> dict:
    p = _paths()
    py = _find_python(p)
    # marker .complete = extracción entera; legado: algún .npy (instalaciones previas)
    rtt_ok = ((p["rtt"] / ".complete").is_file() or
              next(p["rtt"].rglob("*.npy"), None) is not None) if p["rtt"].is_dir() else False
    return {
        "root": str(p["root"]),
        "build_ready": bool(py and p["main"].parent.is_dir() and _find_main(p)),
        "rtt_ready": rtt_ok,
        "rtm_ready": rtm_ready(),
        "python": str(py) if py else None,
    }


def _find_main(p: dict) -> Optional[Path]:
    if p["main"].is_file():
        return p["main"]
    internal = p["internal"]
    if internal.is_dir():
        for cand in internal.glob("DeepFaceLab*/main.py"):
            return cand
    return None


# ---------------------------------------------------------------------------
# Descarga con reanudación
# ---------------------------------------------------------------------------
def _download(url: str, dst: Path, progress: Optional[Callable] = None, label: str = "",
              retry_wait: int = 15, max_stalls: int = 2000) -> Path:
    """Descarga en streaming REANUDABLE y tolerante a cortes de internet.

    - Estado en ``<dst>.part``: si el proceso o la conexión mueren, la próxima
      llamada (o el reintento automático) continúa con HTTP Range desde donde
      quedó — nunca se re-descarga lo ya bajado.
    - Ante un corte reintenta solo (espera ``retry_wait`` s), sin abortar el
      paso. ``max_stalls`` = tope de reintentos SIN progreso (guardia anti-bucle;
      con progreso el contador se resetea).
    """
    import urllib.error
    import urllib.request

    dst.parent.mkdir(parents=True, exist_ok=True)
    part = dst.with_suffix(dst.suffix + ".part")
    total_known = 0
    stalls = 0
    while True:
        have = part.stat().st_size if part.exists() else 0
        req = urllib.request.Request(url, headers={"User-Agent": "fuser-dfm-trainer"})
        if have:
            req.add_header("Range", f"bytes={have}-")
        got_bytes = False
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                if have and getattr(r, "status", 206) == 200:
                    # el servidor IGNORÓ el Range y manda el cuerpo completo:
                    # appendear corrompería el .part → reiniciar desde cero.
                    log.warning("%s: el servidor no soporta Range; reiniciando descarga.", dst.name)
                    part.unlink(missing_ok=True)
                    continue
                cl = int(r.headers.get("Content-Length") or 0)
                total_known = have + cl if cl else total_known
                done = have
                with open(part, "ab" if have else "wb") as fh:
                    while True:
                        chunk = r.read(1024 * 512)
                        if not chunk:
                            break
                        fh.write(chunk)
                        done += len(chunk)
                        got_bytes = True
                        if progress and total_known:
                            progress(done / total_known,
                                     f"{label} {done // 1048576}/{total_known // 1048576} MB")
            # stream terminó limpio: ¿está completa?
            size = part.stat().st_size if part.exists() else 0
            if not total_known or size >= total_known:
                break
            # quedó corta (corte silencioso): reanudar
            log.warning("%s: stream cortado en %d/%d MB; reanudando…",
                        dst.name, size // 1048576, total_known // 1048576)
        except urllib.error.HTTPError as exc:
            if exc.code == 416:  # Range fuera de rango = ya estaba completa
                break
            if exc.code in (403, 404, 410):
                # error PERMANENTE (URL rota/mirror caído): reintentar no ayuda.
                raise RuntimeError(f"{dst.name}: HTTP {exc.code} — la URL parece rota; "
                                   f"actualizá el mirror (env FUSER_DFL_*_URL).")
            log.warning("%s: HTTP %s; reintento en %ds…", dst.name, exc.code, retry_wait)
        except OSError as exc:
            if isinstance(exc, (urllib.error.URLError, TimeoutError, ConnectionError)):
                log.warning("%s: conexión interrumpida (%s); reintento en %ds…",
                            dst.name, str(exc)[:80], retry_wait)
                if progress:
                    progress(min(0.99, (part.stat().st_size if part.exists() else 0) /
                                 max(1, total_known or 1)),
                             f"{label} conexión caída — reintentando solo…")
            else:
                # error LOCAL (disco lleno/permisos): fallar rápido, no loopear.
                raise
        stalls = 0 if got_bytes else stalls + 1
        if stalls >= max_stalls:
            raise RuntimeError(f"{dst.name}: {max_stalls} reintentos sin progreso; "
                               f"revisá la conexión y volvé a intentar (lo bajado se conserva).")
        time.sleep(retry_wait)
    os.replace(part, dst)  # atómico y pisa dst si quedó de una corrida anterior
    return dst


def _sha256(path: Path, progress: Optional[Callable] = None, label: str = "") -> str:
    import hashlib
    h = hashlib.sha256()
    total = path.stat().st_size
    done = 0
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(1024 * 1024 * 4)
            if not chunk:
                break
            h.update(chunk)
            done += len(chunk)
            if progress and total:
                progress(done / total, f"{label} verificando integridad…")
    return h.hexdigest()


def _fetch_verified(url: str, dst: Path, sha256: str, progress, label: str) -> Path:
    """Descarga (con reanudación) y verifica SHA256; borra y reintenta una vez si falla."""
    for attempt in (1, 2):
        if not dst.exists():
            _download(url, dst, progress, label)
        digest = _sha256(dst, progress, label)
        if digest == sha256:
            return dst
        log.warning("%s: SHA256 no coincide (intento %d): %s", dst.name, attempt, digest)
        dst.unlink(missing_ok=True)
    raise RuntimeError(f"{dst.name}: la descarga llegó corrupta dos veces (SHA256 no coincide). "
                       f"Reintentá más tarde.")


def _seven_zip() -> Optional[Path]:
    for cand in (Path("C:/Program Files/7-Zip/7z.exe"),
                 Path("C:/Program Files (x86)/7-Zip/7z.exe")):
        if cand.is_file():
            return cand
    return None


def install(progress: Optional[Callable] = None, backend: Optional[str] = None) -> str:
    """Instala un motor de entrenamiento (+ el preentrenado RTT, compartido).

    Los motores CONVIVEN en carpetas separadas: instalar CUDA no toca el
    DirectX12 que ya funciona. ``backend`` default = el activo.
    """
    backend = backend if backend in BACKENDS else active_backend()
    cfg = BACKENDS[backend]
    p = _paths(backend)
    msgs = []
    if not backend_ready(backend):
        exe = p["downloads"] / cfg["exe"]
        _fetch_verified(cfg["url"], exe, cfg["sha256"], progress,
                        f"Build {cfg['label']}:")
        if progress:
            progress(0.5, "Desempaquetando el build (unos GB)…")
        p["build"].mkdir(parents=True, exist_ok=True)
        # el .exe es un 7-Zip SFX (verificado): extrae headless con -y -o<dir>;
        # si hay 7-Zip local, es aún más robusto.
        sz = _seven_zip()
        if sz is not None:
            r = subprocess.run([str(sz), "x", str(exe), f"-o{p['build']}", "-y"],
                               capture_output=True, timeout=3600)
        else:
            r = subprocess.run([str(exe), "-y", f"-o{p['build']}"], capture_output=True, timeout=3600)
        if r.returncode != 0:
            raise RuntimeError(f"No pude desempaquetar el build (rc={r.returncode}).")
        msgs.append(f"build {cfg['label']} instalado")
    if not status()["rtt_ready"]:
        z = p["downloads"] / "RTT_model_224_V2.zip"
        _fetch_verified(RTT_URL, z, RTT_SHA256, progress, "Preentrenado RTT 224:")
        if progress:
            progress(0.9, "Desempaquetando el preentrenado…")
        # extraer a staging y renombrar: un crash a mitad no deja un RTT "a medias"
        staging = p["rtt"].with_name(p["rtt"].name + "_staging")
        shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(z) as zf:
            zf.extractall(staging)
        shutil.rmtree(p["rtt"], ignore_errors=True)
        os.replace(staging, p["rtt"])
        (p["rtt"] / ".complete").write_text("ok", encoding="utf-8")
        msgs.append("preentrenado RTT listo")
    if not backend_ready(backend):
        raise RuntimeError(f"El build {cfg['label']} quedó incompleto: "
                           f"no encuentro python embebido/main.py.")
    _patch_trainer_autosave(backend=backend)
    return (f"✅ Motor «{cfg['label']}» instalado ("
            + ", ".join(msgs or ["ya estaba"]) + f") en {p['root']}")


# ---------------------------------------------------------------------------
# Subprocesos DFL
# ---------------------------------------------------------------------------
def _run_dfl(args: List[str], log_file: Path, cwd: Optional[Path] = None,
             detach: bool = False, stdin_text: Optional[str] = None,
             backend: Optional[str] = None):
    p = _paths(backend)
    py = _find_python(p)
    main = _find_main(p)
    if not (py and main):
        raise RuntimeError("El entrenador no está instalado (falta build). Corré la instalación.")
    env = dict(os.environ)
    internal = main.parent.parent  # _internal
    # réplica del setenv.bat del build (INTERNAL/DFL_ROOT/PATH del python embebido
    # + ffmpeg incluido). Sin esto algunos módulos no resuelven rutas/DLLs.
    env["INTERNAL"] = str(internal)
    env["DFL_ROOT"] = str(main.parent)
    env["PYTHONPATH"] = str(main.parent)
    # PATH según el setenv.bat del build. Las carpetas CUDA/CUDNN son OBLIGATORIAS:
    # sin ellas TensorFlow no encuentra cudart64_110.dll/cudnn64_8.dll, imprime
    # "Skipping registering GPU devices" y entrena EN CPU (medido: 20-30 s/iter
    # contra 3 s de DirectX12). Era la causa real del "CUDA es lentísimo".
    path_parts = [str(py.parent), str(py.parent / "Scripts")]
    for sub in ("CUDA", "CUDNN", "CUDNN/Win10.0", "CUDNN/Win6.x"):
        d = internal / sub
        if d.is_dir():
            path_parts.append(str(d))
    path_parts += [str(internal / "ffmpeg"), env.get("PATH", "")]
    env["PATH"] = os.pathsep.join(path_parts)
    env.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    if p["backend"] == "cuda":
        # MEDIDO: el build trae CUDA 11.0 (cudart64_110.dll), que solo tiene
        # kernels hasta sm_80; esta GPU es Ada (sm_89), así que el driver compila
        # PTX en vivo (JIT). Con la caché por defecto quedaban 45 KB / 5 archivos
        # = se RECOMPILA casi todo, todo el tiempo -> 20 s/iter (vs 3 s en DX12) y
        # el "freeze" inicial. Caché grande y propia: compila UNA vez y reutiliza.
        cache = dfl_root() / "cuda_jit_cache"
        cache.mkdir(parents=True, exist_ok=True)
        env["CUDA_CACHE_PATH"] = str(cache)
        env["CUDA_CACHE_MAXSIZE"] = str(4 * 1024 ** 3)   # 4 GB (default ~256 MB)
        env["CUDA_CACHE_DISABLE"] = "0"
        env.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
        # El allocator de memoria HOST ANCLADA (gpu_host_bfc) tiene un tope chico
        # por defecto y revienta con "OOM shape[25088,512] ... device:CPU:0" al
        # construir el grafo del optimizador (medido, incluso con el optimizador
        # en GPU). Con 40 GB de RAM podemos darle 12 GB sin drama.
        env.setdefault("TF_GPU_HOST_MEM_LIMIT_IN_MB", "12288")
    cmd = [str(py), str(main)] + args
    log_file.parent.mkdir(parents=True, exist_ok=True)
    lf = open(log_file, "ab")
    kw = dict(cwd=str(cwd or main.parent), env=env, stdout=lf, stderr=subprocess.STDOUT)
    if detach:
        # DETACHED_PROCESS | NEW_PROCESS_GROUP.
        # ⚠️ NO cambiar a CREATE_NO_WINDOW para ocultar las ventanas de los ~25
        # workers: se probó y CUELGA el entrenamiento justo tras "Sort by yaw"
        # (medido: 0,1 s de CPU en 20 s, sin una sola iteración, tanto en DX12
        # como en CUDA). Los workers de DFL son procesos de consola y necesitan
        # la consola propia que les da DETACHED. Las ventanas son molestas pero
        # el entrenamiento funciona; si se quiere ocultarlas, hacerlo DESPUÉS del
        # arranque con ShowWindow, nunca cambiando estos flags.
        kw["creationflags"] = 0x00000008 | 0x00000200  # DETACHED_PROCESS | NEW_PROCESS_GROUP
        kw["stdin"] = subprocess.DEVNULL if stdin_text is None else subprocess.PIPE
        proc = subprocess.Popen(cmd, **kw)
        if stdin_text is not None:
            try:
                proc.stdin.write(stdin_text.encode()); proc.stdin.close()
            except Exception:
                pass
        return proc
    kw["stdin"] = subprocess.PIPE
    proc = subprocess.Popen(cmd, **kw)
    if stdin_text:
        try:
            proc.stdin.write(stdin_text.encode())
        except Exception:
            pass
    try:
        proc.stdin.close()
    except Exception:
        pass
    return proc


def _read_model_options(model_dir: Path) -> dict:
    """Opciones guardadas del modelo SAEHD (lectura; {} si no se puede leer)."""
    for dat in model_dir.glob("*_SAEHD_data.dat"):
        try:
            with open(dat, "rb") as fh:
                data = pickle.load(fh)
            saved = data.get("options", data) if isinstance(data, dict) else None
            if isinstance(saved, dict):
                return saved
        except Exception as exc:
            log.warning("No pude leer las opciones de %s: %s", dat, exc)
    return {}


def _patch_model_options(model_dir: Path, **opts) -> bool:
    """Edita las opciones guardadas del modelo SAEHD (…_data.dat, pickle plano).

    Es el camino robusto para forzar pretrain=False/batch sin coreografiar
    prompts interactivos: DFL lee estas opciones al reanudar con --silent-start.
    """
    for dat in model_dir.glob("*_SAEHD_data.dat"):
        try:
            with open(dat, "rb") as fh:
                data = pickle.load(fh)
            saved = data.get("options", data) if isinstance(data, dict) else None
            if saved is None:
                continue
            saved.update(opts)
            if isinstance(data, dict) and "options" in data:
                data["options"] = saved
            with open(dat, "wb") as fh:
                pickle.dump(data, fh)
            log.info("Opciones del modelo parcheadas en %s: %s", dat.name, opts)
            return True
        except Exception as exc:
            log.warning("No pude parchear %s: %s", dat, exc)
    return False


# ---------------------------------------------------------------------------
# Preparación del workspace
# ---------------------------------------------------------------------------
def _ensure_rtm(progress: Optional[Callable] = None) -> Path:
    """Garantiza el faceset genérico RTM (DST universal) y devuelve su carpeta.

    Contiene caras YA alineadas (o un faceset.pak, que DFL lee nativo): no se
    re-extraen. Descarga única de ~8.8 GB compartida entre todos los modelos.
    """
    p = _paths()
    dst = p["root"] / "assets" / "rtm_faceset"
    def _content(d: Path):
        if not d.is_dir():
            return None
        paks = list(d.rglob("faceset.pak"))
        if paks:
            return paks[0].parent
        jpgs = [x for x in d.rglob("*.jpg")][:1]
        return jpgs[0].parent if jpgs else None
    found = _content(dst)
    if found:
        return found
    z = p["downloads"] / "RTM_WF_Faceset.zip"
    if not (z.exists() and z.stat().st_size == RTM_SIZE):
        _download(RTM_URL, z, progress, "Faceset genérico (8.8 GB):")
        if z.stat().st_size != RTM_SIZE:
            z.unlink(missing_ok=True)
            raise RuntimeError("La descarga del faceset genérico llegó incompleta. Reintentá.")
    if progress:
        progress(0.95, "Desempaquetando el faceset genérico…")
    dst.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(z) as zf:
        zf.extractall(dst)
    found = _content(dst)
    if not found:
        raise RuntimeError("El faceset genérico no trae caras reconocibles (revisá el zip).")
    return found


def prepare(name: str, src_dir: Path, dst_videos: List[str],
            progress: Optional[Callable] = None) -> str:
    """Arma TODO el material que DeepFaceLab necesita, desde las imágenes.

    - SRC: fotos curadas (paso ①) → caras alineadas (extractor S3FD de DFL).
    - DST: dos caminos —
        · CON videos destino: frames de tus videos → caras alineadas (el modelo
          aprende las condiciones reales de ESOS videos; 100% local).
        · SIN videos (solo imágenes): usa el faceset genérico RTM (descarga
          automática única de ~8.8 GB) → modelo "universal" para cualquier video.
    - Modelo: semilla del preentrenado RTT (warm-start) + pretrain=OFF forzado.
    """
    from ..core.face_library import _slug
    from ..utils import video as videoutil

    slug = _slug(name)
    if not slug:
        raise ValueError("Nombre inválido.")
    src_dir = Path(src_dir)
    if not src_dir.is_dir() or not any(src_dir.iterdir()):
        raise ValueError("No encuentro las fotos curadas. Corré primero el paso ① (curar fotos).")
    st = status()
    if not st["build_ready"]:
        raise ValueError("El entrenador no está instalado. Corré el paso ② (instalar).")

    ws = workspace_of(slug)
    data_src = ws / "data_src"
    data_dst = ws / "data_dst"
    model = ws / "model"

    def _is_junction(p: Path) -> bool:
        import stat as _stat
        try:
            return bool(os.stat(p, follow_symlinks=False).st_file_attributes
                        & _stat.FILE_ATTRIBUTE_REPARSE_POINT)
        except OSError:
            return False

    def _rm_dir_safe(p: Path) -> None:
        """Borra un dir del workspace SIN atravesar junctions (protege el RTM en E:)."""
        if not p.exists():
            return
        aligned = p / "aligned"
        if aligned.exists() and _is_junction(aligned):
            os.rmdir(aligned)  # desmonta el junction; el destino queda intacto
        if _is_junction(p):
            os.rmdir(p)
            return
        shutil.rmtree(p, ignore_errors=True)

    # LIMPIEZA de corridas anteriores: sin esto se mezclan datasets viejos y
    # nuevos (y el conteo de éxito del extract se satisface con jpgs rancios).
    for d in (data_src, data_dst, ws / "synth", ws / "synth_p1"):
        _rm_dir_safe(d)
    for d in (data_src, data_dst, model):
        d.mkdir(parents=True, exist_ok=True)

    real_files = [f for f in sorted(src_dir.iterdir())
                  if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp", ".bmp")]
    n_real = len(real_files)

    def _extract_faces(in_dir: Path, phase_name: str, frac: float, timeout: int = 14400) -> int:
        """Extractor S3FD de DFL → aligned/. JPEG q100 = sin pérdida visible."""
        if progress:
            progress(frac, f"Detectando caras ({phase_name}) — puede tardar bastante…")
        out_aligned = in_dir / "aligned"
        out_aligned.mkdir(exist_ok=True)
        logf = ws.parent / "prepare.log"
        proc = _run_dfl([
            "extract", "--input-dir", str(in_dir), "--output-dir", str(out_aligned),
            "--detector", "s3fd", "--face-type", "whole_face",
            "--max-faces-from-image", "1", "--image-size", "512", "--jpeg-quality", "100",
            "--no-output-debug",
        ], logf, stdin_text="\n" * 8)
        try:
            rc = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()  # sin esto queda un DFL zombie ocupando la GPU
            raise RuntimeError(f"Extracción ({phase_name}) superó {timeout}s; proceso matado.")
        n_faces = len(list(out_aligned.glob("*.jpg")))
        if rc != 0 or n_faces == 0:
            raise RuntimeError(
                f"La extracción de caras ({phase_name}) falló (rc={rc}, caras={n_faces}). "
                f"Mirá el log: {logf}")
        return n_faces

    # ---- 1) DST primero (además provee las caras DONANTES para la síntesis) ----
    dst_mode = "videos" if dst_videos else "universal"
    donor_files: List[Path] = []
    if dst_videos:
        # 1a) frames de los videos del usuario (cap total ~1500 frames)
        if progress:
            progress(0.05, "Extrayendo frames de tus videos…")
        total_frames = 0
        per_video = max(200, 1500 // max(1, len(dst_videos)))
        for vi, v in enumerate(dst_videos):
            try:
                info = videoutil.probe(v)
                step = max(1, info.frame_count // per_video)
                idxs = list(range(0, info.frame_count, step))[:per_video]
                frames = videoutil.get_frames_at(v, idxs)
                import cv2
                for fi, fr in enumerate(frames):
                    if fr is not None:
                        # PNG sin pérdida: es material de entrenamiento
                        cv2.imwrite(str(data_dst / f"v{vi:02d}_{fi:05d}.png"), fr)
                        total_frames += 1
            except Exception as exc:
                log.warning("No pude extraer frames de %s: %s", v, exc)
        if total_frames == 0:
            raise ValueError("No pude extraer frames de los videos destino.")
        _extract_faces(data_dst, "destino", 0.1)
        donor_files = sorted((data_dst / "aligned").glob("*.jpg"))
    else:
        # 1b) DST universal: faceset genérico RTM (ya alineado, NO se re-extrae).
        if progress:
            progress(0.05, "Sin videos destino: preparando el faceset genérico (única vez ~8.8 GB)…")
        rtm_dir = _ensure_rtm(progress)
        aligned = data_dst / "aligned"
        aligned.mkdir(exist_ok=True)
        donor_files = sorted(rtm_dir.glob("*.jpg"))
        # junction (sin copiar 8.8 GB): data_dst/aligned -> carpeta compartida
        try:
            aligned.rmdir()
            subprocess.run(["cmd", "/c", "mklink", "/J", str(aligned), str(rtm_dir)],
                           capture_output=True, timeout=30, check=True)
        except Exception:
            aligned.mkdir(exist_ok=True)
            pak = next(iter(rtm_dir.glob("faceset.pak")), None)
            if pak is not None:
                shutil.copyfile(pak, aligned / "faceset.pak")
        # VALIDACIÓN: el DST tiene que existir de verdad (pak o miles de jpgs);
        # sin esto un fallo silencioso entrenaría contra un destino vacío.
        has_dst = (aligned / "faceset.pak").is_file() or \
            next(iter(aligned.glob("*.jpg")), None) is not None
        if not has_dst:
            raise RuntimeError("No pude montar el faceset genérico como destino "
                               "(ni junction ni faceset.pak). Revisá E: y permisos.")

    # ---- 2) SRC: con pocas caras reales, SINTETIZAR el faceset ------------------
    from . import faceset_synth
    synth_info = None
    if n_real < faceset_synth.SYNTH_THRESHOLD and not donor_files and not dst_videos:
        # RTM viene SOLO como faceset.pak (verificado): para donar caras a la
        # síntesis hay que desempaquetarlo UNA vez con el util de DFL.
        rtm_dir = _ensure_rtm(progress)
        pak = next(iter(rtm_dir.glob("faceset.pak")), None)
        if pak is not None:
            if progress:
                progress(0.15, "Desempaquetando el faceset genérico para la síntesis (una vez)…")
            logf = ws.parent / "prepare.log"
            proc = _run_dfl(["util", "--input-dir", str(rtm_dir), "--unpack-faceset"],
                            logf, stdin_text="\n" * 4)
            try:
                proc.wait(timeout=3600)
            except subprocess.TimeoutExpired:
                proc.kill()
            donor_files = sorted(rtm_dir.glob("*.jpg"))
    if n_real < faceset_synth.SYNTH_THRESHOLD and donor_files:
        if progress:
            progress(0.2, f"Solo {n_real} fotos reales: sintetizando faceset "
                          f"(~{min(faceset_synth.SYNTH_TARGET, len(donor_files))} caras; "
                          f"puede tardar 1-2 h, una sola vez)…")
        synth_dir = ws / "synth"
        synth_info = faceset_synth.synthesize(
            src_dir, donor_files, synth_dir,
            progress=lambda f, m="": progress(0.2 + f * 0.4, m) if progress else None)
        dup = faceset_synth.real_duplication(n_real, synth_info["synthetic"])
        # dataset SRC = reales ×dup (ancla, ~15%) + sintéticas (cobertura de poses)
        k = 0
        for f in real_files:
            for d in range(dup):
                shutil.copyfile(f, data_src / f"real_{k:05d}{f.suffix.lower()}")
                k += 1
        for f in sorted(synth_dir.glob("*.png")):
            shutil.copyfile(f, data_src / f.name)
    else:
        if progress:
            progress(0.2, "Copiando fotos fuente…")
        for i, f in enumerate(real_files):
            shutil.copyfile(f, data_src / f"{i:05d}{f.suffix.lower()}")

    # ---- 3) Extraer caras (WF) del SRC -----------------------------------------
    n_src = _extract_faces(data_src, "fuente", 0.72)

    # ---- 4) Semilla del modelo: RTT (warm-start) + opciones seguras 8GB DX12 ----
    if progress:
        progress(0.92, "Sembrando el preentrenado RTT…")
    p = _paths()
    seeded = 0
    for f in p["rtt"].rglob("*"):
        if f.is_file() and f.suffix in (".npy", ".dat"):
            shutil.copyfile(f, model / f.name)
            seeded += 1
    if seeded == 0:
        raise RuntimeError("El preentrenado RTT no está instalado (paso ②).")
    # MAXIMIZAR VRAM+RAM (principio de la app): optimizer a RAM (40 GB) libera
    # VRAM → batch más alto = más VRAM útil por iteración. Env-tuneable; el
    # harness de prueba valida el valor en esta GPU (8GB DX12, res 224).
    batch = int(os.environ.get("FUSER_DFM_BATCH", "8"))
    # write_preview_history: DFL guarda en <modelo>_history/ una imagen por
    # autoguardado (cada 5 min) con [tu cara | reconstruida | destino |
    # reconstruido | TU CARA EN EL DESTINO]. Es la ventana para ver si el
    # modelo va bien SIN esperar a que termine (la muestra la UI).
    _patch_model_options(model, pretrain=False, batch_size=batch,
                         models_opt_on_gpu=False, write_preview_history=True)
    # el RTT trae sus PROPIAS muestras de preview cacheadas: limpiarlas para que
    # el preview se regenere con TU cara (si no, muestra las caras del RTT).
    _reset_preview_samples(model)
    shutil.rmtree(model / "new_SAEHD_history", ignore_errors=True)  # historia del preentrenado

    n_dst = len(list((data_dst / "aligned").glob("*.jpg")))
    dst_txt = (f"{n_dst} caras destino (de tus videos)" if dst_videos
               else "faceset genérico universal (miles de caras)")
    src_txt = (f"{n_src} caras fuente ({n_real} reales ancladas + "
               f"{synth_info['synthetic']} sintetizadas con el motor 🎯➕)" if synth_info
               else f"{n_src} caras fuente")
    _write_state(slug, phase="prepared", dst_mode=dst_mode,
                 src_faces=n_src, dst_faces=n_dst,
                 synthetic=(synth_info or {}).get("synthetic", 0), real=n_real,
                 prepared_at=time.strftime("%Y-%m-%d %H:%M:%S"))
    return (f"✅ Workspace listo: {src_txt} · {dst_txt} · "
            f"modelo sembrado del RTT (pretrain OFF, batch {batch}). Ya podés entrenar.")


# ---------------------------------------------------------------------------
# Entrenamiento
# ---------------------------------------------------------------------------
# Objetivo de iteraciones al que el AUTOPILOTO corta, exporta y registra el .dfm.
# Con warm-start del RTT, ~400k es un buen equilibrio calidad/tiempo en 8GB DX12.
AUTO_TARGET_ITERS = int(os.environ.get("FUSER_DFM_TARGET_ITERS", "400000"))


def _patch_trainer_autosave(minutes: int = 5, backend: Optional[str] = None) -> None:
    """Acorta el autosave del Trainer de DFL (default 25 min → 5).

    Sin esto, pausar/exportar puede tirar hasta 25 min de entrenamiento (medido:
    465 iters perdidas — el modelo quedó en iter=1). Idempotente; corre en
    install() y como cinturón en start().
    """
    main = _find_main(_paths(backend))
    if not main:
        return
    trainer = main.parent / "mainscripts" / "Trainer.py"
    try:
        src = trainer.read_text(encoding="utf-8")
        if "save_interval_min = 25" in src:
            trainer.write_text(src.replace("save_interval_min = 25",
                                           f"save_interval_min = {minutes}"),
                               encoding="utf-8")
            log.info("Trainer de DFL parcheado: autosave cada %d min.", minutes)
    except OSError as exc:
        log.warning("No pude parchear el autosave del Trainer: %s", exc)


def _reset_preview_samples(model_dir: Path) -> bool:
    """Borra las muestras de preview CACHEADAS que trae el preentrenado.

    ModelBase guarda ``sample_for_preview`` DENTRO del .dat del modelo y solo lo
    regenera si está vacío (ModelBase.py:151/227). Como sembramos desde el RTT,
    sin esto el preview del entrenamiento mostraría para siempre las caras del
    RTT (Keanu + famosos) en vez de las TUYAS — verificado a ojo. También limpia
    ``loss_history`` para que el gráfico sea el de TU entrenamiento.
    """
    ok = False
    for dat in model_dir.glob("*_SAEHD_data.dat"):
        try:
            with open(dat, "rb") as fh:
                data = pickle.load(fh)
            if not isinstance(data, dict):
                continue
            changed = False
            if data.get("sample_for_preview") is not None:
                data["sample_for_preview"] = None
                changed = True
            if data.get("loss_history"):
                data["loss_history"] = []
                changed = True
            if changed:
                with open(dat, "wb") as fh:
                    pickle.dump(data, fh)
                log.info("Muestras de preview del preentrenado limpiadas en %s", dat.name)
            ok = True
        except Exception as exc:  # pragma: no cover
            log.warning("No pude limpiar el preview cacheado de %s: %s", dat, exc)
    return ok


def _model_iter(model_dir: Path) -> int:
    """Iteración actual guardada en el _data.dat del modelo (0 si no se puede leer)."""
    for dat in model_dir.glob("*_SAEHD_data.dat"):
        try:
            with open(dat, "rb") as fh:
                data = pickle.load(fh)
            if isinstance(data, dict):
                return int(data.get("iter", 0) or 0)
        except Exception:
            pass
    return 0


# Al RETOMAR un modelo que ya llegó a su objetivo, cuánto más entrenar antes del
# próximo auto-export (evita el bug de matar/re-exportar en loop al retomar).
RESUME_EXTRA_ITERS = int(os.environ.get("FUSER_DFM_RESUME_EXTRA", "100000"))


def _optimizer_on_gpu(is_cuda: bool) -> bool:
    """¿El optimizador de Adam vive en la GPU o en la RAM?

    ``FUSER_DFM_OPT_GPU`` fuerza la ubicación (1=GPU, 0=RAM); sin la variable se
    usa el default histórico por motor (RAM en DirectX12, GPU en CUDA).

    Por qué existe: medido 2026-07-26 en la 4060 Ti con batch 8 — la GPU trabaja
    solo ~31% del tiempo (52% totalmente parada) y el proceso principal quema
    ~14 núcleos: el cuello es el optimizador corriendo en CPU cada iteración.
    Con VRAM libre, subirlo a la GPU apunta a ~2× de velocidad. Si no entra, DFL
    muere al construir el grafo (ANTES de entrenar, los pesos no se dañan): se
    quita la variable del lanzador y se retoma como antes.
    """
    env = os.environ.get("FUSER_DFM_OPT_GPU", "").strip().lower()
    if env in ("1", "true", "yes", "y"):
        on_gpu = True
    elif env in ("0", "false", "no", "n"):
        on_gpu = False
    else:
        return is_cuda
    if on_gpu != is_cuda:
        log.info("Optimizador del entrenamiento en %s (FUSER_DFM_OPT_GPU=%s)",
                 "GPU" if on_gpu else "RAM", env)
    return on_gpu


# Tramo FINAL del entrenamiento (iteraciones restantes) en el que se reactiva el
# refinado lr_dropout. Antes de eso corre apagado: es una opción de pulido final
# según el propio help de DFL, y en fase gruesa solo cuesta tiempo.
FINE_PHASE_ITERS = 80_000


# Fuerza del GAN en la fase de pulido. El help de DFL: "Typical fine value is
# 0.1". Más alto = más detalle fino pero más chance de artefactos.
GAN_POWER_FINE = float(os.environ.get("FUSER_DFM_GAN_POWER", "0.1"))


def _phase_opts(is_cuda: bool, remaining: int, model: Optional[Path] = None,
                sin_gan: bool = False) -> dict:
    """Opciones de entrenamiento por FASE, decididas en cada start/resume.

    Por qué existe (auditoría 2026-07-26): el modelo se siembra del preentrenado
    RTT y HEREDA sus opciones del ``_data.dat`` — Fuser no fijaba gan_power /
    true_face_power / lr_dropout en DirectX12, así que "default" no garantizaba
    nada. Costos por iteración de lo heredable:
      - gan_power>0: un 2º sess.run COMPLETO (discriminador) cada iter (~20-30%).
      - true_face_power>0: otro sess.run extra (D_train).
      - lr_dropout 'y': máscara aleatoria del tamaño de CADA matriz de pesos
        (random_uniform + tf.where) re-muestreada cada iter (~3-10%).
    Política: fase gruesa = todo apagado (lo que manda el help de DFL: GAN y
    lr_dropout son refinado final); el lr_dropout se reactiva solo en las últimas
    ``FINE_PHASE_ITERS`` del objetivo. En CUDA lr_dropout queda SIEMPRE 'n': su
    tf.where es exactamente la op que revienta el allocator (OOM Select_22).

    ``eyes_mouth_prio`` va SIEMPRE encendido (2026-07-26). Añade dos términos L1
    ×300 sobre las máscaras de ojos y boca, o sea obliga al modelo a priorizar
    esas dos regiones en vez de repartir el error por toda la cara. El propio
    help de DFL dice que arregla los "alien eyes" y la dirección de mirada
    equivocada, y que mejora el detalle de los dientes — exactamente el síntoma
    reportado ("los ojos apuntan al vacío"). Los ojos son diminutos a 224 px:
    sin esta prioridad pesan casi nada en el error y el decoder dibuja un ojo
    promedio. Cuesta dos reduce_mean por iteración (~1-3%): barato para lo que
    arregla. Viene apagado por defecto y se heredaba así del preentrenado RTT.

    ``gan_power`` (2026-07-27) se enciende SOLO en la fase de pulido, junto con
    lr_dropout, que es la condición que pide el help de DFL ("enable it only
    when the face is trained enough with lr_dropout(on)"). Es la palanca de
    poros y microtextura: fuerza al modelo a aprender el detalle fino que el
    error L1/SSIM por sí solo no premia. Dos reglas heredadas del propio DFL:
      - "**and don't disable**": una vez encendido NO se apaga. Por eso se mira
        el .dat: si ya venía con gan>0, sigue encendido aunque el objetivo se
        haya movido (retomar tras 'done' recalcula ``remaining`` y lo habría
        apagado solo).
      - Cuesta un sess.run entero más por iteración (~20-30%) y crea una red
        discriminadora CON SU PROPIO OPTIMIZADOR, que va a la misma memoria que
        el resto: con 7,6/8,2 GB de VRAM puede no entrar. ``sin_gan`` permite
        forzarlo apagado tras un OOM (ver la red de seguridad en ``start``).
    """
    fine = remaining <= FINE_PHASE_ITERS
    opts = {"true_face_power": 0.0, "eyes_mouth_prio": True}
    if is_cuda:
        opts["lr_dropout"] = "n"
    else:
        opts["lr_dropout"] = "y" if fine else "n"
    ya_activo = False
    if model is not None:
        try:
            ya_activo = float(_read_model_options(model).get("gan_power", 0) or 0) > 0
        except (TypeError, ValueError):
            ya_activo = False
    opts["gan_power"] = 0.0 if sin_gan else (GAN_POWER_FINE if (fine or ya_activo) else 0.0)
    return opts


def _vram_ocupada_mb() -> Optional[int]:
    """VRAM total en uso (MB) según nvidia-smi; None si no se puede medir."""
    try:
        r = subprocess.run(
            [r"C:\Windows\System32\nvidia-smi.exe", "--query-gpu=memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15)
        return int(r.stdout.strip().splitlines()[0])
    except Exception:
        return None


# Umbral de VRAM ocupada por encima del cual NO se lanza un entrenamiento.
# Escritorio ocioso ≈ 0,9-1,3 GB; un montaje/preview cargado ≈ 5,5 GB. 0 = off.
VRAM_GUARD_MB = int(os.environ.get("FUSER_DFM_VRAM_GUARD_MB", "2500"))


def _check_gpu_despejada() -> None:
    """Corta el arranque del entrenamiento si la GPU ya está ocupada.

    La guarda inversa ya existía (no montar con un entrenamiento vivo); esta
    dirección faltaba: retomar el entrenamiento MIENTRAS un montaje/preview
    tiene sus modelos cargados deja al entrenamiento sin VRAM → DirectML
    derrama a RAM (lento sin avisar) y bajo presión extrema es el principal
    sospechoso del colapso de pesos del 2026-07-29. Medir la VRAM real atrapa
    CUALQUIER consumidor (montaje en curso, preview, otra app), sin banderas.
    """
    if VRAM_GUARD_MB <= 0:
        return
    usada = _vram_ocupada_mb()
    if usada is not None and usada > VRAM_GUARD_MB:
        raise ValueError(
            f"La GPU ya tiene {usada} MB ocupados (umbral {VRAM_GUARD_MB}): hay un "
            f"montaje/preview en curso o modelos cargados. Entrenar así derrama a "
            f"RAM (todo lento) y arriesga los pesos. Esperá a que termine el "
            f"montaje —o Detenelo— y retomá; si es la app reteniendo memoria, "
            f"relanzala primero.")


def _espera_primera_iteracion(proc, logf: Path, timeout: int = 240) -> bool:
    """¿El entrenamiento llegó a iterar? True si sí; False si murió antes.

    DFL construye el grafo y recién después itera. Un OOM (p. ej. el GAN que no
    entra en VRAM) mata el proceso EN ESA construcción, sin escribir ninguna
    línea de iteración. Miramos el log —``[#iter][NNNms]``— porque es la única
    señal inequívoca de que está entrenando de verdad. Si el proceso sigue vivo
    al vencer el plazo lo damos por bueno: puede ser una máquina lenta cargando
    muestras, y matarlo sería peor que esperar.
    """
    patron = re.compile(r"\]\[\d+ms\]")
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            if patron.search(logf.read_text("utf-8", errors="ignore")[-8000:]):
                return True
        except OSError:
            pass
        if proc.poll() is not None:
            return False                      # murió sin iterar ni una vez
        time.sleep(3)
    return proc.poll() is None


def start(name: str, backend: Optional[str] = None) -> str:
    """Lanza (o retoma) el entrenamiento con el motor pedido (default: el activo).

    Los motores son intercambiables sobre el MISMO modelo: los pesos son .npy de
    numpy, no dependen del backend. Se puede entrenar en DirectX12 y seguir en
    CUDA (o al revés) sin perder iteraciones.
    """
    from ..core.face_library import _slug
    backend = backend if backend in BACKENDS else active_backend()
    if not backend_ready(backend):
        raise ValueError(f"El motor «{BACKENDS[backend]['label']}» no está instalado.")
    slug = _slug(name)
    ws = workspace_of(slug)
    model = ws / "model"
    # guard REAL de preparado: caras fuente alineadas + modelo sembrado
    if not (model.is_dir() and any(model.glob("*.npy"))
            and any((ws / "data_src" / "aligned").glob("*.jpg"))):
        raise ValueError("Ese modelo no está preparado. Usá 🧬 CREAR MODELO DFM primero.")
    if is_running(slug):
        return "Ya está entrenando."
    # UN entrenamiento a la vez: la GPU (8 GB) no aloja dos — el activo ocupa
    # ~7,6 GB con el optimizador en GPU (y con el optimizador en RAM eran ~5,4:
    # dos tampoco entraban). Sin esta guarda, el segundo muere construyendo el
    # grafo con un "Resource exhausted" críptico de TensorFlow (visto 2026-07-26).
    # Dos a la vez tampoco convendría: cada uno iría a <50% — mismos días totales.
    otros = [n for n in running_trainings() if _slug(n) != slug]
    if otros:
        raise ValueError(
            f"La GPU ya está entrenando a «{otros[0]}» y en esta placa solo entra "
            f"UN entrenamiento a la vez. Pausalo primero (⏸️ en su pestaña) y "
            f"después arrancá este. No perdés tiempo total: dos a la vez irían "
            f"cada uno a menos de la mitad de velocidad.")
    _check_gpu_despejada()   # y tampoco arrancar sobre un montaje/preview activo
    # objetivo con LÍNEA BASE del contador del modelo (el iter de SAEHD persiste
    # en el .dat): primer arranque → base+objetivo; retomar tras 'done' → +extra.
    cur = _model_iter(model)
    st = _read_state(slug)
    saved_target = int(st.get("target_iters") or 0)
    if saved_target and cur < saved_target:
        target = saved_target                      # sigue rumbo al objetivo original
    else:
        target = cur + (RESUME_EXTRA_ITERS if st.get("phase") == "done"
                        else AUTO_TARGET_ITERS)
    _patch_trainer_autosave(backend=backend)  # cinturón: reinstalaciones del build
    # cinturón: modelos preparados antes de existir el preview también lo activan
    # + el optimizador se ubica SEGÚN EL MOTOR: en RAM ahorra VRAM y con DirectX12
    # anda bien, pero en CUDA esa memoria es HOST ANCLADA (allocator gpu_host_bfc)
    # con cupo chico y revienta con "OOM shape[25088,512] ... device:CPU:0" en el
    # optimizador (medido). En CUDA va en la GPU.
    _is_cuda = _paths(backend)["backend"] == "cuda"
    _opt_gpu = _optimizer_on_gpu(_is_cuda)
    # autobackup_hour=1: DFL guarda una copia COMPLETA del modelo cada hora en
    # model/<nombre>_autobackups/ (01=la más nueva, rota hasta 24 ≈ 50 GB máx).
    # Se fuerza SIEMPRE tras el colapso del 2026-07-29: los pesos se rompieron
    # en una corrida, el autoguardado de 5 min cementó la rotura pisando la
    # única copia y no había NINGÚN backup — 232k iteraciones sin red. Con esto
    # un colapso cuesta a lo sumo una hora de entrenamiento.
    _opts = {"write_preview_history": True, "models_opt_on_gpu": _opt_gpu,
             "autobackup_hour": int(os.environ.get("FUSER_DFM_AUTOBACKUP_HOUR", "1"))}
    _gan_previo = float(_read_model_options(model).get("gan_power", 0) or 0)
    _opts.update(_phase_opts(_is_cuda, remaining=max(0, target - cur), model=model,
                             sin_gan=bool(st.get("gan_no_entra"))))
    # ¿Es el PRIMER arranque con GAN? Entonces hay que verificar que entre en VRAM
    # antes de darlo por bueno (el discriminador y su optimizador son memoria nueva).
    _gan_estrena = _opts.get("gan_power", 0) > 0 and _gan_previo <= 0
    _patch_model_options(model, **_opts)
    if _model_iter(model) <= 1:      # recién sembrado del RTT y aún sin entrenar
        _reset_preview_samples(model)
    # cinturón: un .dfm dentro de model/ CONGELA el resume (workspaces viejos)
    exports = ws.parent / "exports"
    for stray in model.glob("*.dfm"):
        exports.mkdir(exist_ok=True)
        os.replace(stray, exports / stray.name)
    # rotar el log: el tail viejo mostraría iters >= objetivo y confundiría al autopiloto
    logf = ws.parent / "train.log"
    if logf.exists():
        try:
            os.replace(logf, ws.parent / "train.prev.log")
        except OSError:
            pass
    def _lanzar():
        p = _run_dfl([
            "train",
            "--training-data-src-dir", str(ws / "data_src" / "aligned"),
            "--training-data-dst-dir", str(ws / "data_dst" / "aligned"),
            "--model-dir", str(model),
            "--model", "SAEHD",
            "--silent-start", "--no-preview",
        ], logf, detach=True, backend=backend)
        (ws.parent / "train.pid").write_text(str(p.pid), encoding="utf-8")
        return p

    proc = _lanzar()
    aviso = ""
    if _gan_estrena:
        # RED DE SEGURIDAD del estreno del GAN: el discriminador + su optimizador
        # son VRAM nueva sobre 7,6/8,2 GB ya ocupados. Si no entra, DFL muere
        # construyendo el grafo (los pesos quedan intactos) — y sin esto el
        # usuario se enteraría a la mañana siguiente, con la noche perdida.
        if _espera_primera_iteracion(proc, logf):
            aviso = (f"\n\n🧬 **Fase de pulido**: GAN activado (power {GAN_POWER_FINE}) "
                     f"— aprende poros y microtextura. Verificado que entra en VRAM. "
                     f"Cada iteración cuesta ~20-30% más.")
        else:
            log.warning("El GAN no entró en VRAM: relanzo sin GAN.")
            _patch_model_options(model, gan_power=0.0)
            _write_state(slug, gan_no_entra=True)
            proc = _lanzar()
            aviso = ("\n\n⚠️ **El GAN no entra en la VRAM de esta placa** (el "
                     "discriminador necesita memoria que no hay con 8 GB). Relancé "
                     "sin GAN para no perder la sesión; el resto del pulido "
                     "(lr_dropout) sigue activo. Queda anotado para no reintentarlo.")
    (ws.parent / "train.pid").write_text(str(proc.pid), encoding="utf-8")
    _write_state(slug, phase="training", name=name, target_iters=target,
                 backend=backend, started_at=time.strftime("%Y-%m-%d %H:%M:%S"))
    return (f"🏋️ Entrenamiento lanzado con «{BACKENDS[backend]['label']}» (pid {proc.pid}, "
            f"iter actual {cur:,} → objetivo {target:,}). Corre en segundo plano aunque "
            f"cierres Fuser.{aviso}")


def _pid_of(slug: str) -> Optional[int]:
    try:
        return int((workspace_of(slug).parent / "train.pid").read_text().strip())
    except Exception:
        return None


def _clear_pid(slug: str) -> None:
    try:
        (workspace_of(slug).parent / "train.pid").unlink()
    except OSError:
        pass


def is_running(slug: str) -> bool:
    """¿El PID guardado sigue siendo NUESTRO entrenamiento?

    Windows reusa PIDs (garantizado tras un reinicio): validamos que el proceso
    sea python Y que su línea de comandos contenga main.py train — si no, el pid
    es rancio y se limpia (evita que taskkill mate un proceso ajeno).
    """
    pid = _pid_of(slug)
    if not pid:
        return False
    try:
        r = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             f"(Get-CimInstance Win32_Process -Filter 'ProcessId={pid}').CommandLine"],
            capture_output=True, text=True, timeout=20)
        cmdline = (r.stdout or "").strip().lower()
        alive = "main.py" in cmdline and "train" in cmdline and "python" in cmdline
        if not alive and cmdline:
            _clear_pid(slug)  # PID reusado por otro proceso: pid rancio
        elif not cmdline:
            _clear_pid(slug)  # proceso muerto
        return alive
    except Exception:
        return False


def progress_info(name: str) -> dict:
    from ..core.face_library import _slug
    slug = _slug(name)
    st = _read_state(slug)
    info = {"phase": st.get("phase", "—"), "running": is_running(slug),
            "iter": None, "ms": None, "loss_src": None, "loss_dst": None, "tail": ""}
    logf = workspace_of(slug).parent / "train.log"
    if logf.is_file():
        try:
            tail = logf.read_bytes()[-6000:].decode("utf-8", "replace")
            info["tail"] = "\n".join(tail.splitlines()[-8:])
            for m in PROGRESS_RE.finditer(tail):
                info["iter"] = int(m.group(1)); info["ms"] = int(m.group(2))
                info["loss_src"] = float(m.group(3)); info["loss_dst"] = float(m.group(4))
        except Exception:
            pass
    return info


def preview_images(name: str, limit: int = 6) -> List[tuple]:
    """Previews del entrenamiento (del más viejo al más nuevo) como (ruta, etiqueta).

    DFL escribe una imagen por autoguardado en ``<modelo>_history/<preview>/``:
    cada una es una grilla [tu cara | reconstruida | destino | reconstruido |
    TU CARA EN EL DESTINO]. Ver varias en orden muestra si el modelo MEJORA.
    Devuelve [] si todavía no hay ninguna (los primeros minutos).
    """
    from ..core.face_library import _slug
    model = workspace_of(_slug(name)) / "model"
    if not model.is_dir():
        return []
    hist_dir = None
    for d in sorted(model.glob("*_history")):
        for sub in sorted(p for p in d.iterdir() if p.is_dir()):
            # preferir el preview SIN máscara (se lee mucho mejor)
            if "mask" not in sub.name.lower():
                hist_dir = sub
                break
        if hist_dir is None and d.is_dir():
            subs = [p for p in d.iterdir() if p.is_dir()]
            hist_dir = subs[0] if subs else None
        if hist_dir:
            break
    if hist_dir is None:
        return []
    shots = []
    for f in hist_dir.glob("*.jpg"):
        if f.stem.startswith("_"):
            continue                       # _last.jpg duplica la última numerada
        try:
            shots.append((int(f.stem), f))
        except ValueError:
            continue
    shots.sort()
    if not shots:
        last = hist_dir / "_last.jpg"
        return [(str(last), "último")] if last.is_file() else []
    return [(str(f), f"iter {it:,}") for it, f in shots[-max(1, limit):]]


def _stop_slug(slug: str) -> str:
    pid = _pid_of(slug)
    if not pid or not is_running(slug):
        _clear_pid(slug)
        return "No hay entrenamiento corriendo."
    subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True, timeout=30)
    _clear_pid(slug)
    _write_state(slug, phase="stopped")
    return ("⏸️ Entrenamiento detenido. DFL autoguarda cada ~15 min: como mucho se pierde ese tramo. "
            "Podés retomarlo cuando quieras (mismo botón de entrenar).")


def stop(name: str) -> str:
    from ..core.face_library import _slug
    return _stop_slug(_slug(name))


def _live_train_pids() -> set:
    """PIDs de entrenamientos DFL vivos, en UNA sola consulta a Windows.

    ``is_running`` lanza un PowerShell por workspace: con varias personas en la
    biblioteca eso son segundos de espera. Para recorrer todos (botón de cierre,
    panel de estado) conviene un único barrido de procesos.
    """
    try:
        r = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "Get-CimInstance Win32_Process | Where-Object { $_.Name -like '*python*' -and "
             "$_.CommandLine -like '*main.py*' -and $_.CommandLine -like '*train*' } | "
             "ForEach-Object { $_.ProcessId }"],
            capture_output=True, text=True, timeout=30)
        return {int(x) for x in (r.stdout or "").split() if x.strip().isdigit()}
    except Exception as exc:
        log.warning("No pude listar los procesos de entrenamiento: %s", exc)
        return set()


def _workspaces_with_pid() -> List[tuple]:
    """(slug, nombre, pid) de cada workspace con un ``train.pid`` escrito."""
    wsr = _paths()["workspaces"]
    if not wsr.is_dir():
        return []
    out = []
    for d in sorted(wsr.iterdir()):
        if not d.is_dir():
            continue
        pid = _pid_of(d.name)
        if pid:
            out.append((d.name, _read_state(d.name).get("name") or d.name, pid))
    return out


def running_trainings() -> List[str]:
    """Nombres de las personas con un entrenamiento DFL vivo AHORA MISMO.

    Lee los workspaces en disco (no la memoria del proceso) porque los
    entrenamientos sobreviven al cierre de la app: al reabrir Fuser hay que
    poder verlos igual.
    """
    live = _live_train_pids()
    return [name for _slug_, name, pid in _workspaces_with_pid() if pid in live]


def stop_all() -> List[str]:
    """Pausa TODOS los entrenamientos DFL vivos. Devuelve los que paró.

    Los procesos de DeepFaceLab son subprocesos desacoplados: matar Fuser no los
    toca. Para que "cerrar la app" signifique de verdad "no queda nada corriendo"
    hay que pararlos uno a uno antes de salir.
    """
    live = _live_train_pids()
    stopped = []
    for slug, name, pid in _workspaces_with_pid():
        if pid not in live:
            continue
        try:
            _stop_slug(slug)
            stopped.append(name)
            log.info("Entrenamiento pausado al cerrar Fuser: %s", name)
        except Exception as exc:
            log.warning("No pude pausar el entrenamiento de %s: %s", slug, exc)
    return stopped


def test_backend(backend: str, slug: str = "q_stock", minutes: int = 8,
                 progress: Optional[Callable] = None) -> dict:
    """Prueba si un motor ENTRENA de verdad en esta GPU (o se cuelga).

    Corre un entrenamiento corto sobre un workspace YA preparado (por defecto el
    de stock) y mide si el contador de iteraciones avanza. El build CUDA es de
    2021 y esta placa es Ada: el modo de falla conocido es quedarse colgado tras
    cargar los datos, sin consumir GPU — por eso el criterio es "avanzó o no".

    Devuelve {ok, iters, ms_iter, detail}. NO toca el entrenamiento del usuario:
    usá un slug de prueba distinto al suyo.
    """
    if not backend_ready(backend):
        return {"ok": False, "iters": 0, "ms_iter": None,
                "detail": f"El motor «{BACKENDS[backend]['label']}» no está instalado."}
    ws = workspace_of(slug)
    if not (ws / "model").is_dir() or not any((ws / "data_src" / "aligned").glob("*.jpg")):
        return {"ok": False, "iters": 0, "ms_iter": None,
                "detail": f"No hay workspace de prueba preparado en «{slug}»."}
    if is_running(slug):
        return {"ok": False, "iters": 0, "ms_iter": None,
                "detail": "Ese workspace ya está entrenando."}

    base = _model_iter(ws / "model")
    msg = start(slug, backend=backend)
    log.info("test_backend(%s): %s", backend, msg)
    deadline = time.time() + minutes * 60
    last = {"iter": None, "ms": None}
    try:
        while time.time() < deadline:
            time.sleep(20)
            info = progress_info(slug)
            last = {"iter": info["iter"], "ms": info["ms"]}
            if progress:
                el = int(time.time() - (deadline - minutes * 60))
                progress(min(0.95, el / (minutes * 60)),
                         f"Probando {BACKENDS[backend]['label']}: iter={info['iter']}…")
            if info["iter"] and info["iter"] > 10:
                break                     # avanzó de verdad: alcanza para saber
            if not info["running"]:
                break                     # murió (crash a la vista en el log)
    finally:
        stop(slug)
    ok = bool(last["iter"] and last["iter"] > 10)
    detail = ("Entrena correctamente." if ok else
              "NO avanzó (el build se cuelga o falla en esta GPU). "
              f"Mirá {ws.parent / 'train.log'}.")
    return {"ok": ok, "iters": last["iter"] or 0, "ms_iter": last["ms"],
            "detail": detail, "base_iter": base}


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------
def export(name: str, timeout: int = 1800) -> Path:
    from ..core.face_library import _slug
    slug = _slug(name)
    ws = workspace_of(slug)
    model = ws / "model"
    if is_running(slug):
        raise ValueError("Pará el entrenamiento antes de exportar.")
    # exportdfm corre en CPU: sirve cualquier motor INSTALADO (preferimos el que
    # entrenó este modelo; si no está, el activo o el primero disponible).
    be = _read_state(slug).get("backend")
    # OJO: exigir be ∈ BACKENDS ademas de ready. backend_ready("") devuelve True
    # (resuelve las rutas del motor default), asi que un state viejo SIN campo
    # backend pasaba el guard con be=None y reventaba mas abajo con el error
    # engañoso "no hay ningún motor instalado" (medido 2026-07-26, prueba_stock).
    if be not in BACKENDS or not backend_ready(be):
        be = active_backend() if backend_ready(active_backend()) else \
            next((b for b in BACKENDS if backend_ready(b)), None)
    if be is None:
        raise ValueError("No hay ningún motor de entrenamiento instalado.")
    with _EXPORT_LOCK:  # nunca dos exportdfm sobre el mismo model-dir
        logf = ws.parent / "export.log"
        t0 = time.time()
        proc = _run_dfl(["exportdfm", "--model-dir", str(model), "--model", "SAEHD"],
                        logf, stdin_text="\n" * 4, backend=be)
        try:
            rc = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            raise RuntimeError(f"exportdfm superó {timeout}s; proceso matado. Log: {logf}")
        # exigir un .dfm NUEVO (mtime >= inicio): sin esto, un export fallido
        # devolvería el .dfm viejo de una corrida anterior como si fuera fresco.
        fresh = [f for f in model.glob("*.dfm") if f.stat().st_mtime >= t0 - 5]
        if not fresh:
            raise RuntimeError(f"El export no generó un .dfm nuevo (rc={rc}). Log: {logf}")
        newest = max(fresh, key=lambda f: f.stat().st_mtime)
        # SACAR el .dfm de model/: un .dfm dentro del dir del modelo CONGELA el
        # próximo resume de DFL en esta GPU (medido: colgado silencioso post
        # "Sort by yaw"; sin el archivo, entrena normal).
        exports = ws.parent / "exports"
        exports.mkdir(exist_ok=True)
        dst = exports / newest.name
        os.replace(newest, dst)
        for stray in model.glob("*.dfm"):
            os.replace(stray, exports / stray.name)
        return dst


# ---------------------------------------------------------------------------
# Autopiloto: exporta y registra SOLO al llegar al objetivo de iteraciones
# ---------------------------------------------------------------------------
_AUTOPILOT = {"started": False}


def autopilot_scan() -> List[str]:
    """Revisa los entrenamientos: al llegar al objetivo corta, exporta y registra.

    Decisiones anti-bug (revisión adversarial):
    - El avance se mide con el ITER DEL MODELO (.dat), no con el tail del log
      (el log rota en start(), pero el .dat es la verdad).
    - Actúa aunque el proceso ya no corra (export fallido previo, reinicio de la
      máquina con el objetivo alcanzado): sin exigir running=True no hay deadlock.
    - phase='exporting' como guard de reentrada; 'export_failed' se reintenta.
    """
    acts: List[str] = []
    wsr = _paths()["workspaces"]
    if not wsr.is_dir():
        return acts
    for d in wsr.iterdir():
        try:
            st = json.loads((d / "state.json").read_text(encoding="utf-8"))
        except Exception:
            continue
        if st.get("phase") not in ("training", "export_failed"):
            continue
        slug = d.name
        name = st.get("name", slug)
        target = int(st.get("target_iters", AUTO_TARGET_ITERS))
        cur = _model_iter(workspace_of(slug) / "model")
        if cur < target:
            continue
        _write_state(slug, phase="exporting")
        try:
            if is_running(slug):
                stop(slug)
                time.sleep(5)  # que DFL suelte los archivos del modelo
            dfm = export(slug)
            from .face_library import set_dfm
            msg = set_dfm(name, str(dfm))
            _write_state(slug, phase="done", exported=str(dfm))
            log.info("Autopiloto: %s → %s", name, msg)
            acts.append(f"{name}: {msg}")
        except Exception as exc:  # se reintenta en el próximo tick (sin exigir running)
            log.warning("Autopiloto falló en %s: %s", name, exc)
            _write_state(slug, phase="export_failed", last_error=str(exc)[:200])
    return acts


def start_autopilot(interval: int = 600) -> None:
    """Hilo daemon: cada ``interval`` s corre autopilot_scan. Idempotente."""
    if _AUTOPILOT["started"]:
        return
    _AUTOPILOT["started"] = True
    import threading

    def _loop():
        while True:
            try:
                autopilot_scan()
            except Exception:  # pragma: no cover
                pass
            time.sleep(interval)

    threading.Thread(target=_loop, daemon=True, name="dfm-autopilot").start()
    log.info("Autopiloto de entrenamiento .dfm activo (objetivo %s iters).", f"{AUTO_TARGET_ITERS:,}")
