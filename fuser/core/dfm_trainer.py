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
def _paths() -> dict:
    root = dfl_root()
    build = root / "build"
    # el autoextraíble crea una carpeta DeepFaceLab_DirectX12 adentro
    inner = next((p for p in build.glob("DeepFaceLab*") if p.is_dir()), build)
    internal = inner / "_internal"
    return {
        "root": root,
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


def _read_state(slug: str) -> dict:
    try:
        return json.loads(_state_file(slug).read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_state(slug: str, **kw) -> None:
    st = _read_state(slug)
    st.update(kw)
    _state_file(slug).parent.mkdir(parents=True, exist_ok=True)
    _state_file(slug).write_text(json.dumps(st, ensure_ascii=False, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Estado global del entrenador
# ---------------------------------------------------------------------------
def status() -> dict:
    p = _paths()
    py = _find_python(p)
    rtt_ok = any(p["rtt"].glob("*_SAEHD_*.npy")) or any(p["rtt"].glob("*.npy")) if p["rtt"].is_dir() else False
    return {
        "root": str(p["root"]),
        "build_ready": bool(py and p["main"].parent.is_dir() and _find_main(p)),
        "rtt_ready": rtt_ok,
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
def _download(url: str, dst: Path, progress: Optional[Callable] = None, label: str = "") -> Path:
    """Descarga en streaming con reanudación (.part + Range)."""
    import urllib.request

    dst.parent.mkdir(parents=True, exist_ok=True)
    part = dst.with_suffix(dst.suffix + ".part")
    have = part.stat().st_size if part.exists() else 0
    req = urllib.request.Request(url, headers={"User-Agent": "fuser-dfm-trainer"})
    if have:
        req.add_header("Range", f"bytes={have}-")
    mode = "ab" if have else "wb"
    with urllib.request.urlopen(req, timeout=60) as r:
        total = have + int(r.headers.get("Content-Length") or 0)
        done = have
        with open(part, mode) as fh:
            while True:
                chunk = r.read(1024 * 512)
                if not chunk:
                    break
                fh.write(chunk)
                done += len(chunk)
                if progress and total:
                    progress(done / total, f"{label} {done // 1048576}/{total // 1048576} MB")
    part.rename(dst)
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


def install(progress: Optional[Callable] = None) -> str:
    """Instala el entrenador local: build DX12 + preentrenado RTT (una vez)."""
    p = _paths()
    msgs = []
    if not status()["build_ready"]:
        exe = p["downloads"] / Path(BUILD_URL).name
        _fetch_verified(BUILD_URL, exe, BUILD_SHA256, progress, "Build DeepFaceLab DX12:")
        if progress:
            progress(0.5, "Desempaquetando el build (~2.6 GB)…")
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
        msgs.append("build instalado")
    if not status()["rtt_ready"]:
        z = p["downloads"] / "RTT_model_224_V2.zip"
        _fetch_verified(RTT_URL, z, RTT_SHA256, progress, "Preentrenado RTT 224:")
        if progress:
            progress(0.9, "Desempaquetando el preentrenado…")
        p["rtt"].mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(z) as zf:
            zf.extractall(p["rtt"])
        msgs.append("preentrenado RTT listo")
    st = status()
    if not st["build_ready"]:
        raise RuntimeError("El build quedó incompleto: no encuentro python embebido/main.py.")
    return "✅ Entrenador local instalado (" + ", ".join(msgs or ["ya estaba"]) + f") en {st['root']}"


# ---------------------------------------------------------------------------
# Subprocesos DFL
# ---------------------------------------------------------------------------
def _run_dfl(args: List[str], log_file: Path, cwd: Optional[Path] = None,
             detach: bool = False, stdin_text: Optional[str] = None):
    p = _paths()
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
    env["PATH"] = os.pathsep.join([
        str(py.parent), str(py.parent / "Scripts"),
        str(internal / "ffmpeg"), env.get("PATH", ""),
    ])
    env.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    cmd = [str(py), str(main)] + args
    log_file.parent.mkdir(parents=True, exist_ok=True)
    lf = open(log_file, "ab")
    kw = dict(cwd=str(cwd or main.parent), env=env, stdout=lf, stderr=subprocess.STDOUT)
    if detach:
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
        rc = proc.wait(timeout=timeout)
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
            pak = next(iter(rtm_dir.glob("faceset.pak")), None)
            if pak is not None:
                shutil.copyfile(pak, aligned / "faceset.pak")

    # ---- 2) SRC: con pocas fotos reales, SINTETIZAR el faceset ------------------
    from . import faceset_synth
    synth_info = None
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
    _patch_model_options(model, pretrain=False, batch_size=4, models_opt_on_gpu=False)

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
            f"modelo sembrado del RTT (pretrain OFF, batch 4). Ya podés entrenar.")


# ---------------------------------------------------------------------------
# Entrenamiento
# ---------------------------------------------------------------------------
def start(name: str) -> str:
    from ..core.face_library import _slug
    slug = _slug(name)
    ws = workspace_of(slug)
    if not (ws / "model").is_dir():
        raise ValueError("Ese modelo no está preparado (corré el paso ③).")
    if is_running(slug):
        return "Ya está entrenando."
    logf = ws.parent / "train.log"
    proc = _run_dfl([
        "train",
        "--training-data-src-dir", str(ws / "data_src" / "aligned"),
        "--training-data-dst-dir", str(ws / "data_dst" / "aligned"),
        "--model-dir", str(ws / "model"),
        "--model", "SAEHD",
        "--silent-start", "--no-preview",
    ], logf, detach=True)
    (ws.parent / "train.pid").write_text(str(proc.pid), encoding="utf-8")
    _write_state(slug, phase="training", started_at=time.strftime("%Y-%m-%d %H:%M:%S"))
    return f"🏋️ Entrenamiento lanzado (pid {proc.pid}). Corre en segundo plano aunque cierres Fuser."


def _pid_of(slug: str) -> Optional[int]:
    try:
        return int((workspace_of(slug).parent / "train.pid").read_text().strip())
    except Exception:
        return None


def is_running(slug: str) -> bool:
    pid = _pid_of(slug)
    if not pid:
        return False
    try:
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}"], capture_output=True,
                             text=True, timeout=15).stdout
        return str(pid) in out
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


def stop(name: str) -> str:
    from ..core.face_library import _slug
    slug = _slug(name)
    pid = _pid_of(slug)
    if not pid or not is_running(slug):
        return "No hay entrenamiento corriendo."
    subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True, timeout=30)
    _write_state(slug, phase="stopped")
    return ("⏸️ Entrenamiento detenido. DFL autoguarda cada ~15 min: como mucho se pierde ese tramo. "
            "Podés retomarlo cuando quieras (mismo botón de entrenar).")


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
    logf = ws.parent / "export.log"
    proc = _run_dfl(["exportdfm", "--model-dir", str(model), "--model", "SAEHD"],
                    logf, stdin_text="\n" * 4)
    rc = proc.wait(timeout=timeout)
    dfms = sorted(model.glob("*.dfm"), key=lambda f: f.stat().st_mtime, reverse=True)
    if not dfms:
        raise RuntimeError(f"El export no generó un .dfm (rc={rc}). Log: {logf}")
    return dfms[0]
