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


# URL del build DirectX12 y del preentrenado RTT. Se rellenan/ajustan según la
# verificación (ver research); si cambian de mirror, actualizá SOLO estas líneas.
BUILD_URL = os.environ.get(
    "FUSER_DFL_BUILD_URL",
    "https://github.com/iperov/DeepFaceLab/releases/download/DeepFaceLab_DirectX12_build_05_04_2022/DeepFaceLab_DirectX12_build_05_04_2022.exe",
)
RTT_URL = os.environ.get(
    "FUSER_DFL_RTT_URL",
    "https://huggingface.co/datasets/dimanchkek/Deepfacelive-DFM-Models/resolve/main/Pretrained/RTT%20model%20224%20V2.zip",
)

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


def install(progress: Optional[Callable] = None) -> str:
    """Instala el entrenador local: build DX12 + preentrenado RTT (una vez)."""
    p = _paths()
    msgs = []
    if not status()["build_ready"]:
        exe = p["downloads"] / Path(BUILD_URL).name
        if not exe.exists():
            _download(BUILD_URL, exe, progress, "Build DeepFaceLab DX12:")
        if progress:
            progress(0.5, "Desempaquetando el build…")
        p["build"].mkdir(parents=True, exist_ok=True)
        if exe.suffix.lower() == ".zip":
            with zipfile.ZipFile(exe) as z:
                z.extractall(p["build"])
        else:
            # autoextraíble 7z SFX: -y silencioso, -o<dir> destino
            r = subprocess.run([str(exe), "-y", f"-o{p['build']}"], capture_output=True, timeout=1800)
            if r.returncode != 0:
                raise RuntimeError(f"No pude desempaquetar el build (rc={r.returncode}).")
        msgs.append("build instalado")
    if not status()["rtt_ready"]:
        z = p["downloads"] / "RTT_model_224_V2.zip"
        if not z.exists():
            _download(RTT_URL, z, progress, "Preentrenado RTT 224:")
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
    internal = str(main.parent.parent)
    # réplica del setenv.bat del build: rutas del python embebido
    env["PYTHONPATH"] = str(main.parent)
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
def prepare(name: str, src_dir: Path, dst_videos: List[str],
            progress: Optional[Callable] = None) -> str:
    """Arma el workspace de entrenamiento para una Cara.

    - SRC: fotos curadas (carpeta de la pestaña ① / faceset_<slug>).
    - DST: frames de TUS videos (a los que después vas a montar la cara) —
      100% local, sin descargas. Extrae caras de ambos con el detector de DFL.
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
    if not dst_videos:
        raise ValueError("Agregá al menos un video destino (los videos donde vas a montar la cara).")
    st = status()
    if not st["build_ready"]:
        raise ValueError("El entrenador no está instalado. Corré el paso ② (instalar).")

    ws = workspace_of(slug)
    data_src = ws / "data_src"
    data_dst = ws / "data_dst"
    model = ws / "model"
    for d in (data_src, data_dst, model):
        d.mkdir(parents=True, exist_ok=True)

    # 1) SRC: copiar fotos curadas
    if progress:
        progress(0.05, "Copiando fotos fuente…")
    n = 0
    for i, f in enumerate(sorted(src_dir.iterdir())):
        if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp", ".bmp"):
            shutil.copyfile(f, data_src / f"{i:05d}{f.suffix.lower()}")
            n += 1

    # 2) DST: frames de los videos del usuario (cap total ~1500 frames)
    if progress:
        progress(0.1, "Extrayendo frames de tus videos…")
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
                    cv2.imwrite(str(data_dst / f"v{vi:02d}_{fi:05d}.jpg"), fr,
                                [cv2.IMWRITE_JPEG_QUALITY, 92])
                    total_frames += 1
        except Exception as exc:
            log.warning("No pude extraer frames de %s: %s", v, exc)
    if total_frames == 0:
        raise ValueError("No pude extraer frames de los videos destino.")

    # 3) Extraer caras (WF) de SRC y DST con el detector S3FD de DFL
    logf = ws.parent / "prepare.log"
    for phase_name, in_dir, frac in (("fuente", data_src, 0.3), ("destino", data_dst, 0.6)):
        if progress:
            progress(frac, f"Detectando caras ({phase_name}) — puede tardar varios minutos…")
        out_aligned = in_dir / "aligned"
        out_aligned.mkdir(exist_ok=True)
        proc = _run_dfl([
            "extract", "--input-dir", str(in_dir), "--output-dir", str(out_aligned),
            "--detector", "s3fd", "--face-type", "whole_face",
            "--max-faces-from-image", "1", "--image-size", "512", "--jpeg-quality", "90",
            "--no-output-debug",
        ], logf, stdin_text="\n" * 8)
        rc = proc.wait(timeout=7200)
        n_faces = len(list(out_aligned.glob("*.jpg")))
        if rc != 0 or n_faces == 0:
            raise RuntimeError(
                f"La extracción de caras ({phase_name}) falló (rc={rc}, caras={n_faces}). "
                f"Mirá el log: {logf}")

    # 4) Semilla del modelo: RTT (warm-start) + opciones seguras para 8GB DX12
    if progress:
        progress(0.9, "Sembrando el preentrenado RTT…")
    p = _paths()
    seeded = 0
    for f in p["rtt"].rglob("*"):
        if f.is_file() and f.suffix in (".npy", ".dat"):
            shutil.copyfile(f, model / f.name)
            seeded += 1
    if seeded == 0:
        raise RuntimeError("El preentrenado RTT no está instalado (paso ②).")
    _patch_model_options(model, pretrain=False, batch_size=4, models_opt_on_gpu=False)

    _write_state(slug, phase="prepared", src_faces=len(list((data_src / 'aligned').glob('*.jpg'))),
                 dst_faces=len(list((data_dst / 'aligned').glob('*.jpg'))),
                 prepared_at=time.strftime("%Y-%m-%d %H:%M:%S"))
    stx = _read_state(slug)
    return (f"✅ Workspace listo: {stx['src_faces']} caras fuente · {stx['dst_faces']} caras destino · "
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
