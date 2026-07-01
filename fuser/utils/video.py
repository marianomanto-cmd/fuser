"""Entrada/salida de vídeo.

Estrategia:
- Lectura frame a frame con OpenCV (rápido, sin cargar todo el vídeo en memoria).
- Escritura mediante un *pipe* directo a FFmpeg en formato rawvideo: permite
  codificar a H.264 con CRF configurable y calidad alta sin depender de los
  códecs de OpenCV. Si FFmpeg no está, hay un fallback a ``cv2.VideoWriter``.
- El audio original se vuelve a multiplexar al final con FFmpeg.

El binario de FFmpeg se obtiene del sistema o, si no existe, del que empaqueta
``imageio-ffmpeg`` (cero instalación manual -> "plug and play").
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional

import cv2
import numpy as np

from .logging import get_logger
from .system import ffmpeg_path

log = get_logger(__name__)


@dataclass
class VideoInfo:
    path: str
    fps: float
    width: int
    height: int
    frame_count: int
    has_audio: bool

    @property
    def duration(self) -> float:
        return self.frame_count / self.fps if self.fps else 0.0


def probe(path: str) -> VideoInfo:
    """Obtiene metadatos del vídeo (fps, tamaño, nº de frames, audio)."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir el vídeo: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if frame_count <= 0:
        frame_count = _count_frames_slow(path)
    return VideoInfo(
        path=str(path),
        fps=float(fps),
        width=width,
        height=height,
        frame_count=frame_count,
        has_audio=_has_audio(path),
    )


def _count_frames_slow(path: str) -> int:
    cap = cv2.VideoCapture(str(path))
    count = 0
    while cap.grab():
        count += 1
    cap.release()
    return count


def _has_audio(path: str) -> bool:
    ff = ffmpeg_path()
    if not ff:
        return False
    try:
        proc = subprocess.run(
            [ff, "-hide_banner", "-i", str(path)],
            capture_output=True,
            text=True,
            timeout=30,
        )
        return "Audio:" in (proc.stderr or "")
    except Exception:
        return False


def read_frames(path: str, start: int = 0, count: Optional[int] = None) -> Iterator[np.ndarray]:
    """Generador de frames BGR. No carga todo el vídeo en memoria."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir el vídeo: {path}")
    if start > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    emitted = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield frame
            emitted += 1
            if count is not None and emitted >= count:
                break
    finally:
        cap.release()


def get_frames_at(path: str, indices: List[int]) -> List[np.ndarray]:
    """Extrae frames concretos por índice (para previsualizaciones)."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir el vídeo: {path}")
    frames = []
    try:
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if ok:
                frames.append(frame)
    finally:
        cap.release()
    return frames


def keyframe_indices(frame_count: int, n: int) -> List[int]:
    """Índices equiespaciados para muestrear ``n`` frames clave."""
    if frame_count <= 0:
        return []
    n = max(1, min(n, frame_count))
    if n == 1:
        return [frame_count // 2]
    step = (frame_count - 1) / (n - 1)
    return [int(round(i * step)) for i in range(n)]


class FFmpegVideoWriter:
    """Escribe frames BGR a un vídeo H.264 mediante un pipe a FFmpeg."""

    def __init__(
        self,
        path: str,
        width: int,
        height: int,
        fps: float,
        crf: int = 18,
        encoder: str = "libx264",
    ):
        self.path = str(path)
        self.width = width
        self.height = height
        ff = ffmpeg_path()
        if not ff:
            raise RuntimeError("FFmpeg no disponible para escribir el vídeo.")
        cmd = [
            ff, "-y", "-hide_banner", "-loglevel", "error",
            "-f", "rawvideo", "-vcodec", "rawvideo",
            "-s", f"{width}x{height}", "-pix_fmt", "bgr24", "-r", f"{fps}",
            "-i", "-",
            "-an",
            "-vcodec", encoder, "-crf", str(crf),
            "-pix_fmt", "yuv420p",
            # Garantiza dimensiones pares (requisito de yuv420p).
            "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            self.path,
        ]
        self.proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
        )

    def write(self, frame: np.ndarray) -> None:
        if frame.shape[1] != self.width or frame.shape[0] != self.height:
            frame = cv2.resize(frame, (self.width, self.height))
        try:
            self.proc.stdin.write(np.ascontiguousarray(frame).tobytes())
        except BrokenPipeError as exc:  # pragma: no cover
            err = self.proc.stderr.read().decode("utf-8", "ignore") if self.proc.stderr else ""
            raise RuntimeError(f"FFmpeg falló al escribir: {err}") from exc

    def close(self) -> None:
        if self.proc.stdin:
            self.proc.stdin.close()
        self.proc.wait()


def last_frame(path: str) -> np.ndarray:
    """Devuelve el ÚLTIMO frame (BGR) de un vídeo, de forma robusta.

    Usado por la función *Extender vídeo*: el último frame de un clip se usa como
    imagen de arranque del siguiente. Busca unos frames antes del final y lee
    hasta el último real (``CAP_PROP_FRAME_COUNT`` puede quedarse corto).
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir el vídeo: {path}")
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total > 5:
            cap.set(cv2.CAP_PROP_POS_FRAMES, total - 5)
        last = None
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            last = frame
        if last is None:  # seek falló: releer desde el principio
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                last = frame
        if last is None:
            raise RuntimeError(f"El vídeo no tiene frames legibles: {path}")
        return last
    finally:
        cap.release()


def save_last_frame(path: str, out_image: str) -> str:
    """Extrae el último frame y lo guarda como imagen. Devuelve la ruta."""
    frame = last_frame(path)
    Path(out_image).parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out_image), frame):
        raise RuntimeError(f"No se pudo escribir el frame en {out_image}")
    return str(out_image)


def concat_videos(paths: List[str], output: str, *, drop_seam: bool = True,
                  crf: int = 18, encoder: str = "libx264") -> bool:
    """Concatena varios clips en uno solo (para *Extender/Encadenar vídeo*).

    ``drop_seam``: en un encadenado, el frame 0 de cada clip a partir del 2º es la
    imagen de arranque = el ÚLTIMO frame del clip anterior (duplicado). Con
    ``drop_seam`` se descarta ese frame repetido para que la unión no "congele" un
    frame. Reescala todos los clips al tamaño del primero (defensivo) y reencoda a
    H.264. Devuelve True si tuvo éxito. No mezcla audio (video-only): el audio del
    resultado se genera aparte sobre el clip final.
    """
    paths = [str(p) for p in paths if p]
    if not paths:
        return False
    if len(paths) == 1:
        # Nada que concatenar: copia tal cual.
        try:
            import shutil
            shutil.copyfile(paths[0], str(output))
            return True
        except Exception as exc:  # pragma: no cover
            log.warning("No se pudo copiar el clip único: %s", exc)
            return False
    ff = ffmpeg_path()
    if not ff:
        log.warning("FFmpeg no disponible: no se puede concatenar.")
        return False
    info0 = probe(paths[0])
    w = info0.width - (info0.width % 2)
    h = info0.height - (info0.height % 2)
    if w <= 0 or h <= 0:
        log.warning("concat: dimensiones inválidas del primer clip (%sx%s).",
                    info0.width, info0.height)
        return False
    labels, filters = [], []
    for i, path in enumerate(paths):
        # Recorta el frame de enlace SOLO si el clip tiene ≥2 frames: si tuviera 1
        # (render truncado), trim=start_frame=1 lo dejaría vacío y desaparecería sin
        # aviso. Escala con letterbox (decrease+pad), no estirando, por si algún clip
        # llegara con otro aspecto (los callers normales ya vienen todos iguales).
        do_trim = drop_seam and i > 0
        if do_trim:
            try:
                if probe(path).frame_count < 2:
                    do_trim = False
            except Exception:
                do_trim = False
        trim = "trim=start_frame=1," if do_trim else ""
        filters.append(
            f"[{i}:v]{trim}scale={w}:{h}:force_original_aspect_ratio=decrease,"
            f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,setsar=1,setpts=PTS-STARTPTS[v{i}]"
        )
        labels.append(f"[v{i}]")
    filter_complex = ";".join(filters) + ";" + "".join(labels) + \
        f"concat=n={len(paths)}:v=1:a=0[out]"
    cmd = [ff, "-y", "-hide_banner", "-loglevel", "error"]
    for p in paths:
        cmd += ["-i", p]
    cmd += ["-filter_complex", filter_complex, "-map", "[out]",
            "-c:v", encoder, "-crf", str(crf), "-pix_fmt", "yuv420p", str(output)]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as exc:  # pragma: no cover
        err = (exc.stderr or b"").decode("utf-8", "ignore")
        log.warning("Falló la concatenación de vídeo: %s", err[:500])
        return False
    except Exception as exc:  # pragma: no cover
        log.warning("Falló la concatenación de vídeo: %s", exc)
        return False


def mux_external_audio(video: str, audio: str, output: str) -> bool:
    """Mezcla una pista de audio EXTERNA (wav/flac/mp3) sobre un vídeo.

    Usado por la función *Imagen → Vídeo*: Wan genera el vídeo sin sonido y el
    audio se genera por separado. ``-shortest`` recorta al más corto de los dos
    (normalmente ya vienen igualados en duración). Devuelve True si tuvo éxito.
    """
    ff = ffmpeg_path()
    if not ff:
        return False
    cmd = [
        ff, "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(video), "-i", str(audio),
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
        "-shortest", str(output),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except Exception as exc:  # pragma: no cover
        log.warning("No se pudo mezclar el audio externo: %s", exc)
        return False


def mux_audio(video_no_audio: str, original: str, output: str) -> bool:
    """Copia la pista de audio de ``original`` al vídeo procesado.

    Devuelve True si se incrustó audio. El sufijo ``?`` en el mapeo hace que el
    audio sea opcional: si el original no tenía audio, no falla.
    """
    ff = ffmpeg_path()
    if not ff:
        return False
    cmd = [
        ff, "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(video_no_audio), "-i", str(original),
        "-map", "0:v:0", "-map", "1:a:0?",
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
        "-shortest", str(output),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except Exception as exc:  # pragma: no cover
        log.warning("No se pudo multiplexar el audio: %s", exc)
        return False
