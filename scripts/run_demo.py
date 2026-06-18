"""Prueba automática de Fuser (lo PRIMERO que conviene correr tras instalar).

Qué hace:
1. Descarga una **foto fuente** de stock (una cara realista) a ``prueba/``.
2. Consigue un **vídeo objetivo** (idealmente una mujer cantando que se pasa la
   mano por la cara, para probar boca abierta + oclusión + perfiles):
      - usa ``--video URL`` o la variable ``FUSER_DEMO_VIDEO``, o
      - usa ``prueba/target.mp4`` si ya existe, o
      - te indica de dónde bajar uno (las webs de stock bloquean la descarga
        automática por bot, así que ese clip se pone a mano una sola vez).
3. Recorta el vídeo a un clip corto y prueba **varias configuraciones**
   (InsightFace rápido, modo musical, y FaceFusion si está instalado),
   guardando previsualizaciones y vídeos en la carpeta ``prueba/``.

Uso:
    python scripts/run_demo.py
    python scripts/run_demo.py --video https://.../clip.mp4
    FUSER_DEMO_VIDEO=/ruta/clip.mp4 python scripts/run_demo.py
"""
from __future__ import annotations

import argparse
import os
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402

from fuser import config  # noqa: E402
from fuser.config import (  # noqa: E402
    ENGINE_FACEFUSION, ENGINE_INSIGHTFACE, EXPR_MUSIC_VIDEO, EXPR_STANDARD, Settings,
)
from fuser.utils import video as videoutil  # noqa: E402

PRUEBA = config.PROJECT_ROOT / "prueba"
_BROWSER_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"

# Foto fuente: cara realista generada (siempre disponible, sin login).
SOURCE_URL = os.environ.get("FUSER_DEMO_SOURCE", "https://thispersondoesnotexist.com/")
# Vídeo objetivo: configurable. Las webs de stock bloquean la descarga por bot.
VIDEO_URL = os.environ.get("FUSER_DEMO_VIDEO", "")

DEMO_MAX_FRAMES = int(os.environ.get("FUSER_DEMO_FRAMES", "120"))  # clip corto = demo rápida


def _download(url: str, dest: Path) -> bool:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": _BROWSER_UA})
        with urllib.request.urlopen(req, timeout=90) as resp, open(dest, "wb") as out:
            out.write(resp.read())
        return dest.stat().st_size > 10_000
    except Exception as exc:
        print(f"  ⚠️  No se pudo descargar {url}: {exc}")
        return False


def _ensure_source() -> Path:
    dest = PRUEBA / "source.jpg"
    if dest.exists() and dest.stat().st_size > 10_000:
        print(f"  ✅ Foto fuente ya presente: {dest}")
        return dest
    print(f"  ⬇️  Descargando foto fuente de stock…")
    if _download(SOURCE_URL, dest) and cv2.imread(str(dest)) is not None:
        print(f"  ✅ Foto fuente: {dest}")
        return dest
    raise SystemExit(
        f"\n❌ No pude obtener la foto fuente. Pon una foto de cara como {dest} y reejecuta."
    )


def _ensure_video(arg_url: str) -> Path:
    # 1) URL por argumento o variable de entorno.
    url = arg_url or VIDEO_URL
    dest = PRUEBA / "target.mp4"
    if url:
        print(f"  ⬇️  Descargando vídeo objetivo…")
        if _download(url, dest):
            print(f"  ✅ Vídeo objetivo: {dest}")
            return dest
    # 2) Archivo ya presente en prueba/.
    for name in ("target.mp4", "target.mov", "target.webm", "target.mkv"):
        cand = PRUEBA / name
        if cand.exists() and cand.stat().st_size > 10_000:
            print(f"  ✅ Vídeo objetivo ya presente: {cand}")
            return cand
    # 3) Instrucción de un paso.
    raise SystemExit(
        "\n❌ Falta el vídeo objetivo. Las webs de stock bloquean la descarga automática, así que:\n"
        "   1) Descarga un clip CORTO de una mujer cantando que se pase la mano por la cara, p. ej. de:\n"
        "        https://www.pexels.com/search/videos/woman%20singing/\n"
        "        https://mixkit.co/free-stock-video/singing/\n"
        f"   2) Guárdalo como:  {PRUEBA / 'target.mp4'}\n"
        "   3) Reejecuta:  python scripts/run_demo.py\n"
        "   (o pásalo directo:  python scripts/run_demo.py --video <URL_o_ruta>)"
    )


def _trim(src: Path, max_frames: int) -> Path:
    """Recorta a un clip corto para que la demo sea rápida."""
    info = videoutil.probe(str(src))
    if info.frame_count <= max_frames:
        return src
    dst = PRUEBA / "target_clip.mp4"
    writer = videoutil.FFmpegVideoWriter(str(dst), info.width, info.height, info.fps, crf=18)
    try:
        for i, frame in enumerate(videoutil.read_frames(str(src))):
            if i >= max_frames:
                break
            writer.write(frame)
    finally:
        writer.close()
    print(f"  ✂️  Clip de prueba: {dst} ({max_frames} frames)")
    return dst


def _settings_for(engine: str, mode: str) -> Settings:
    """Settings aplicando el preset de expresión (como hace la UI)."""
    preset = config.EXPRESSION_PRESETS.get(mode, {})
    fields = {k: v for k, v in preset.items()
              if k in Settings.__dataclass_fields__ and k != "engine"}
    fields["processing_resolution"] = 720  # demo: rápido
    return Settings(engine=engine, expression_mode=mode, **fields)


def _run_config(name: str, settings: Settings, source: Path, video: Path) -> None:
    from fuser.core.pipeline import SwapPipeline

    print(f"\n=== Configuración: {name} ===")
    try:
        pipe = SwapPipeline(settings)
        pipe.load_models(progress=lambda f, m="": None)
        img = cv2.imread(str(source))
        stats = pipe.prepare_source([img])
        print(f"  Fuente: {stats.summary() if stats else 'ok'}")

        # Previsualizaciones (frames clave) -> imágenes
        previews = pipe.preview(str(video), n_frames=4)
        for rgb, caption in previews:
            safe = caption.replace(" ", "_").replace("/", "")
            cv2.imwrite(str(PRUEBA / f"preview_{name}_{safe}.png"),
                        cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        print(f"  ✅ {len(previews)} previsualizaciones guardadas")

        # Vídeo completo (clip corto)
        out = PRUEBA / f"out_{name}.mp4"
        pipe.process_video(str(video), output_path=str(out), progress=lambda f, m="": None)
        print(f"  ✅ Vídeo: {out}")
    except Exception as exc:
        print(f"  ❌ Falló '{name}': {exc}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Prueba automática de Fuser")
    parser.add_argument("--video", default="", help="URL o ruta del vídeo objetivo")
    args = parser.parse_args()

    PRUEBA.mkdir(parents=True, exist_ok=True)
    print(f"Carpeta de resultados: {PRUEBA}\n")

    source = _ensure_source()
    video = _trim(_ensure_video(args.video), DEMO_MAX_FRAMES)

    configs = [
        ("insightface_rapido", _settings_for(ENGINE_INSIGHTFACE, EXPR_STANDARD)),
        ("insightface_musical", _settings_for(ENGINE_INSIGHTFACE, EXPR_MUSIC_VIDEO)),
    ]
    # FaceFusion solo si ya está instalado (no disparamos su instalación en la demo).
    try:
        from fuser.engines.facefusion_engine import is_available

        if is_available():
            configs.append(("facefusion_musical", _settings_for(ENGINE_FACEFUSION, EXPR_MUSIC_VIDEO)))
        else:
            print("ℹ️  FaceFusion no instalado: se omite esa prueba. "
                  "Instálalo (scripts/install_facefusion.py) para compararlo.")
    except Exception:
        pass

    for name, settings in configs:
        _run_config(name, settings, source, video)

    print(f"\n✅ Listo. Revisa los resultados en:  {PRUEBA}")
    print("   - preview_*.png  (frames clave por configuración)")
    print("   - out_*.mp4      (vídeos por configuración)")
    print("   Compara las distintas configuraciones y quédate con la que mejor se vea.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
