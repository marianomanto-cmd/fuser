"""Doctor de entorno de Fuser.

Comprueba que la máquina local está lista para correr la app y da recomendaciones
claras. Pensado para ejecutarse justo después de instalar las dependencias:

    python scripts/check_env.py

Solo usa la librería estándar para arrancar; las dependencias pesadas
(onnxruntime, psutil, pynvml) se consultan de forma perezosa y tolerante a
fallos, así que el script funciona incluso si falta algo (y te dice qué falta).
"""
from __future__ import annotations

import importlib
import platform
import sys
from pathlib import Path

# Permite ejecutarlo sin instalar el paquete.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fuser import __app_name__, __version__, config  # noqa: E402
from fuser.models.downloader import is_downloaded  # noqa: E402

OK = "✅"
WARN = "⚠️ "
BAD = "❌"


def _check_import(module: str) -> tuple[bool, str]:
    try:
        m = importlib.import_module(module)
        return True, getattr(m, "__version__", "ok")
    except Exception as exc:
        return False, str(exc)


def main() -> int:
    print("=" * 64)
    print(f" {__app_name__} v{__version__} · diagnóstico de entorno")
    print("=" * 64)
    print(f"Python   : {platform.python_version()}  ({sys.executable})")
    print(f"Sistema  : {platform.system()} {platform.release()} ({platform.machine()})")
    print("-" * 64)

    # --- Dependencias ---
    deps = ["numpy", "cv2", "gradio", "onnx", "insightface", "imageio_ffmpeg", "psutil"]
    print("Dependencias:")
    missing = []
    for d in deps:
        ok, info = _check_import(d)
        print(f"  {OK if ok else BAD} {d:16s} {info if ok else 'NO INSTALADO'}")
        if not ok:
            missing.append(d)

    # onnxruntime (gpu o cpu)
    ort_ok, ort_info = _check_import("onnxruntime")
    print(f"  {OK if ort_ok else BAD} {'onnxruntime':16s} {ort_info if ort_ok else 'NO INSTALADO'}")
    if not ort_ok:
        missing.append("onnxruntime / onnxruntime-gpu")

    print("-" * 64)

    # --- Hardware / aceleración ---
    try:
        from fuser.utils.system import get_system_info, ffmpeg_path

        info = get_system_info()
        print("Hardware y aceleración:")
        if info.has_cuda:
            print(f"  {OK} GPU CUDA: {info.gpu_name or 'detectada'}")
            if info.vram_total_gb:
                print(f"     VRAM: {info.vram_free_gb:.1f} GB libres / {info.vram_total_gb:.1f} GB")
            else:
                print(f"  {WARN}VRAM no consultable (instala 'pynvml' para verla en la UI)")
        else:
            print(f"  {BAD} Sin CUDA: la app correrá en CPU (lento; solo para probar la UI).")
            print("     → Instala 'onnxruntime-gpu' acorde a tu CUDA y revisa los drivers NVIDIA.")
        if info.ram_total_gb:
            print(f"  {OK} RAM: {info.ram_available_gb:.1f} GB libres / {info.ram_total_gb:.1f} GB")
        print(f"  {OK if info.ffmpeg_available else BAD} FFmpeg: "
              f"{ffmpeg_path() or 'no encontrado (instala imageio-ffmpeg)'}")
        print(f"     Providers ONNX: {', '.join(info.providers) or 'ninguno'}")
    except Exception as exc:
        print(f"  {WARN}No se pudo consultar el hardware: {exc}")
        info = None

    print("-" * 64)

    # --- Modelos ---
    print("Modelos (se descargan solos en el primer uso):")
    for key, m in config.MODEL_REGISTRY.items():
        present = is_downloaded(key)
        print(f"  {OK if present else WARN}{'descargado' if present else 'pendiente '} · {key}")
    print(f"  Carpeta de modelos: {config.MODELS_DIR}")

    print("=" * 64)

    # --- Veredicto ---
    if missing:
        print(f"{BAD} Faltan dependencias: {', '.join(missing)}")
        print("   Instálalas con:  pip install -r requirements.txt   (GPU)")
        print("                o:  pip install -r requirements-cpu.txt  (CPU / pruebas)")
        return 1
    if info and not info.has_cuda:
        print(f"{WARN}Todo instalado, pero SIN GPU CUDA. La UI funciona; el procesado será lento.")
        return 0
    print(f"{OK} ¡Entorno listo! Lanza la app con:  python app.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
