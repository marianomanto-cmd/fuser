"""Doctor / instalador de la función *Imagen → Vídeo* (Wan 2.2 I2V vía ComfyUI).

Esta función usa un **ComfyUI aparte** como motor. Este script NO instala
ComfyUI (eso lo haces una vez, ver ``docs/IMAGE_TO_VIDEO.md``); lo que hace es:

    python scripts/setup_i2v.py                 # diagnóstico: ¿ComfyUI vivo? ¿nodos? ¿modelos?
    python scripts/setup_i2v.py --list          # lista de modelos + URLs + carpetas (descarga manual)
    python scripts/setup_i2v.py --download \\
        --comfy-path /ruta/a/ComfyUI            # descarga los modelos a ComfyUI/models/

Para los modelos *gated* (Stable Audio) exporta tu token: ``export HF_TOKEN=hf_...``
tras aceptar la licencia en https://huggingface.co/stabilityai/stable-audio-open-1.0

Solo usa la librería estándar + el paquete ``fuser`` (no necesita gradio/cv2).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fuser.i2v import config as C  # noqa: E402
from fuser.i2v import models as M  # noqa: E402
from fuser.i2v.comfy_client import ComfyUIClient  # noqa: E402

OK, WARN, BAD = "✅", "⚠️ ", "❌"

# Nodo -> de qué custom node viene (para las pistas de instalación).
NODE_SOURCES = {
    "UnetLoaderGGUF": "ComfyUI-GGUF · https://github.com/city96/ComfyUI-GGUF",
    "CLIPLoaderGGUF": "ComfyUI-GGUF · https://github.com/city96/ComfyUI-GGUF",
    "VHS_VideoCombine": "VideoHelperSuite · https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite",
    "UnetLoaderGGUFDisTorch2MultiGPU": "ComfyUI-MultiGPU · https://github.com/pollockjj/ComfyUI-MultiGPU",
    "WanImageToVideo": "ComfyUI core (actualiza ComfyUI)",
    "ModelSamplingSD3": "ComfyUI core",
    "CLIPLoader": "ComfyUI core",
    "VAELoader": "ComfyUI core",
    "KSamplerAdvanced": "ComfyUI core",
    "VAEDecode": "ComfyUI core",
    "CheckpointLoaderSimple": "ComfyUI core",
    "EmptyLatentAudio": "ComfyUI core",
    "VAEDecodeAudio": "ComfyUI core",
    "SaveAudio": "ComfyUI core",
}


def _required_nodes() -> list[str]:
    from fuser.i2v import workflow as wf
    types: set[str] = set()
    for name in (C.WF_WAN22_I2V_GGUF, C.WF_WAN22_I2V_DISTORCH, C.WF_STABLE_AUDIO):
        try:
            for node in wf.load_workflow(name).values():
                types.add(node["class_type"])
        except Exception:
            pass
    return sorted(types)


def cmd_list() -> int:
    print("Modelos necesarios para Imagen → Vídeo:\n")
    for m in M.ALL_MODELS:
        tag = " (GATED: requiere token HF)" if m.gated else ""
        print(f"  • {m.filename}{tag}")
        print(f"      carpeta : ComfyUI/models/{m.subfolder}/")
        print(f"      tamaño  : ~{m.size_gb:.1f} GB · {m.kind}")
        print(f"      URL     : {m.url}")
        print()
    print("Custom nodes recomendados:")
    for repo in sorted(set(NODE_SOURCES.values())):
        if "core" not in repo:
            print(f"  • {repo}")
    return 0


def cmd_download(comfy_path: str, include_audio: bool) -> int:
    root = M.comfy_models_root(comfy_path)
    if root is None:
        print(f"{BAD} No parece una instalación de ComfyUI: {comfy_path}")
        print("   Pasa --comfy-path /ruta/a/ComfyUI (la carpeta que tiene main.py).")
        return 1
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    print(f"➡️  Descargando modelos a {root} (esto pesa decenas de GB)…")

    def progress(frac, msg):
        bar = int(frac * 30)
        print(f"\r   [{'#' * bar}{'.' * (30 - bar)}] {int(frac*100):3d}% {msg}", end="", flush=True)

    msgs = M.download_models(root, include_audio=include_audio, token=token, progress=progress)
    print()
    if msgs:
        print("\nAvisos:")
        for m in msgs:
            print(f"  {m}")
    print(f"\n{OK} Descarga terminada (revisa avisos arriba).")
    return 0


def cmd_check(url: str) -> int:
    print("=" * 64)
    print(" Fuser · Imagen → Vídeo — diagnóstico de ComfyUI")
    print("=" * 64)
    print(f"URL de ComfyUI : {url}")
    client = ComfyUIClient(url, timeout=8)
    if not client.is_available():
        print(f"{BAD} ComfyUI no responde. Arráncalo (en su carpeta):")
        for label, key in C.OFFLOAD_LABELS.items():
            flags = " ".join(C.OFFLOAD_PRESETS[key]["comfy_flags"])
            print(f"     [{label}]  python main.py --listen 127.0.0.1 --port 8188 {flags}")
        return 1
    print(f"{OK} ComfyUI responde.")

    try:
        info = client.get_object_info()
    except Exception as exc:
        print(f"{WARN}No pude leer /object_info: {exc}")
        return 1

    print("-" * 64)
    print("Custom nodes:")
    missing = []
    for node in _required_nodes():
        present = node in info
        src = NODE_SOURCES.get(node, "?")
        print(f"  {OK if present else BAD} {node:34s} {'' if present else '← ' + src}")
        if not present:
            missing.append(node)

    print("-" * 64)
    print("Modelos (vistos por ComfyUI):")
    absent = []
    for row in M.model_report(info, include_audio=True):
        m, ok = row["model"], row["present"]
        mark = OK if ok else (BAD if ok is False else WARN)
        print(f"  {mark} {m.filename}")
        if ok is False:
            absent.append(m.filename)

    print("=" * 64)
    if missing:
        print(f"{BAD} Faltan nodos: instálalos con ComfyUI-Manager y reinicia ComfyUI.")
    if absent:
        print(f"{BAD} Faltan modelos: descárgalos con  python scripts/setup_i2v.py --download "
              f"--comfy-path /ruta/a/ComfyUI")
    if not missing and not absent:
        print(f"{OK} ¡Todo listo! Genera desde la pestaña 'Imagen → Vídeo' de la app.")
        return 0
    return 1


def main() -> int:
    p = argparse.ArgumentParser(description="Doctor/instalador de Imagen → Vídeo (ComfyUI)")
    p.add_argument("--url", default=C.DEFAULT_COMFY_URL, help="URL de ComfyUI")
    p.add_argument("--list", action="store_true", help="Lista modelos + URLs (descarga manual)")
    p.add_argument("--download", action="store_true", help="Descarga modelos a ComfyUI/models/")
    p.add_argument("--comfy-path", default=C.COMFY_PATH, help="Ruta a la instalación de ComfyUI")
    p.add_argument("--no-audio", action="store_true", help="No descargar modelos de audio")
    args = p.parse_args()

    if args.list:
        return cmd_list()
    if args.download:
        return cmd_download(args.comfy_path, include_audio=not args.no_audio)
    return cmd_check(args.url)


if __name__ == "__main__":
    raise SystemExit(main())
