"""Curador de faceset para entrenar un modelo .dfm (DeepFaceLab) — CLI.

Envuelve ``fuser.core.faceset.curate`` (la misma lógica que usa la pestaña
"🧬 Crear modelo" de la app). Cura una carpeta de imágenes de UNA persona y deja
solo las útiles, listas para el "extract faces" de DeepFaceLab.

Uso:
    python scripts/prep_faceset.py --input  C:\\ruta\\fotos_persona
                                   [--output C:\\ruta\\faceset_listo]
                                   [--min-face 128] [--min-sharpness 60] [--dedup 0.96]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> int:
    ap = argparse.ArgumentParser(description="Cura un faceset para entrenar un .dfm")
    ap.add_argument("--input", required=True, help="Carpeta con las fotos candidatas de la persona")
    ap.add_argument("--output", default=None, help="Carpeta de salida (default: <input>_faceset)")
    ap.add_argument("--min-face", type=int, default=128, help="Lado mínimo de la cara en px (default 128)")
    ap.add_argument("--min-sharpness", type=float, default=60.0, help="Varianza Laplaciana mínima (nitidez)")
    ap.add_argument("--dedup", type=float, default=0.96, help="Cosine sobre el que 2 caras son casi-idénticas")
    args = ap.parse_args()

    in_dir = Path(args.input).expanduser().resolve()
    if not in_dir.is_dir():
        print(f"[ERROR] No existe la carpeta: {in_dir}")
        return 2
    out_dir = Path(args.output).expanduser().resolve() if args.output else in_dir.with_name(in_dir.name + "_faceset")

    from fuser.core import faceset

    images = list(faceset.iter_images(in_dir))
    if not images:
        print(f"[ERROR] No hay imágenes en {in_dir}")
        return 2
    print(f"Escaneando {len(images)} imágenes…")

    def _p(frac, msg):
        print(f"\r  {msg}", end="", flush=True)

    rep = faceset.curate(images, out_dir=out_dir, min_face=args.min_face,
                         min_sharpness=args.min_sharpness, dedup=args.dedup, progress=_p)
    print("\n" + "=" * 58)
    print(f"  RESULTADO: {rep['kept']} buenas de {rep['scanned']} escaneadas")
    print("=" * 58)
    labels = {"ilegibles": "ilegibles", "sin_cara": "sin cara detectable",
              "varias_caras": "varias caras (ambiguo)", "cara_chica": f"cara < {args.min_face}px",
              "borrosas": "borrosas", "luz_mala": "muy oscuras/quemadas", "casi_duplicadas": "casi-duplicadas"}
    for r, n in rep["dropped"].items():
        print(f"  · {labels.get(r, r):24} {n:4}   (ej: {rep['dropped_examples'].get(r,'')})")
    c = rep["coverage"]
    print(f"\nCobertura de ángulos → frontal {c['front']} · perfil A {c['left']} · perfil B {c['right']}")
    idn = rep.get("identity") or {}
    if idn:
        print(f"Consistencia de identidad → cos min {idn['min_cos']} media {idn['mean_cos']}")
    for r in rep.get("recommendations", []):
        print(f"  · {r}")
    if rep.get("out_dir"):
        print(f"\n✅ {rep['kept']} imágenes curadas -> {rep['out_dir']}")
        print("   Subí ESA carpeta a data_src de DeepFaceLab y corré 'extract faces' (Whole Face).")
    else:
        print("\n[!] No quedó ninguna imagen útil. Bajá --min-sharpness/--min-face o sumá mejores fotos.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
