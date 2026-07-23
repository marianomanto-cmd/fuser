"""Curador de faceset para entrenar un modelo .dfm (DeepFaceLab).

El parecido final del .dfm depende MÁS del faceset que de las horas de GPU. Este
script cura una carpeta de imágenes candidatas de UNA persona y deja solo las
útiles, listas para el paso "extract faces" de DeepFaceLab:

  - descarta: ilegibles, sin cara, cara muy chica, borrosas, muy oscuras/quemadas
  - deduplica casi-idénticas (embedding ArcFace)
  - avisa si se coló OTRA persona (consistencia de identidad)
  - reporta la COBERTURA de ángulos (frontal / perfiles) para que sepas qué falta

NO recorta caras (eso lo hace DeepFaceLab): copia las imágenes BUENAS completas a
la carpeta de salida, renumeradas. Todo local; procesa por RUTA (no "mira" nada).

Uso:
    python scripts/prep_faceset.py --input  C:\\ruta\\fotos_persona
                                   [--output C:\\ruta\\faceset_listo]
                                   [--min-face 128] [--min-sharpness 60] [--dedup 0.96]
"""
from __future__ import annotations

import argparse
import shutil
import sys
from collections import Counter
from pathlib import Path

# permite correr desde la raíz del repo sin instalar el paquete
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _iter_images(root: Path):
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            yield p


def main() -> int:
    ap = argparse.ArgumentParser(description="Cura un faceset para entrenar un .dfm")
    ap.add_argument("--input", required=True, help="Carpeta con las fotos candidatas de la persona")
    ap.add_argument("--output", default=None, help="Carpeta de salida (default: <input>_faceset)")
    ap.add_argument("--min-face", type=int, default=128, help="Lado mínimo de la cara en px (default 128)")
    ap.add_argument("--min-sharpness", type=float, default=60.0, help="Varianza Laplaciana mínima (nitidez)")
    ap.add_argument("--dedup", type=float, default=0.96, help="Cosine sobre el que 2 caras son casi-idénticas")
    ap.add_argument("--copy", action="store_true", default=True, help="Copiar las buenas a --output (default sí)")
    args = ap.parse_args()

    in_dir = Path(args.input).expanduser().resolve()
    if not in_dir.is_dir():
        print(f"[ERROR] No existe la carpeta: {in_dir}")
        return 2
    out_dir = Path(args.output).expanduser().resolve() if args.output else in_dir.with_name(in_dir.name + "_faceset")

    import cv2
    import numpy as np
    from insightface.app import FaceAnalysis
    from fuser import config

    print("Cargando detector de caras (buffalo_l, CPU)…")
    det = FaceAnalysis(name="buffalo_l", root=str(config.INSIGHTFACE_ROOT),
                       providers=["CPUExecutionProvider"])
    det.prepare(ctx_id=-1, det_size=(640, 640))

    files = list(_iter_images(in_dir))
    if not files:
        print(f"[ERROR] No hay imágenes en {in_dir}")
        return 2
    print(f"Escaneando {len(files)} imágenes…\n")

    kept, kept_embs, kept_yaw = [], [], []
    drop = Counter()
    dropped_examples = {}

    def _note_drop(reason, path):
        drop[reason] += 1
        dropped_examples.setdefault(reason, path.name)

    for p in files:
        img = cv2.imread(str(p))
        if img is None:
            _note_drop("ilegibles", p); continue
        faces = det.get(img)
        if not faces:
            _note_drop("sin_cara", p); continue
        if len(faces) > 1:
            # varias caras: ambiguo para un faceset de 1 identidad
            _note_drop("varias_caras", p); continue
        f = faces[0]
        x1, y1, x2, y2 = f.bbox
        side = min(x2 - x1, y2 - y1)
        if side < args.min_face:
            _note_drop("cara_chica", p); continue
        # nitidez sobre el recorte de la cara
        cx1, cy1 = max(0, int(x1)), max(0, int(y1))
        crop = img[cy1:int(y2), cx1:int(x2)]
        if crop.size == 0:
            _note_drop("sin_cara", p); continue
        sharp = cv2.Laplacian(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
        if sharp < args.min_sharpness:
            _note_drop("borrosas", p); continue
        mean_v = float(crop.mean())
        if mean_v < 25 or mean_v > 235:
            _note_drop("luz_mala", p); continue
        emb = f.normed_embedding
        # dedup contra lo ya aceptado
        if kept_embs and float(np.max(np.dot(np.array(kept_embs), emb))) > args.dedup:
            _note_drop("casi_duplicadas", p); continue
        # yaw proxy: desplazamiento de la nariz respecto al centro de los ojos / io
        kps = f.kps
        io = float(np.linalg.norm(kps[0] - kps[1])) + 1e-6
        yaw = float((kps[2][0] - (kps[0][0] + kps[1][0]) / 2) / io)
        kept.append(p); kept_embs.append(emb); kept_yaw.append(yaw)

    # ---- reporte ----
    print("=" * 58)
    print(f"  RESULTADO: {len(kept)} buenas de {len(files)} escaneadas")
    print("=" * 58)
    if drop:
        print("Descartadas:")
        labels = {"ilegibles": "ilegibles", "sin_cara": "sin cara detectable",
                  "varias_caras": "varias caras (ambiguo)", "cara_chica": f"cara < {args.min_face}px",
                  "borrosas": "borrosas", "luz_mala": "muy oscuras/quemadas",
                  "casi_duplicadas": "casi-duplicadas"}
        for r, n in drop.most_common():
            print(f"  · {labels.get(r, r):24} {n:4}   (ej: {dropped_examples.get(r,'')})")

    if kept:
        yy = np.array(kept_yaw)
        left = int((yy > 0.15).sum()); right = int((yy < -0.15).sum())
        front = len(yy) - left - right
        print("\nCobertura de ángulos (para la cabeza en movimiento):")
        print(f"  · frontal : {front:4}")
        print(f"  · perfil A: {left:4}")
        print(f"  · perfil B: {right:4}")
        # consistencia de identidad
        centroid = np.mean(np.array(kept_embs), axis=0)
        centroid /= (np.linalg.norm(centroid) + 1e-8)
        cos = np.dot(np.array(kept_embs), centroid)
        outliers = [kept[i].name for i in range(len(kept)) if cos[i] < 0.30]
        print(f"\nConsistencia de identidad: cos min={cos.min():.2f} media={cos.mean():.2f}")
        if outliers:
            print(f"  ⚠️ {len(outliers)} posible(s) de OTRA persona (revisá): {', '.join(outliers[:6])}"
                  + (" …" if len(outliers) > 6 else ""))

        # recomendaciones
        print("\nRecomendaciones:")
        if len(kept) < 300:
            print(f"  · Tenés {len(kept)}; apuntá a 500-2000 para un .dfm decente. Sumá más fotos.")
        elif len(kept) < 500:
            print(f"  · {len(kept)} es un piso; 500-2000 da mejor parecido.")
        else:
            print(f"  · {len(kept)} imágenes: buen volumen.")
        if front < max(1, len(kept) // 6):
            print("  · Faltan FRONTALES.")
        if left < max(1, len(kept) // 8):
            print("  · Faltan PERFILES hacia un lado.")
        if right < max(1, len(kept) // 8):
            print("  · Faltan PERFILES hacia el otro lado.")

        # copia
        if args.copy:
            out_dir.mkdir(parents=True, exist_ok=True)
            for i, p in enumerate(kept):
                shutil.copyfile(p, out_dir / f"{i:04d}{p.suffix.lower()}")
            print(f"\n✅ {len(kept)} imágenes curadas -> {out_dir}")
            print("   Subí ESA carpeta a data_src de DeepFaceLab y corré 'extract faces' (Whole Face).")
    else:
        print("\n[!] No quedó ninguna imagen útil. Bajá --min-sharpness/--min-face o sumá mejores fotos.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
