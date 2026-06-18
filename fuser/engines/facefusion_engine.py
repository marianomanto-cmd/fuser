"""Motor FaceFusion (alta calidad) — adaptador a sus módulos internos.

En lugar de llamar a FaceFusion por línea de comandos, importamos sus módulos
(``facefusion.processors.modules.face_swapper`` / ``face_enhancer`` y el
analizador de caras) y los conducimos frame a frame. La configuración (modelos,
máscaras, *pixel boost*) y los **execution providers** se inyectan en el
``state_manager`` de FaceFusion, integrándolos con nuestro ``memory_manager``.

Ventajas para videos musicales:
- **pixel boost** (correr el swapper a 256/512) → dientes y ojos más nítidos.
- máscaras de **oclusión** y por **región** → perfiles y pelo cruzando la cara.
- multi-referencia nativa (cara promedio de varias fotos).
- ``video_memory_strategy`` (strict/moderate/tolerant) → **offloading a CPU/RAM**
  cuando la VRAM está al límite, mapeado desde el modo de memoria de Fuser.

⚠️ Requiere FaceFusion instalado (`pip install facefusion` o el repo). La API
interna varía entre versiones: este adaptador apunta a FaceFusion 3.x y degrada
con un mensaje claro si algo no encaja (puedes usar InsightFace mientras tanto).
"""
from __future__ import annotations

import importlib
from typing import List, Optional

import cv2
import numpy as np

from .. import config
from ..utils.logging import get_logger
from .base import BaseFaceSwapper

log = get_logger(__name__)


# Mapeo modo de memoria de Fuser -> estrategia de VRAM de FaceFusion.
_VRAM_STRATEGY = {
    config.MODE_MAX_QUALITY: "tolerant",
    config.MODE_BALANCED: "tolerant",
    config.MODE_LOW_VRAM: "moderate",
    config.MODE_EXTREME_LOW_VRAM: "strict",
}


# Detección, auto-instalación y error viven en facefusion_bootstrap (reexportados
# aquí para compatibilidad: otros módulos importan is_available desde este motor).
from .facefusion_bootstrap import (  # noqa: E402
    FaceFusionNotAvailable,
    ensure,
    ensure_on_path,
    is_available,
)


class FaceFusionSwapper(BaseFaceSwapper):
    name = config.ENGINE_FACEFUSION
    display_name = "FaceFusion (Alta Calidad)"
    description = "Mejor en boca abierta, dientes y perfiles. Más lento, más VRAM."

    def __init__(self, settings, memory_manager):
        super().__init__(settings, memory_manager)
        self._modules = {}
        self._source_face = None
        self._analyser = None    # detector InsightFace para el post-procesado por regiones
        self._mouth_enh = None   # enhancer (CodeFormer) para el paso localizado de boca
        self._loaded = False

    # ----- post-procesado por regiones (calidad de boca/dientes/ojos) ---------
    def _mouth_open_boost(self) -> float:
        m = self.settings.expression_mode
        if m == config.EXPR_HIGH_EXPRESSION:
            return 2.0
        if m == config.EXPR_MUSIC_VIDEO:
            return 1.5
        return 1.0

    def _ensure_analyser(self):
        if self._analyser is not None:
            return self._analyser
        from ..models.face_analyser import FaceAnalyser

        self._analyser = FaceAnalyser(self.mm.analyser_providers(), self.mm.ctx_id(), self.mm.det_size)
        self._analyser.load()
        return self._analyser

    def _detect(self, frame: np.ndarray):
        try:
            return self._ensure_analyser().get_faces(frame)
        except Exception as exc:  # pragma: no cover
            log.warning("Detección para post-procesado falló: %s", exc)
            return []

    def _yaw(self, face) -> float:
        from ..models.face_analyser import FaceAnalyser

        return FaceAnalyser.estimate_yaw(face)

    def _ensure_mouth_enhancer(self):
        """Carga perezosa de CodeFormer para el realce LOCALIZADO de la boca."""
        if self._mouth_enh is not None:
            return self._mouth_enh
        from ..models.downloader import ensure_model
        from ..models.face_enhancer import FaceEnhancer

        info = config.ENHANCER_MODELS["codeformer"]
        path = ensure_model("codeformer")
        on_cpu = not self.mm.use_gpu
        self._mouth_enh = FaceEnhancer(path, self.mm.enhancer_providers(), info,
                                       self.mm.session_options(on_cpu))
        self._mouth_enh.load()
        return self._mouth_enh

    def _enhance_mouth_region(self, frame, face, openness: float, profile: float):
        """Pasa CodeFormer por la cara alineada y pega SOLO la boca (dientes nítidos).

        El enhancer se ejecuta sobre la cara alineada completa (como debe), pero se
        compone de vuelta únicamente en la región de la boca, con fuerza escalada
        por la apertura y atenuada en perfiles fuertes (donde el alineado es menos
        fiable). Es un 2.º paso de enhancer dedicado a la boca/dientes.
        """
        from insightface.utils import face_align

        from ..utils import image as imageutil

        aligned, M = face_align.norm_crop2(frame, face.kps, 512)
        enhanced = self._ensure_mouth_enhancer().run(aligned, fidelity=self.settings.codeformer_fidelity)
        kps_aligned = imageutil.transform_points(face.kps, M)
        _, mouth = imageutil.frame_eye_mouth_masks(kps_aligned, (512, 512), self._mouth_open_boost())
        strength = float(np.clip(self.settings.mouth_detail * openness * (1.0 - 0.4 * profile), 0.0, 1.0))
        return imageutil.paste_back_with_mask(frame, enhanced, M, mouth * strength, opacity=1.0)

    def _postprocess(self, frame: np.ndarray, faces) -> np.ndarray:
        """Post-procesado que hace a FaceFusion claramente superior en casos difíciles.

        Por cada cara:
        1) Detecta la **apertura de boca** por landmarks (MAR) — fuerte solo si abierta.
        2) Realza ojos (siempre) y boca (escalado por apertura) con detalle local.
        3) Si la boca está abierta, aplica un **enhancer localizado (CodeFormer)** en
           la boca → dientes nítidos, no borrosos.
        4) En **perfiles** suaviza el blending (evita costuras en mandíbula/oreja).

        Se puede desactivar por completo con ``mouth_detail=0``/``eye_preservation=0``
        y el paso de enhancer con ``mouth_enhancer=False``.
        """
        from ..utils import image as imageutil

        s = self.settings
        if not faces or (s.eye_preservation <= 0 and s.mouth_detail <= 0):
            return frame

        out = frame
        boost = self._mouth_open_boost()
        for f in faces:
            kps = f.kps
            lmk = getattr(f, "landmark_2d_106", None)
            _, mouth_mask = imageutil.frame_eye_mouth_masks(kps, out.shape, boost)
            openness = imageutil.mouth_openness(kps, lmk, out, mouth_mask)
            profile = float(np.clip((abs(self._yaw(f)) - 20.0) / 50.0, 0.0, 1.0))

            mouth_amt = s.mouth_detail * (0.5 + 0.9 * openness)  # dientes solo si abierta
            out = imageutil.enhance_regions(
                out, [kps], eye_strength=s.eye_preservation, mouth_strength=mouth_amt,
                mouth_open_boost=boost, adaptive_mouth=False,
            )
            if s.mouth_enhancer and s.mouth_detail > 0 and openness > 0.35:
                try:
                    out = self._enhance_mouth_region(out, f, openness, profile)
                except Exception as exc:  # pragma: no cover - requiere insightface/modelo
                    log.warning("Enhancer localizado de boca no disponible (%s); uso solo realce.", exc)
        return out

    # ----- importación perezosa de FaceFusion ---------------------------------
    def _import(self):
        if self._modules:
            return self._modules
        ensure_on_path()
        try:
            state_manager = importlib.import_module("facefusion.state_manager")
            # Analizador de caras (la ruta cambió entre versiones).
            face_analyser = None
            for path in ("facefusion.face_analyser", "facefusion.face_helper"):
                try:
                    face_analyser = importlib.import_module(path)
                    break
                except Exception:
                    continue
            # Procesadores (3.x: processors.modules; 2.x: processors.frame.modules).
            swapper = enhancer = None
            for base in ("facefusion.processors.modules", "facefusion.processors.frame.modules"):
                try:
                    swapper = importlib.import_module(base + ".face_swapper")
                    enhancer = importlib.import_module(base + ".face_enhancer")
                    break
                except Exception:
                    continue
            if not (state_manager and face_analyser and swapper):
                raise ImportError("módulos internos de FaceFusion no encontrados")
            self._modules = dict(
                state=state_manager, face_analyser=face_analyser,
                swapper=swapper, enhancer=enhancer,
            )
        except Exception as exc:  # pragma: no cover - depende de la instalación
            raise FaceFusionNotAvailable(str(exc))
        return self._modules

    def _set(self, key: str, value) -> None:
        """Fija un item del state_manager de FaceFusion de forma tolerante."""
        state = self._modules["state"]
        for fn in ("set_item", "init_item"):
            if hasattr(state, fn):
                try:
                    getattr(state, fn)(key, value)
                    return
                except Exception:
                    continue

    # ----- configuración (integra con memory_manager + modo musical) ----------
    def _configure(self) -> None:
        s = self.settings
        music = s.expression_mode in (config.EXPR_MUSIC_VIDEO, config.EXPR_HIGH_EXPRESSION)

        providers = ["cuda"] if self.mm.use_gpu else ["cpu"]
        # Ejecución / memoria
        self._set("execution_providers", providers)
        self._set("execution_device_id", "0")
        self._set("execution_thread_count", max(1, self.mm.info.cpu_count // 2))
        self._set("execution_queue_count", 1)
        self._set("download_providers", ["github", "huggingface"])
        # Offloading VRAM->RAM/CPU según el modo de memoria.
        self._set("video_memory_strategy", _VRAM_STRATEGY.get(s.memory_mode, "moderate"))

        # Detector (yoloface va bien en perfiles); tamaño según det_size.
        self._set("face_detector_model", "yoloface")
        self._set("face_detector_size", f"{self.mm.det_size}x{self.mm.det_size}")
        self._set("face_detector_angles", [0])
        # Umbral más permisivo: detecta mejor perfiles laterales.
        self._set("face_detector_score", 0.5 if music else 0.6)

        # Selección de caras objetivo.
        selector = {
            config.FACE_SELECTOR_ALL: "many",
            config.FACE_SELECTOR_LARGEST: "one",
            config.FACE_SELECTOR_REFERENCE: "reference",
            config.FACE_SELECTOR_INDEX: "one",
        }.get(s.face_selector, "many")
        self._set("face_selector_mode", selector)
        self._set("face_selector_order", "large-small")

        # Swapper + pixel boost (clave de calidad: corre el swap a mayor resolución).
        self._set("face_swapper_model", s.ff_swapper_model)
        self._set("face_swapper_pixel_boost", "512x512" if music else s.ff_pixel_boost)

        # Enhancer: en modo musical, CodeFormer fuerte (mejor dientes/textura).
        enhancer_model = "codeformer" if music else (
            s.enhancer_model if s.enhancer_model and s.enhancer_model != "none" else "gfpgan_1.4"
        )
        self._set("face_enhancer_model", enhancer_model)
        self._set("face_enhancer_blend", int(round(max(s.enhancer_blend, 0.85 if music else 0.6) * 100)))

        # Máscaras: oclusión (perfiles/pelo/manos) + región (ojos/boca) + caja.
        if music or s.mask_mode == config.MASK_PARSING:
            self._set("face_mask_types", ["box", "occlusion", "region"])
        else:
            self._set("face_mask_types", ["box", "occlusion"])
        self._set("face_mask_blur", float(np.clip(s.mask_blur, 0.0, 1.0)))
        self._set("face_mask_padding", (0, 0, 0, 0))
        # Regiones a conservar/incluir (todas las faciales, incluida la boca).
        self._set("face_mask_regions", [
            "skin", "left-eyebrow", "right-eyebrow", "left-eye", "right-eye",
            "nose", "mouth", "upper-lip", "lower-lip",
        ])

    # ----- ciclo de vida -------------------------------------------------------
    def load(self, progress=None) -> None:
        if self._loaded:
            return
        # Auto-instala FaceFusion la primera vez (sin pasos manuales).
        ensure(progress, auto=getattr(self.settings, "ff_auto_install", True))
        if progress:
            progress(0.2, "Importando FaceFusion…")
        self._import()
        self._configure()
        if progress:
            progress(0.5, "Descargando/preparando modelos de FaceFusion…")
        # pre_check descarga modelos y valida; lo llamamos por módulo si existe.
        for key in ("swapper", "enhancer"):
            mod = self._modules.get(key)
            if mod is not None and hasattr(mod, "pre_check"):
                try:
                    mod.pre_check()
                except Exception as exc:  # pragma: no cover
                    log.warning("FaceFusion %s.pre_check() falló: %s", key, exc)
        self._loaded = True
        if progress:
            progress(1.0, "Motor FaceFusion listo.")

    @property
    def loaded(self) -> bool:
        return self._loaded

    @property
    def has_source(self) -> bool:
        return self._source_face is not None

    def update_runtime(self, settings) -> None:
        self.settings = settings
        if self._loaded:
            self._configure()

    # ----- fuente (multi-referencia nativa de FaceFusion) ----------------------
    def prepare_source(self, images: List[np.ndarray]):
        if not self._loaded:
            self.load()
        fa = self._modules["face_analyser"]
        get_average = getattr(fa, "get_average_face", None)
        if get_average is None:
            raise FaceFusionNotAvailable("face_analyser.get_average_face no encontrado")
        try:
            self._source_face = get_average(images)
        except TypeError:
            # Algunas versiones esperan (frames, position).
            self._source_face = get_average(images, 0)
        if self._source_face is None:
            raise ValueError("FaceFusion no detectó ninguna cara en las imágenes fuente.")
        from ..core.face_store import SourceStats

        return SourceStats(n_input=len(images), n_used=len(images), mean_yaw=0.0, rejected=0)

    # ----- procesamiento -------------------------------------------------------
    def _run_module(self, mod, inputs: dict) -> Optional[np.ndarray]:
        """Llama a process_frame del módulo, tolerando variaciones de firma."""
        fn = getattr(mod, "process_frame", None)
        if fn is None:
            return None
        try:
            return fn(inputs)
        except TypeError:
            # Variante posicional (target_vision_frame,)
            return fn(inputs.get("target_vision_frame"))

    def _ff_swap(self, frame: np.ndarray) -> np.ndarray:
        """Swap + enhancer nativos de FaceFusion sobre un frame completo."""
        inputs = {
            "reference_faces": None,
            "source_face": self._source_face,
            "target_vision_frame": frame,
        }
        out = self._run_module(self._modules["swapper"], inputs)
        out = out if out is not None else frame
        enhancer = self._modules.get("enhancer")
        if enhancer is not None and self.settings.enhancer_model != "none":
            enh = self._run_module(enhancer, {"reference_faces": None, "target_vision_frame": out})
            if enh is not None:
                out = enh
        return out

    def process_frame(self, frame: np.ndarray, use_smoothing: bool = True) -> np.ndarray:
        if not self._loaded:
            self.load()
        if self._source_face is None:
            raise RuntimeError("Falta la cara fuente (FaceFusion). Sube imágenes fuente primero.")
        out = self._ff_swap(frame)
        # Post-procesado: detectar sobre la salida para alinear el realce de regiones.
        faces = self._detect(out)
        return self._postprocess(out, faces)

    # ----- 2 pasadas: FF swap + post-procesado de regiones ESTABILIZADO --------
    # Pasada 1: detectamos (barato) y suavizamos los landmarks usando RAM.
    # Pasada 2: FaceFusion swapea y aplicamos el realce de boca/ojos con kps
    # suavizados -> dientes/ojos nítidos y SIN parpadeo entre frames.
    def supports_two_pass(self) -> bool:
        return True

    def supports_adaptive_mouth(self) -> bool:
        return True

    def enhance_mouth_region(self, frame: np.ndarray, face, openness: float = 1.0) -> np.ndarray:
        """Realce localizado de boca/dientes (interfaz pública). Guardado."""
        if not self.settings.mouth_enhancer or self.settings.mouth_detail <= 0:
            return frame
        try:
            profile = float(np.clip((abs(self._yaw(face)) - 20.0) / 50.0, 0.0, 1.0))
            return self._enhance_mouth_region(frame, face, openness, profile)
        except Exception as exc:  # pragma: no cover
            log.warning("enhance_mouth_region no disponible: %s", exc)
            return frame

    def get_memory_usage(self) -> dict:
        return {
            "engine": self.name,
            "loaded": self._loaded,
            "providers": "cuda" if self.mm.use_gpu else "cpu",
            "video_memory_strategy": _VRAM_STRATEGY.get(self.settings.memory_mode, "moderate"),
            "swapper_loaded": "swapper" in self._modules,
            "mouth_enhancer_loaded": self._mouth_enh is not None,
            "analyser_loaded": self._analyser is not None,
        }

    def detect(self, frame: np.ndarray):
        return self._detect(frame)

    def select_targets(self, faces):
        return faces

    def render(self, frame: np.ndarray, targets):
        if not self._loaded:
            self.load()
        if self._source_face is None:
            raise RuntimeError("Falta la cara fuente (FaceFusion). Sube imágenes fuente primero.")
        out = self._ff_swap(frame)
        return self._postprocess(out, targets)

    def unload(self) -> None:
        self._modules = {}
        self._source_face = None
        if self._analyser is not None and hasattr(self._analyser, "unload"):
            self._analyser.unload()
        self._analyser = None
        if self._mouth_enh is not None and hasattr(self._mouth_enh, "unload"):
            self._mouth_enh.unload()
        self._mouth_enh = None
        self._loaded = False
