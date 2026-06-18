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
        self._analyser = None  # detector InsightFace para el post-procesado por regiones
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

    def _postprocess(self, frame: np.ndarray, faces) -> np.ndarray:
        """Refuerza boca/dientes y ojos sobre la salida de FaceFusion.

        Es lo que convierte la calidad de FaceFusion en "claramente superior" en
        los casos difíciles: dientes nítidos al cantar y ojos vivos, sin tocar el
        resto de la cara. Más agresivo en modo Videos Musicales.
        """
        from ..utils import image as imageutil

        s = self.settings
        if (s.eye_preservation <= 0 and s.mouth_detail <= 0) or not faces:
            return frame
        kps_list = [f.kps for f in faces]
        return imageutil.enhance_regions(
            frame, kps_list,
            eye_strength=s.eye_preservation, mouth_strength=s.mouth_detail,
            mouth_open_boost=self._mouth_open_boost(),
        )

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
        self._loaded = False
