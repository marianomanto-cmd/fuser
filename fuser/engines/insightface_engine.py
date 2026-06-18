"""Motor InsightFace (inswapper_128) — el pipeline propio, refactorizado.

Rápido y ligero. Hace todo el compositing por regiones de Fuser:
- swap a 128 px (InSwapper) + realce con enhancer ONNX a 512 px,
- realce DIRIGIDO de ojos y boca (regiones derivadas de los kps),
- máscara de **contorno** (casco de 106 landmarks) o segmentación BiSeNet,
- multi-referencia robusta y suavizado temporal (adaptativo / 2 pasadas).
"""
from __future__ import annotations

from typing import List, Optional

import cv2
import numpy as np

from .. import config
from ..models.downloader import ensure_model
from ..utils import image as imageutil
from ..utils.logging import get_logger
from .base import BaseFaceSwapper

log = get_logger(__name__)


class InsightFaceSwapper(BaseFaceSwapper):
    name = config.ENGINE_INSIGHTFACE
    display_name = "InsightFace (Rápido)"
    description = "Rápido y ligero, menos VRAM. Compositing por regiones de Fuser."

    def __init__(self, settings, memory_manager):
        super().__init__(settings, memory_manager)
        self.analyser = None
        self.swapper = None
        self.enhancer = None
        self.enhancer_info = None
        self.parser = None
        self.store = None
        self.smoother = None
        self._loaded = False

    # ----- ciclo de vida -------------------------------------------------------
    def load(self, progress=None) -> None:
        if self._loaded:
            return
        from ..core.face_store import FaceStore
        from ..core.temporal import TemporalSmoother
        from ..models.face_analyser import FaceAnalyser
        from ..models.face_enhancer import FaceEnhancer
        from ..models.face_swapper import FaceSwapper

        s = self.settings
        if progress:
            progress(0.05, "Cargando detector de caras…")
        self.analyser = FaceAnalyser(self.mm.analyser_providers(), self.mm.ctx_id(), self.mm.det_size)
        self.analyser.load()

        if progress:
            progress(0.3, "Preparando el modelo de swap…")
        swap_path = ensure_model(s.swapper_model, progress)
        self.swapper = FaceSwapper(swap_path, self.mm.swapper_providers())
        self.swapper.load()

        if s.enhancer_model and s.enhancer_model != "none":
            if progress:
                progress(0.6, "Preparando el enhancer…")
            info = config.ENHANCER_MODELS[s.enhancer_model]
            enh_path = ensure_model(s.enhancer_model, progress)
            on_cpu = (not self.mm.use_gpu) or s.preset.get("enhancer_device") == "cpu"
            self.enhancer = FaceEnhancer(enh_path, self.mm.enhancer_providers(), info,
                                         self.mm.session_options(on_cpu))
            self.enhancer.load()
            self.enhancer_info = info

        self.store = FaceStore(self.analyser, s)
        self.smoother = (
            TemporalSmoother(s.temporal_alpha, motion_adaptive=s.motion_adaptive)
            if s.temporal_smoothing else None
        )
        self._loaded = True
        if progress:
            progress(1.0, "Motor InsightFace listo.")

    @property
    def loaded(self) -> bool:
        return self._loaded

    def update_runtime(self, settings) -> None:
        from ..core.temporal import TemporalSmoother

        self.settings = settings
        if self.store is not None:
            self.store.settings = settings
        self.smoother = (
            TemporalSmoother(settings.temporal_alpha, motion_adaptive=settings.motion_adaptive)
            if settings.temporal_smoothing else None
        )

    # ----- fuente / referencia -------------------------------------------------
    def prepare_source(self, images: List[np.ndarray]):
        if not self._loaded:
            self.load()
        return self.store.set_source(images)

    def set_reference(self, frame: np.ndarray, index: int = 0) -> bool:
        if not self._loaded:
            self.load()
        return self.store.set_reference(frame, index)

    @property
    def has_source(self) -> bool:
        return self.store is not None and self.store.source_face is not None

    # ----- compositing ---------------------------------------------------------
    @staticmethod
    def _aligned_crop(frame: np.ndarray, affine, size: int) -> np.ndarray:
        return cv2.warpAffine(frame, affine, (size, size), borderMode=cv2.BORDER_REPLICATE)

    def _mouth_open_boost(self) -> float:
        m = self.settings.expression_mode
        if m == config.EXPR_HIGH_EXPRESSION:
            return 2.0
        if m == config.EXPR_MUSIC_VIDEO:
            return 1.5
        return 1.0

    def _ensure_parser(self):
        if self.parser is not None:
            return self.parser
        from ..models.face_parser import FaceParser

        info = config.PARSER_MODELS["face_parser_bisenet"]
        path = ensure_model("face_parser_bisenet")
        on_cpu = not self.mm.use_gpu
        self.parser = FaceParser(path, self.mm.analyser_providers(), info, self.mm.session_options(on_cpu))
        self.parser.load()
        return self.parser

    def _profile_factor(self, target_face) -> float:
        """0 = frontal, 1 = perfil fuerte (a partir del yaw estimado por kps)."""
        try:
            yaw = abs(self.analyser.estimate_yaw(target_face))
        except Exception:
            return 0.0
        return float(np.clip((yaw - 20.0) / 50.0, 0.0, 1.0))

    def _build_face_mask(self, frame, target_face, affine, size):
        s = self.settings
        mode = s.mask_mode
        # En perfiles, suaviza más el borde de la máscara (InsightFace es menos
        # fiable de lado) → evita costuras en mandíbula/oreja.
        profile = self._profile_factor(target_face)
        blur = float(np.clip(s.mask_blur + 0.18 * profile, 0.0, 0.9))

        if mode == config.MASK_PARSING:
            try:
                parser = self._ensure_parser()
                aff512 = affine if size == 512 else imageutil.scale_affine(affine, 512.0 / size)
                target_aligned = self._aligned_crop(frame, aff512, 512)
                masks = parser.region_masks(target_aligned)
                fm = cv2.resize(masks["face"], (size, size), interpolation=cv2.INTER_LINEAR)
                fm = imageutil._feather(fm, blur * 0.5)
                if fm.max() > 0.2:
                    return np.clip(fm, 0.0, 1.0)
            except Exception as exc:  # pragma: no cover
                log.warning("Face parsing no disponible (%s); uso contorno de landmarks.", exc)

        if mode in (config.MASK_HULL, config.MASK_PARSING):
            lmk = getattr(target_face, "landmark_2d_106", None)
            if lmk is not None:
                pts = imageutil.transform_points(lmk, affine)
                return imageutil.convex_hull_mask(pts, size, blur=blur, padding=s.mask_padding)
            kps_a = imageutil.transform_points(target_face.kps, affine)
            return imageutil.ellipse_face_mask(kps_a, size, blur=blur, padding=s.mask_padding)
        if mode == config.MASK_ELLIPSE:
            kps_a = imageutil.transform_points(target_face.kps, affine)
            return imageutil.ellipse_face_mask(kps_a, size, blur=blur, padding=s.mask_padding)
        return imageutil.build_soft_mask(size, size, blur=blur, padding=s.mask_padding)

    def _swap_one(self, frame: np.ndarray, target_face) -> np.ndarray:
        s = self.settings
        bgr_fake, affine = self.swapper.swap_raw(frame, target_face, self.store.source_face)
        in_size = max(1, self.swapper.input_size)

        if self.enhancer is not None:
            up = cv2.resize(bgr_fake, (512, 512), interpolation=cv2.INTER_LANCZOS4)
            enhanced = self.enhancer.run(up, fidelity=s.codeformer_fidelity)
            blend = float(np.clip(s.enhancer_blend, 0.0, 1.0))
            face = cv2.addWeighted(enhanced, blend, up, 1.0 - blend, 0)
            affine = imageutil.scale_affine(affine, 512.0 / in_size)
            size = 512
        else:
            face = bgr_fake
            size = in_size

        kps_aligned = imageutil.transform_points(target_face.kps, affine)
        eyes_m, mouth_m = imageutil.eye_mouth_region_masks(
            kps_aligned, size, mouth_open_boost=self._mouth_open_boost()
        )
        if s.eye_preservation > 0:
            face = imageutil.apply_local_detail(face, eyes_m, amount=s.eye_preservation * 1.2)
        if s.mouth_detail > 0:
            face = imageutil.apply_local_detail(face, mouth_m, amount=s.mouth_detail * 1.2)

        if s.color_match:
            ref = self._aligned_crop(frame, affine, size)
            face = imageutil.apply_color_transfer(face, ref)

        mask = self._build_face_mask(frame, target_face, affine, size)
        return imageutil.paste_back_with_mask(frame, face, affine, mask, opacity=s.face_opacity)

    # ----- API del motor -------------------------------------------------------
    def detect(self, frame: np.ndarray) -> List:
        return self.analyser.get_faces(frame)

    def select_targets(self, faces: List) -> List:
        return self.store.select_targets(faces)

    def render(self, frame: np.ndarray, targets: List) -> np.ndarray:
        for tf in targets:
            frame = self._swap_one(frame, tf)
        return frame

    def reset_temporal(self) -> None:
        if self.smoother:
            self.smoother.reset()

    def supports_two_pass(self) -> bool:
        return True

    def supports_adaptive_mouth(self) -> bool:
        return True

    def enhance_mouth_region(self, frame: np.ndarray, face, openness: float = 1.0) -> np.ndarray:
        """Realce localizado de boca (versión básica por unsharp, paridad de interfaz)."""
        if self.settings.mouth_detail <= 0:
            return frame
        _, mouth = imageutil.frame_eye_mouth_masks(face.kps, frame.shape, self._mouth_open_boost())
        return imageutil.apply_local_detail(
            frame, mouth, amount=self.settings.mouth_detail * 1.4 * float(np.clip(openness, 0, 1))
        )

    def get_memory_usage(self) -> dict:
        return {
            "engine": self.name,
            "loaded": self._loaded,
            "providers": "cuda" if self.mm.use_gpu else "cpu",
            "swapper_loaded": self.swapper is not None and getattr(self.swapper, "loaded", False),
            "enhancer_loaded": self.enhancer is not None,
            "parser_loaded": self.parser is not None,
        }

    def process_frame(self, frame: np.ndarray, use_smoothing: bool = True) -> np.ndarray:
        faces = self.detect(frame)
        if use_smoothing and self.smoother is not None:
            if faces:
                faces = self.smoother.smooth(faces, frame.shape)
            else:
                # Detección fallida en este frame: reutiliza la última cara conocida
                # (evita el parpadeo de aparecer/desaparecer el swap).
                faces = self.smoother.predict()
        targets = self.select_targets(faces)
        return self.render(frame, targets)

    def unload(self) -> None:
        for m in (self.analyser, self.swapper, self.enhancer, self.parser):
            if m is not None and hasattr(m, "unload"):
                m.unload()
        self._loaded = False
