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
        self._mar_ema = None     # baseline dinámico del MAR (neutral por persona)
        self._loaded = False

    def reset_temporal(self) -> None:
        self._mar_ema = None

    def get_mouth_open_intensity(self, face) -> float:
        """Intensidad de apertura de boca (0..1) por landmarks + **umbral dinámico**.

        Calcula el ratio **altura/ancho** de la boca (distancia entre los labios
        superior↔inferior dividida por el ancho) y lo compara con un baseline EMA
        del propio sujeto (su boca 'neutral'), en vez de un umbral fijo. Devuelve
        0 si no hay 106 landmarks. Ver también ``is_mouth_open``.
        """
        from ..utils import image as imageutil

        mar = imageutil.mouth_aspect_ratio(face.kps, getattr(face, "landmark_2d_106", None))
        if mar is None:
            return 0.0
        if self._mar_ema is None:
            self._mar_ema = mar
        elif mar < self._mar_ema:
            # Adaptamos el baseline SOLO hacia abajo (hacia la boca neutral/cerrada).
            # Si la cantante sostiene una nota con la boca abierta, el baseline no
            # "persigue" la apertura y el realce de dientes no se apaga a mitad de nota.
            self._mar_ema += 0.04 * (mar - self._mar_ema)
        floor = float(np.clip(self._mar_ema, 0.12, 0.22))  # umbral dinámico por persona
        return float(np.clip((mar - floor) / 0.33, 0.0, 1.0))

    def is_mouth_open(self, face, threshold: float = 0.35) -> bool:
        """Booleano de boca abierta (intensidad por encima del umbral)."""
        return self.get_mouth_open_intensity(face) >= threshold

    def _dynamic_openness(self, face, frame, mouth_mask) -> float:
        """Intensidad de apertura; cae al contraste local si no hay 106 landmarks."""
        if getattr(face, "landmark_2d_106", None) is not None:
            return self.get_mouth_open_intensity(face)
        from ..utils import image as imageutil

        return imageutil._region_contrast(frame, mouth_mask)

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

    def _enhance_mouth_region(self, frame, face, intensity: float, profile: float):
        """Pasa CodeFormer por la cara alineada a 512 y pega SOLO la boca (dientes nítidos).

        El enhancer se ejecuta sobre la cara alineada completa a **512x512** (pixel
        boost localizado), pero se compone de vuelta únicamente en la región de la
        boca (máscara suave/gaussiana), con fuerza escalada por la apertura y por
        ``mouth_enhancement_strength``, y atenuada en perfiles fuertes (donde el
        alineado es menos fiable).
        """
        from insightface.utils import face_align

        from ..utils import image as imageutil

        s = self.settings
        aligned, M = face_align.norm_crop2(frame, face.kps, 512)
        enhanced = self._ensure_mouth_enhancer().run(aligned, fidelity=s.codeformer_fidelity)
        kps_aligned = imageutil.transform_points(face.kps, M)
        _, mouth = imageutil.frame_eye_mouth_masks(kps_aligned, (512, 512), self._mouth_open_boost())
        strength = float(np.clip(
            s.mouth_detail * intensity * s.mouth_enhancement_strength * (1.0 - 0.4 * profile),
            0.0, 1.0,
        ))
        return imageutil.paste_back_with_mask(frame, enhanced, M, mouth * strength, opacity=1.0)

    def _apply_profile_blending(self, out, original, face, profile: float):
        """En perfiles laterales, mezcla los **bordes** de la cara hacia el original.

        Reduce la opacidad efectiva del swap en mandíbula/oreja (zonas que más se
        deforman de lado) usando una máscara de cara suave: el núcleo mantiene el
        swap, el borde recupera el original. Intensidad por ``profile_blending_strength``.
        """
        from ..utils import image as imageutil

        strength = float(np.clip(profile * self.settings.profile_blending_strength, 0.0, 1.0))
        if strength <= 0.01:
            return out
        face_mask = imageutil.frame_face_mask(face.kps, out.shape)  # 1 en el núcleo, 0 fuera
        edge = (1.0 - face_mask)                                    # alto en los bordes
        a = np.clip(1.0 - strength * edge, 0.0, 1.0)[:, :, None]    # alpha del swap
        return (out.astype(np.float32) * a + original.astype(np.float32) * (1.0 - a)).astype(np.uint8)

    def _postprocess(self, frame: np.ndarray, faces, original=None) -> np.ndarray:
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
        music = s.expression_mode in (config.EXPR_MUSIC_VIDEO, config.EXPR_HIGH_EXPRESSION)
        if not faces:
            return frame

        out = frame
        boost = self._mouth_open_boost()
        for f in faces:
            kps = f.kps
            _, mouth_mask = imageutil.frame_eye_mouth_masks(kps, out.shape, boost)
            intensity = self._dynamic_openness(f, out, mouth_mask)   # 0..1
            # En modo musical bajamos el umbral: el pase de dientes (CodeFormer 512)
            # se mantiene activo en boca entreabierta y notas sostenidas, no solo en
            # boca muy abierta -> dientes nítidos durante toda la frase cantada.
            mouth_open = intensity >= (0.15 if music else 0.35)
            profile = float(np.clip((abs(self._yaw(f)) - 20.0) / 50.0, 0.0, 1.0))

            # 1) Realce de regiones: ojos siempre; boca escalada por la apertura.
            if s.eye_preservation > 0 or s.mouth_detail > 0:
                mouth_amt = s.mouth_detail * (0.5 + 0.9 * intensity)
                out = imageutil.enhance_regions(
                    out, [kps], eye_strength=s.eye_preservation, mouth_strength=mouth_amt,
                    mouth_open_boost=boost, adaptive_mouth=False,
                )
            # 2) Enhancer LOCALIZADO (CodeFormer a 512) solo si la boca está abierta.
            if mouth_open:
                out = self.enhance_mouth_region(out, f, intensity)
            # 3) Blending de PERFIL: en caras de lado, recupera el borde original.
            if original is not None and profile > 0.0:
                out = self._apply_profile_blending(out, original, f, profile)
            # 4) Anti-plástico: reinyecta la textura de piel del frame original
            #    dentro de la cara (quita el look ceroso del swap 128 + enhancer).
            if original is not None and s.skin_detail > 0:
                face_mask = imageutil.frame_face_mask(kps, out.shape)
                out = imageutil.transfer_skin_detail(out, original, face_mask, amount=s.skin_detail)
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

    def _init_ff_defaults(self) -> None:
        """Puebla el ``state_manager`` de FaceFusion con TODOS sus defaults (igual
        que hace su CLI), para que los items que el adaptador no setea no queden en
        ``None`` (lo que rompe el detector/landmarker/máscaras en FaceFusion 3.x).

        FaceFusion asume CWD = su repo para descubrir procesadores, así que lo
        fijamos temporalmente a ``vendor/facefusion`` solo durante la inicialización.
        """
        import os
        import sys

        from .facefusion_bootstrap import vendor_dir

        try:
            from facefusion import args as ff_args
            from facefusion import program as ff_program
            from facefusion import state_manager as ff_state
        except Exception as exc:  # pragma: no cover
            log.warning("No pude importar program/args de FaceFusion: %s", exc)
            return

        old_argv, old_cwd = sys.argv, os.getcwd()
        sys.argv = ["facefusion"]
        try:
            os.chdir(str(vendor_dir()))
            prog = ff_program.create_program()
            parsed = prog.parse_args(["headless-run"])
            ff_args.apply_args(vars(parsed), ff_state.init_item)
            log.info("Defaults de FaceFusion inicializados.")
        except SystemExit:  # pragma: no cover
            log.warning("FaceFusion program salió al parsear; sigo con overrides.")
        except Exception as exc:  # pragma: no cover
            log.warning("No pude inicializar defaults de FaceFusion: %s", exc)
        finally:
            sys.argv = old_argv
            os.chdir(old_cwd)

    # ----- configuración (integra con memory_manager + modo musical) ----------
    def _configure(self) -> None:
        s = self.settings
        music = s.expression_mode in (config.EXPR_MUSIC_VIDEO, config.EXPR_HIGH_EXPRESSION)

        # En DirectML añadimos "cpu" como fallback por-operación: CodeFormer tiene un
        # input float64 que DirectML no soporta; onnxruntime coloca ESA op en CPU (en
        # vez de reventar el proceso) y mantiene el swap y el resto en la GPU.
        ff_provider = self.mm.facefusion_provider()
        providers = [ff_provider, "cpu"] if ff_provider == "directml" else [ff_provider]
        # Ejecución / memoria
        self._set("execution_providers", providers)
        self._set("execution_device_id", "0")
        self._set("execution_thread_count", max(1, self.mm.info.cpu_count // 2))
        self._set("execution_queue_count", 1)
        self._set("download_providers", ["github", "huggingface"])
        # Offloading VRAM->RAM/CPU según el modo de memoria. En DirectML (8 GB) la
        # estrategia "tolerant" mantiene TODO en VRAM y la satura (CodeFormer no
        # entra -> crash duro = "connection lost"). Usamos al menos "moderate" para
        # que FaceFusion COMBINE VRAM+RAM (descarga modelos a RAM cuando hace falta).
        strategy = _VRAM_STRATEGY.get(s.memory_mode, "moderate")
        if self.mm.info.gpu_provider == "DmlExecutionProvider" and strategy == "tolerant":
            strategy = "moderate"
        self._set("video_memory_strategy", strategy)

        # Detector (yoloface va bien en perfiles); tamaño según det_size.
        self._set("face_detector_model", "yoloface")
        self._set("face_detector_size", f"{self.mm.det_size}x{self.mm.det_size}")
        # Ángulos de rotación del detector: en cabeza-atrás / caras inclinadas FF
        # reintenta la detección sobre el frame rotado y recupera caras que a 0° se
        # pierden (causa principal de "pierde la cara" en pitch extremo).
        self._set("face_detector_angles", list(s.ff_detector_angles) or [0])
        # Umbral de detección permisivo: conserva cajas de baja confianza (mentón arriba).
        self._set("face_detector_score", float(s.ff_detector_score) if music else 0.6)
        # Umbral de landmarks bajo: evita que FaceFusion DESCARTE la cara en
        # cabeza-atrás (cuando cae la confianza del landmarker) -> sin salto de máscara.
        self._set("face_landmarker_score", float(s.ff_landmarker_score))

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
        # Pixel boost = resolución interna del swap. AHORA es consciente del modelo:
        # cada swapper tiene una resolución NATIVA mínima (FaceFusion rechaza un boost
        # por debajo de ella). inswapper admite 128; hififace/ghost/simswap_256/uniface
        # exigen 256; simswap_512 exige 512. Nunca bajamos de la nativa.
        native = config.FF_SWAPPER_NATIVE_RES.get(s.ff_swapper_model, 256)
        if self.mm.info.gpu_provider == "DmlExecutionProvider":
            # En DirectML (8 GB) corremos a la resolución NATIVA del modelo: es toda la
            # calidad del swap sin saturar la VRAM (CodeFormer corre a 512 aparte y pone
            # la nitidez final). Antes forzábamos 128, lo que con un modelo de 256 es
            # inválido (rompía el motor) y desperdiciaba su resolución nativa.
            res = native
        else:
            # CUDA: deja subir por encima de la nativa (más VRAM disponible).
            want = 512 if music else int(str(s.ff_pixel_boost).split("x")[0])
            res = max(native, want)
        self._set("face_swapper_pixel_boost", f"{res}x{res}")

        # Enhancer: en modo musical, CodeFormer fuerte (mejor dientes/textura).
        enhancer_model = "codeformer" if music else (
            s.enhancer_model if s.enhancer_model and s.enhancer_model != "none" else "gfpgan_1.4"
        )
        self._set("face_enhancer_model", enhancer_model)
        self._set("face_enhancer_blend", int(round(max(s.enhancer_blend, 0.85 if music else 0.6) * 100)))
        # Peso del CodeFormer NATIVO de FaceFusion. FF usa 1.0 por defecto (= máxima
        # fidelidad a la entrada, que aquí es un swap de baja resolución -> dientes
        # borrosos). Bajarlo lleva a CodeFormer a RESTAURAR detalle desde su codebook
        # = dientes nítidos. (0 = detalle/nítido, 1 = fiel a la entrada borrosa.)
        self._set("face_enhancer_weight", float(np.clip(s.ff_enhancer_weight, 0.0, 1.0)))

        # ----- Máscaras: CLAVE para que el swap NO se "salga" de la cara -------
        # FaceFusion INTERSECTA las máscaras (numpy.minimum.reduce) y el PADDING solo
        # afecta a la CAJA (box). Los modelos que transfieren la FORMA de cara
        # (ghost/hififace/simswap/uniface/blendswap) dibujan una cara MÁS GRANDE que
        # la del objetivo; el parser (region) sigue ESA cara grande -> se "sale" en
        # mentón/cuello. Solución correcta: MANTENER la caja y RETRAERLA con padding;
        # como las máscaras se intersectan, la caja retraída recorta el sobrante del
        # parser. (Quitar la caja NO sirve: el padding deja de aplicarse. Investigado:
        # r/FF + lectura de facefusion/face_masker.py — ver research_cheatsheet.json.)
        shape_transfer = s.ff_swapper_model not in ("inswapper_128", "inswapper_128_fp16")
        # Modelos de máscara de máxima calidad (se descargan solos la 1ª vez).
        self._set("face_occluder_model", "xseg_1")            # oclusión pelo/manos/micro
        self._set("face_parser_model", "bisenet_resnet_34")   # parser de cara más fino
        if shape_transfer:
            self._set("face_mask_types", ["box", "occlusion", "region"])  # box + padding
            # (top, right, bottom, left) en %. Bottom alto retrae fuera del mentón/cuello;
            # laterales para que la mandíbula no invada; top suave (no cortar la frente).
            self._set("face_mask_padding", (2, 9, 18, 9))
            # Difuminado mayor: funde el cambio tonal del modelo en la piel objetivo.
            self._set("face_mask_blur", float(np.clip(max(s.mask_blur, 0.42), 0.0, 1.0)))
        else:
            if music or s.mask_mode == config.MASK_PARSING:
                self._set("face_mask_types", ["box", "occlusion", "region"])
            else:
                self._set("face_mask_types", ["box", "occlusion"])
            self._set("face_mask_padding", (0, 0, 0, 0))
            self._set("face_mask_blur", float(np.clip(s.mask_blur, 0.0, 1.0)))
        # Regiones faciales a INCLUIR. Mantenemos boca/labios (clave para el canto:
        # dientes y boca abierta). El recorte fino del contorno lo hacen el parser
        # (region) + occluder + el padding inferior, no la exclusión de la boca.
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
        self._init_ff_defaults()
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
        get_many = getattr(fa, "get_many_faces", None)
        if get_average is None:
            raise FaceFusionNotAvailable("face_analyser.get_average_face no encontrado")
        # FaceFusion 3.x: get_average_face espera CARAS ya detectadas (no imágenes).
        # Detectamos primero con el detector de FaceFusion y luego promediamos.
        if get_many is not None:
            faces = []
            for img in images:
                try:
                    faces.extend(get_many([img]) or [])
                except Exception as exc:  # pragma: no cover
                    log.warning("FaceFusion get_many_faces falló: %s", exc)
            n_used = len(faces)
            self._source_face = get_average(faces) if faces else None
        else:
            # Compatibilidad con APIs antiguas (get_average_face aceptaba imágenes).
            try:
                self._source_face = get_average(images)
            except TypeError:
                self._source_face = get_average(images, 0)
            n_used = len(images)
        if self._source_face is None:
            raise ValueError("FaceFusion no detectó ninguna cara en las imágenes fuente.")
        from ..core.face_store import SourceStats

        return SourceStats(n_input=len(images), n_used=n_used, mean_yaw=0.0, rejected=0)

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
        return self._postprocess(out, faces, original=frame)

    # ----- 2 pasadas: FF swap + post-procesado de regiones ESTABILIZADO --------
    # Pasada 1: detectamos (barato) y suavizamos los landmarks usando RAM.
    # Pasada 2: FaceFusion swapea y aplicamos el realce de boca/ojos con kps
    # suavizados -> dientes/ojos nítidos y SIN parpadeo entre frames.
    def supports_two_pass(self) -> bool:
        return True

    def supports_adaptive_mouth(self) -> bool:
        return True

    def prefers_two_pass(self) -> bool:
        return True

    def get_capabilities(self) -> dict:
        caps = super().get_capabilities()
        caps["high_res_region"] = True  # pixel boost + enhancer localizado de boca
        return caps

    def enhance_mouth_region(self, frame: np.ndarray, face, intensity: float = 1.0) -> np.ndarray:
        """Realce localizado de boca/dientes (CodeFormer dentro de una máscara suave).

        ``intensity`` (0..1) escala la fuerza (normalmente la apertura de boca).
        Controlado por ``mouth_enhancer`` + ``use_mouth_pixel_boost`` en config.
        """
        s = self.settings
        if not (s.mouth_enhancer and s.use_mouth_pixel_boost) or s.mouth_detail <= 0:
            return frame
        try:
            profile = float(np.clip((abs(self._yaw(face)) - 20.0) / 50.0, 0.0, 1.0))
            return self._enhance_mouth_region(frame, face, intensity, profile)
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
        return self._postprocess(out, targets, original=frame)

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
