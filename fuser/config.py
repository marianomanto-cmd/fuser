"""Configuración central de Fuser.

Contiene:
- Rutas del proyecto.
- ``Settings``: todos los parámetros del pipeline con valores por defecto.
- ``MEMORY_PRESETS``: presets de memoria (calidad vs velocidad vs VRAM).
- ``MODEL_REGISTRY``: registro de modelos ONNX (swappers y enhancers) con sus
  URLs de descarga y metadatos.

Este módulo NO importa dependencias pesadas (onnxruntime / insightface / torch),
de modo que puede ser usado por la UI para construir los controles sin coste.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional

# ----------------------------------------------------------------------------
# Rutas del proyecto
# ----------------------------------------------------------------------------
PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent

MODELS_DIR = Path(os.environ.get("FUSER_MODELS_DIR", PROJECT_ROOT / "models")).resolve()
OUTPUTS_DIR = Path(os.environ.get("FUSER_OUTPUT_DIR", PROJECT_ROOT / "outputs")).resolve()
TEMP_DIR = Path(os.environ.get("FUSER_TEMP_DIR", PROJECT_ROOT / "tmp")).resolve()
# InsightFace busca aquí los packs de detección (buffalo_l, etc.).
INSIGHTFACE_ROOT = Path(os.environ.get("FUSER_INSIGHTFACE_ROOT", MODELS_DIR)).resolve()


def ensure_dirs() -> None:
    """Crea las carpetas de trabajo si no existen."""
    for d in (MODELS_DIR, OUTPUTS_DIR, TEMP_DIR):
        d.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------------------
# Modos de memoria
# ----------------------------------------------------------------------------
# Nombres "máquina" de los modos (se usan como claves internas).
MODE_MAX_QUALITY = "max_quality"
MODE_BALANCED = "balanced"
MODE_LOW_VRAM = "low_vram"
MODE_EXTREME_LOW_VRAM = "extreme_low_vram"

# Etiquetas legibles para la UI -> clave interna.
MEMORY_MODE_LABELS: Dict[str, str] = {
    "Calidad máxima (usa toda la VRAM)": MODE_MAX_QUALITY,
    "Equilibrado (recomendado, 8 GB)": MODE_BALANCED,
    "Bajo VRAM (6 GB o menos)": MODE_LOW_VRAM,
    "VRAM mínima / usar más RAM (4 GB)": MODE_EXTREME_LOW_VRAM,
}

# Parámetros que cambian con cada modo de memoria.
#   gpu_mem_limit_gb : límite de arena de memoria de onnxruntime por sesión GPU.
#   det_size         : resolución del detector de caras (mayor = detecta más / más VRAM).
#   enhancer_device  : dónde corre el enhancer ("gpu" o "cpu" -> offload a RAM/CPU).
#   prefetch_frames  : nº de frames decodificados que mantenemos en RAM (buffer).
#   writer_queue     : nº de frames procesados en cola para escribir a disco.
#   det_batch        : nº de frames que se detectan por lote (reuso de la sesión).
MEMORY_PRESETS: Dict[str, dict] = {
    MODE_MAX_QUALITY: dict(
        gpu_mem_limit_gb=7.0,
        det_size=640,
        enhancer_device="gpu",
        prefetch_frames=96,
        writer_queue=96,
        det_batch=1,
    ),
    MODE_BALANCED: dict(
        gpu_mem_limit_gb=5.5,
        det_size=640,
        enhancer_device="gpu",
        prefetch_frames=64,
        writer_queue=64,
        det_batch=1,
    ),
    MODE_LOW_VRAM: dict(
        gpu_mem_limit_gb=3.8,
        det_size=512,
        enhancer_device="gpu",
        prefetch_frames=32,
        writer_queue=32,
        det_batch=1,
    ),
    MODE_EXTREME_LOW_VRAM: dict(
        gpu_mem_limit_gb=2.4,
        det_size=320,
        # El enhancer se mueve a CPU: libera VRAM a costa de velocidad,
        # aprovechando la RAM y los núcleos del sistema.
        enhancer_device="cpu",
        prefetch_frames=16,
        writer_queue=16,
        det_batch=1,
    ),
}


# ----------------------------------------------------------------------------
# Perfiles de RAM (cuánta RAM del sistema usar para buffers y 2 pasadas)
# ----------------------------------------------------------------------------
RAM_CONSERVATIVE = "conservador"
RAM_BALANCED = "equilibrado"
RAM_MAX = "maximo"

RAM_MODE_LABELS: Dict[str, str] = {
    "Conservador (poca RAM)": RAM_CONSERVATIVE,
    "Equilibrado (recomendado)": RAM_BALANCED,
    "Máximo aprovechamiento (32 GB+ RAM)": RAM_MAX,
}

# Fracción de la RAM LIBRE para los buffers de frames y para el tramo de 2
# pasadas, con topes de seguridad (nº de frames). El motor FaceFusion recibe un
# extra encima de estos valores (ver memory_manager).
RAM_FRACTIONS: Dict[str, dict] = {
    RAM_CONSERVATIVE: dict(buffer=0.15, chunk=0.30, buffer_cap=500, chunk_cap=2000),
    RAM_BALANCED: dict(buffer=0.30, chunk=0.45, buffer_cap=900, chunk_cap=4000),
    RAM_MAX: dict(buffer=0.50, chunk=0.70, buffer_cap=2200, chunk_cap=9000),
}


# ----------------------------------------------------------------------------
# Registro de modelos
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class ModelInfo:
    """Metadatos de un modelo ONNX descargable."""

    key: str
    label: str
    filename: str
    urls: List[str]
    kind: str  # "swapper" | "enhancer"
    size: int = 128  # resolución de entrada/salida nativa del modelo
    sha256: Optional[str] = None
    # Para enhancers: si el modelo tiene un segundo input de "peso/fidelidad"
    # (como CodeFormer) se indica aquí.
    has_weight_input: bool = False
    note: str = ""

    @property
    def path(self) -> Path:
        return MODELS_DIR / self.filename


# --- Swappers -----------------------------------------------------------------
# inswapper_128: el caballo de batalla de InsightFace. 128x128 por cara.
# Es el modelo one-shot (una sola imagen de referencia) más probado y compatible.
SWAPPER_MODELS: Dict[str, ModelInfo] = {
    "inswapper_128": ModelInfo(
        key="inswapper_128",
        label="InSwapper 128 (recomendado, máxima compatibilidad)",
        filename="inswapper_128.onnx",
        urls=[
            "https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx",
            "https://huggingface.co/xingren23/comfyflow-models/resolve/976de8449674de379b02c144d0b3cfa2b61482f2/insightface/inswapper_128.onnx",
            "https://github.com/facefusion/facefusion-assets/releases/download/models/inswapper_128.onnx",
        ],
        kind="swapper",
        size=128,
        note="Modelo de InsightFace. Uso de investigación / no comercial.",
    ),
    "inswapper_128_fp16": ModelInfo(
        key="inswapper_128_fp16",
        label="InSwapper 128 FP16 (menos VRAM, calidad casi idéntica)",
        filename="inswapper_128_fp16.onnx",
        urls=[
            "https://huggingface.co/facefusion/models-3.0.0/resolve/main/inswapper_128_fp16.onnx",
            "https://github.com/facefusion/facefusion-assets/releases/download/models/inswapper_128_fp16.onnx",
        ],
        kind="swapper",
        size=128,
        note="Variante FP16: ~50% menos peso en VRAM.",
    ),
}

# --- Enhancers (restauradores de cara) ---------------------------------------
# Todos corren como ONNX puro vía onnxruntime: evita el infierno de
# dependencias de basicsr/torch y mantiene la app ligera y "plug and play".
ENHANCER_MODELS: Dict[str, ModelInfo] = {
    "gfpgan_1.4": ModelInfo(
        key="gfpgan_1.4",
        label="GFPGAN 1.4 (rápido, natural — recomendado)",
        filename="gfpgan_1.4.onnx",
        urls=[
            "https://huggingface.co/facefusion/models-3.0.0/resolve/main/gfpgan_1.4.onnx",
            "https://github.com/facefusion/facefusion-assets/releases/download/models/gfpgan_1.4.onnx",
        ],
        kind="enhancer",
        size=512,
    ),
    "codeformer": ModelInfo(
        key="codeformer",
        label="CodeFormer (más nítido, control de fidelidad)",
        filename="codeformer.onnx",
        urls=[
            "https://huggingface.co/facefusion/models-3.0.0/resolve/main/codeformer.onnx",
            "https://github.com/facefusion/facefusion-assets/releases/download/models/codeformer.onnx",
        ],
        kind="enhancer",
        size=512,
        has_weight_input=True,
    ),
    "gpen_bfr_512": ModelInfo(
        key="gpen_bfr_512",
        label="GPEN BFR 512 (detalle alto)",
        filename="gpen_bfr_512.onnx",
        urls=[
            "https://huggingface.co/facefusion/models-3.0.0/resolve/main/gpen_bfr_512.onnx",
            "https://github.com/facefusion/facefusion-assets/releases/download/models/gpen_bfr_512.onnx",
        ],
        kind="enhancer",
        size=512,
    ),
    "restoreformer_plus_plus": ModelInfo(
        key="restoreformer_plus_plus",
        label="RestoreFormer++ (texturas finas)",
        filename="restoreformer_plus_plus.onnx",
        urls=[
            "https://huggingface.co/facefusion/models-3.0.0/resolve/main/restoreformer_plus_plus.onnx",
            "https://github.com/facefusion/facefusion-assets/releases/download/models/restoreformer_plus_plus.onnx",
        ],
        kind="enhancer",
        size=512,
    ),
}

# --- Face parsing (segmentación de regiones faciales) ------------------------
# Modelo opcional. Si está presente, habilita máscaras por región de máxima
# precisión (piel/ojos/cejas/nariz/boca/labios), clave para perfiles y boca.
PARSER_MODELS: Dict[str, ModelInfo] = {
    "face_parser_bisenet": ModelInfo(
        key="face_parser_bisenet",
        label="BiSeNet face parsing",
        filename="face_parsing_bisenet.onnx",
        urls=[
            "https://huggingface.co/facefusion/models-3.0.0/resolve/main/face_parser.onnx",
            "https://github.com/facefusion/facefusion-assets/releases/download/models/face_parser.onnx",
        ],
        kind="parser",
        size=512,
    ),
}

MODEL_REGISTRY: Dict[str, ModelInfo] = {
    **SWAPPER_MODELS,
    **ENHANCER_MODELS,
    **PARSER_MODELS,
}

# Etiquetas para los dropdowns de la UI.
SWAPPER_CHOICES = [(m.label, k) for k, m in SWAPPER_MODELS.items()]
ENHANCER_CHOICES = [("Ninguno (solo swap)", "none")] + [
    (m.label, k) for k, m in ENHANCER_MODELS.items()
]


# ----------------------------------------------------------------------------
# Selección de caras objetivo
# ----------------------------------------------------------------------------
FACE_SELECTOR_ALL = "all"            # swap a todas las caras del frame
FACE_SELECTOR_REFERENCE = "reference"  # solo a la cara parecida a una referencia
FACE_SELECTOR_LARGEST = "largest"    # solo a la cara más grande (primer plano)
FACE_SELECTOR_INDEX = "index"        # por posición (orden izquierda->derecha)

FACE_SELECTOR_LABELS: Dict[str, str] = {
    "Todas las caras": FACE_SELECTOR_ALL,
    "Cara más grande (primer plano)": FACE_SELECTOR_LARGEST,
    "Por referencia (elige una cara del vídeo)": FACE_SELECTOR_REFERENCE,
    "Por posición (índice)": FACE_SELECTOR_INDEX,
}


# ----------------------------------------------------------------------------
# Máscara de fusión (cómo se recorta la cara al pegarla)
# ----------------------------------------------------------------------------
MASK_HULL = "hull"        # casco convexo de los 106 landmarks (sigue el rostro)
MASK_ELLIPSE = "ellipse"  # elipse a partir de los 5 kps (fallback robusto)
MASK_BOX = "box"          # rectángulo suavizado (v1, el más simple)
MASK_PARSING = "parsing"  # segmentación facial (BiSeNet) — máxima precisión

MASK_MODE_LABELS: Dict[str, str] = {
    "Contorno facial (recomendado, ideal perfiles)": MASK_HULL,
    "Segmentación BiSeNet (máxima precisión)": MASK_PARSING,
    "Elipse (rápida y robusta)": MASK_ELLIPSE,
    "Rectángulo (básico)": MASK_BOX,
}


# ----------------------------------------------------------------------------
# Modo de expresión / caso de uso
# ----------------------------------------------------------------------------
EXPR_STANDARD = "standard"
EXPR_MUSIC_VIDEO = "music_video"
EXPR_HIGH_EXPRESSION = "high_expression"

EXPRESSION_MODE_LABELS: Dict[str, str] = {
    "Estándar": EXPR_STANDARD,
    "🎤 Videos musicales (caras cantando)": EXPR_MUSIC_VIDEO,
    "😮 Alta expresión (boca/ojos extremos)": EXPR_HIGH_EXPRESSION,
}

# Valores recomendados que la UI aplica al elegir un modo de expresión.
# Pensados para el caso de uso musical: ojos vivos, dientes nítidos al cantar,
# buen comportamiento en perfiles y mucho movimiento de cabeza.
# Nota: "facefusion"/"insightface" son los valores de ENGINE_* (definidos más
# abajo); se usan como literales aquí porque este dict se evalúa antes.
EXPRESSION_PRESETS: Dict[str, dict] = {
    EXPR_STANDARD: dict(
        engine="insightface",
        enhancer_model="gfpgan_1.4", enhancer_blend=0.8,
        mask_mode=MASK_HULL, eye_preservation=0.35, mouth_detail=0.35,
        color_match=False, temporal_smoothing=True, temporal_alpha=0.55,
        motion_adaptive=True, two_pass_temporal=False, reference_count=1,
        ram_mode=RAM_BALANCED,
    ),
    EXPR_MUSIC_VIDEO: dict(
        # Modo musical inteligente: recomienda FaceFusion (alta calidad), muchas
        # referencias, post-procesado agresivo de boca/dientes y 2 pasadas (RAM).
        engine="facefusion",
        # Mayor calidad one-shot: ghost_3 (256 px) en vez de inswapper (128 px).
        # Medido como el mejor en identidad de forma consistente (agent_tests), a 256
        # px nativos; CodeFormer pone la nitidez final. Sin entrenar nada. ¿Identidad
        # rara con TU cara? Probá hififace/simswap con "🔬 Comparar modelos".
        ff_swapper_model="ghost_3_256", ff_pixel_boost="256x256",
        enhancer_model="codeformer", enhancer_blend=0.9, codeformer_fidelity=0.5,
        ff_enhancer_weight=0.5,      # CodeFormer nativo hacia "detalle" -> dientes nítidos
        ff_detector_angles=(0, 90, 270),  # recupera caras inclinadas / cabeza atrás
        ff_detector_score=0.3,       # +recall en pitch extremo (mentón arriba)
        ff_landmarker_score=0.2,     # no descarta landmarks en cabeza-atrás (evita salto de máscara)
        ff_temporal_fallback=True,   # mantiene el realce aunque se pierda la detección unos frames
        mask_mode=MASK_HULL,
        eye_preservation=0.8,        # ojos vivos
        mouth_detail=0.9,            # dientes nítidos al cantar
        color_match=True,            # iluminación cambiante de los escenarios
        temporal_smoothing=True, temporal_alpha=0.45,
        motion_adaptive=True,        # nada de "lag" en la boca al cantar
        two_pass_temporal=True,      # estabilidad sin lag (usa RAM)
        reference_count=6,           # 4-6 ángulos/expresiones
        ram_mode=RAM_MAX,            # exprime la RAM (40 GB) para máxima estabilidad
    ),
    EXPR_HIGH_EXPRESSION: dict(
        engine="facefusion",
        ff_swapper_model="ghost_3_256", ff_pixel_boost="256x256",
        enhancer_model="codeformer", enhancer_blend=0.9, codeformer_fidelity=0.5,
        ff_enhancer_weight=0.5,
        ff_detector_angles=(0, 90, 180, 270),  # máxima recuperación de ángulos
        ff_detector_score=0.3, ff_landmarker_score=0.2, ff_temporal_fallback=True,
        mask_mode=MASK_HULL, eye_preservation=0.85, mouth_detail=0.95,
        color_match=True, temporal_smoothing=True, temporal_alpha=0.4,
        motion_adaptive=True, two_pass_temporal=True, reference_count=6,
        ram_mode=RAM_MAX,
    ),
}

# Opciones para el selector de cantidad de referencias.
REFERENCE_COUNT_CHOICES = [
    ("Auto (todas las que subas)", 0), ("1 imagen", 1),
    ("3 imágenes", 3), ("4 imágenes", 4), ("5 imágenes", 5),
    ("6 imágenes", 6), ("8 imágenes", 8),
]


# ----------------------------------------------------------------------------
# Motor de face swap
# ----------------------------------------------------------------------------
ENGINE_INSIGHTFACE = "insightface"
ENGINE_FACEFUSION = "facefusion"

ENGINE_LABELS: Dict[str, str] = {
    "InsightFace (Rápido)": ENGINE_INSIGHTFACE,
    "FaceFusion (Alta Calidad)": ENGINE_FACEFUSION,
}

ENGINE_INFO_MD = (
    "**🧠 Motor de Face Swap**  \n"
    "- **InsightFace (Rápido):** más rápido y consume **menos VRAM**. Buen resultado general; "
    "incluye el compositing por regiones de Fuser (ojos/boca/contorno) y el modo de 2 pasadas.  \n"
    "- **FaceFusion (Alta Calidad):** mejor en **boca abierta, dientes y perfiles laterales** "
    "(usa *pixel boost* y máscaras de oclusión/región), pero es **más lento y usa más VRAM**. "
    "Se **instala solo la primera vez** que lo eliges (o durante `setup`); no tienes que clonar nada."
)

# Modelos de swap disponibles en FaceFusion 3.1.1 (TODOS verificados como presentes
# en vendor/facefusion/.assets/models; todos ONNX, corren en DirectML). El swapper
# es la mayor palanca de calidad one-shot: inswapper_128 es la base (128 px), el
# resto son modelos de 256/512 px con mucho mejor identidad y detalle, SIN entrenar
# nada. Orden por idoneidad para el caso musical (caras cantando):
#   hififace  -> mejor identidad/forma de cara (recomendado, default del modo musical)
#   simswap   -> expresiones/boca más naturales
#   ghost     -> identidad + detalle (ghost_3 el más nuevo)
#   simswap_512 -> máxima resolución (más lento/VRAM)
#   inswapper -> base rápida/compatible
# (hyperswap NO existe en FaceFusion 3.1.1 -> se eliminó para no romper el motor.)
# Orden por idoneidad MEDIDA en agent_tests (cos sim ArcFace a la fuente + revisión
# visual, sobre clips reales de canto/primer plano): ghost_3 gana en identidad de
# forma consistente Y es 256 px; inswapper iguala identidad pero solo 128 px;
# hififace fue inconsistente; simswap suaviza expresiones. Ver agent_tests/RESULTADOS.md.
FF_SWAPPER_CHOICES = [
    ("ghost_3_256 (mejor identidad + detalle — recomendado)", "ghost_3_256"),
    ("hififace_256 (forma de cara, alternativa)", "hififace_unofficial_256"),
    ("simswap_256 (expresiones/boca más suaves)", "simswap_256"),
    ("ghost_2_256 (identidad + detalle)", "ghost_2_256"),
    ("simswap_512 (máxima resolución, más lento)", "simswap_unofficial_512"),
    ("uniface_256", "uniface_256"),
    ("blendswap_256", "blendswap_256"),
    ("inswapper_128 (rápido, base 128 px)", "inswapper_128"),
    ("inswapper_128_fp16 (menos VRAM)", "inswapper_128_fp16"),
]

FF_SWAPPER_LABELS = {k: lbl for lbl, k in FF_SWAPPER_CHOICES}


def short_model(key: str) -> str:
    """Nombre corto y legible de un modelo de swap (para galerías/etiquetas)."""
    return {
        "hififace_unofficial_256": "hififace", "simswap_unofficial_512": "simswap_512",
        "ghost_3_256": "ghost_3", "ghost_2_256": "ghost_2", "ghost_1_256": "ghost_1",
        "simswap_256": "simswap", "inswapper_128": "inswapper",
        "inswapper_128_fp16": "inswapper_fp16", "uniface_256": "uniface",
        "blendswap_256": "blendswap",
    }.get(key, key)

# Resolución NATIVA mínima por modelo: FaceFusion rechaza un pixel-boost menor que
# la resolución a la que se entrenó el modelo. inswapper admite 128; el resto exige
# >=256 (y simswap_512 exige 512). El motor usa esto para no pasarse ni quedarse corto.
FF_SWAPPER_NATIVE_RES = {
    "inswapper_128": 128,
    "inswapper_128_fp16": 128,
    "simswap_unofficial_512": 512,
}  # el resto (hififace/ghost/simswap_256/uniface/blendswap) -> 256
FF_PIXEL_BOOST_CHOICES = [
    ("128x128 (rápido, menos VRAM)", "128x128"),
    ("256x256 (recomendado)", "256x256"),
    ("512x512 (máxima calidad, más VRAM)", "512x512"),
]

# Perfil de memoria recomendado POR MOTOR (vista de alto nivel, legible).
#   ram_buffer_gb       : RAM objetivo para el buffer de frames.
#   two_pass            : preferencia de 2 pasadas por defecto.
#   max_temporal_frames : ventana mínima recomendada para el suavizado temporal.
MEMORY_PROFILES: Dict[str, dict] = {
    ENGINE_INSIGHTFACE: {"ram_buffer_gb": 4, "two_pass": False, "max_temporal_frames": 8},
    ENGINE_FACEFUSION: {"ram_buffer_gb": 8, "two_pass": True, "max_temporal_frames": 12},
}

# Config fina de memoria POR MOTOR (multiplicadores que aplica el memory_manager
# sobre el perfil de RAM elegido por el usuario): FaceFusion recibe buffers y
# tramos de 2 pasadas mayores y prefiere 2 pasadas por defecto.
ENGINE_MEMORY_CONFIG: Dict[str, dict] = {
    ENGINE_INSIGHTFACE: dict(
        buffer_mult=1.0, buffer_cap_mult=1.0, chunk_mult=1.0, chunk_cap_mult=1.0,
        prefers_two_pass=False,
    ),
    ENGINE_FACEFUSION: dict(
        buffer_mult=1.4, buffer_cap_mult=1.5, chunk_mult=1.25, chunk_cap_mult=1.5,
        prefers_two_pass=True,
    ),
}


# ----------------------------------------------------------------------------
# Settings del pipeline
# ----------------------------------------------------------------------------
@dataclass
class Settings:
    """Conjunto completo de parámetros de un trabajo de face swap."""

    # --- Motor de face swap ---
    engine: str = ENGINE_INSIGHTFACE
    ff_swapper_model: str = "inswapper_128"  # solo FaceFusion
    ff_pixel_boost: str = "256x256"          # solo FaceFusion (resolución del swap)
    ff_auto_install: bool = True             # auto-instalar FaceFusion al usarlo
    # FaceFusion: detección robusta para cabeza atrás / ángulos extremos.
    ff_detector_angles: tuple = (0,)         # rotaciones del detector (+ángulos = +recall en caras inclinadas)
    ff_detector_score: float = 0.5           # umbral de confianza del detector (bajo = +recall en pitch extremo)
    ff_landmarker_score: float = 0.5         # umbral de landmarks (bajo = no descarta la cara en cabeza-atrás)
    # FaceFusion: nitidez del enhancer nativo (CodeFormer). 0 = detalle/nítido, 1 = fiel a la entrada (borrosa).
    ff_enhancer_weight: float = 0.8
    ff_temporal_fallback: bool = True        # rellena huecos de detección reusando los últimos kps (anti-salto)

    # --- Modelos ---
    swapper_model: str = "inswapper_128"
    enhancer_model: str = "gfpgan_1.4"      # "none" para desactivar
    enhancer_blend: float = 0.8             # mezcla del enhancer (0..1)
    codeformer_fidelity: float = 0.7        # solo CodeFormer (0=detalle, 1=fidelidad)

    # --- Selección de caras ---
    face_selector: str = FACE_SELECTOR_ALL
    reference_face_index: int = 0           # para selector "index"
    reference_distance: float = 1.2         # umbral de distancia para "reference"
    source_average: bool = True             # promediar embeddings de varias fuentes

    # --- Multi-referencia (varias fotos de origen) ---
    reference_count: int = 0                # 0 = usar todas; si >0 elige las mejores N
    multi_ref_min_sim: float = 0.15         # rechazo de outliers (cos sim al promedio)
    frontal_weighting: bool = True          # pondera más las referencias frontales

    # --- Caso de uso / expresión ---
    expression_mode: str = EXPR_STANDARD

    # --- Calidad del swap ---
    face_opacity: float = 1.0               # 1=swap total, <1 deja ver el original
    mask_mode: str = MASK_HULL              # tipo de máscara de fusión
    mask_blur: float = 0.25                 # suavizado del borde de la máscara (0..1)
    mask_padding: float = 0.0               # recorte interior de la máscara (0..1)
    eye_preservation: float = 0.4           # realce/nitidez localizado en los ojos
    mouth_detail: float = 0.4               # realce localizado en boca/dientes
    skin_detail: float = 0.35               # reinyecta textura de piel del original (anti-plástico)
    mouth_enhancer: bool = True             # 2.º paso de enhancer (CodeFormer) en boca abierta
    mouth_enhancement_strength: float = 1.0  # multiplicador del enhancer localizado de boca
    use_mouth_pixel_boost: bool = True       # pase localizado de boca a 512 (FaceFusion)
    profile_blending_strength: float = 0.5   # suaviza/baja opacidad en perfiles laterales
    color_match: bool = False               # transferencia de color al original
    processing_resolution: int = 0          # 0 = nativa; si >0 limita el lado mayor

    # --- Estabilidad temporal ---
    temporal_smoothing: bool = True
    temporal_alpha: float = 0.55            # EMA de landmarks (0=sin memoria,1=congela)
    motion_adaptive: bool = True            # menos suavizado cuando hay movimiento rápido
    two_pass_temporal: bool = False         # 2 pasadas (estabilidad sin lag, usa RAM)

    # --- Memoria ---
    memory_mode: str = MODE_BALANCED
    gpu_mem_limit_gb: float = 0.0           # 0 = usar el del preset
    force_cpu: bool = False                 # forzar ejecución solo en CPU
    ram_mode: str = RAM_BALANCED            # perfil de RAM (conservador/equilibrado/máximo)

    # --- Salida ---
    keep_audio: bool = True
    keep_fps: bool = True
    output_quality: int = 18                # CRF de x264 (menor = mejor calidad)
    output_video_encoder: str = "libx264"

    # --- Detección ---
    det_size: int = 0                       # 0 = usar el del preset

    def resolved(self) -> "Settings":
        """Aplica el preset de memoria a los campos en modo "automático" (0)."""
        preset = MEMORY_PRESETS[self.memory_mode]
        out = Settings(**asdict(self))
        if out.gpu_mem_limit_gb <= 0:
            out.gpu_mem_limit_gb = preset["gpu_mem_limit_gb"]
        if out.det_size <= 0:
            out.det_size = preset["det_size"]
        return out

    @property
    def preset(self) -> dict:
        return MEMORY_PRESETS[self.memory_mode]

    def to_dict(self) -> dict:
        return asdict(self)
