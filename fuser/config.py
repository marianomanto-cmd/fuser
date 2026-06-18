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
    ),
    EXPR_MUSIC_VIDEO: dict(
        # Modo musical inteligente: recomienda FaceFusion (alta calidad), muchas
        # referencias, post-procesado agresivo de boca/dientes y 2 pasadas (RAM).
        engine="facefusion",
        enhancer_model="codeformer", enhancer_blend=0.9, codeformer_fidelity=0.75,
        mask_mode=MASK_HULL,
        eye_preservation=0.8,        # ojos vivos
        mouth_detail=0.9,            # dientes nítidos al cantar
        color_match=True,            # iluminación cambiante de los escenarios
        temporal_smoothing=True, temporal_alpha=0.45,
        motion_adaptive=True,        # nada de "lag" en la boca al cantar
        two_pass_temporal=True,      # estabilidad sin lag (usa RAM)
        reference_count=6,           # 4-6 ángulos/expresiones
    ),
    EXPR_HIGH_EXPRESSION: dict(
        engine="facefusion",
        enhancer_model="codeformer", enhancer_blend=0.9, codeformer_fidelity=0.8,
        mask_mode=MASK_HULL, eye_preservation=0.85, mouth_detail=0.95,
        color_match=True, temporal_smoothing=True, temporal_alpha=0.4,
        motion_adaptive=True, two_pass_temporal=True, reference_count=6,
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

# Opciones específicas de FaceFusion expuestas en la UI.
FF_SWAPPER_CHOICES = [
    ("inswapper_128 (estándar)", "inswapper_128"),
    ("inswapper_128_fp16 (menos VRAM)", "inswapper_128_fp16"),
    ("hyperswap_1a_256 (mayor resolución)", "hyperswap_1a_256"),
    ("hyperswap_1b_256", "hyperswap_1b_256"),
]
FF_PIXEL_BOOST_CHOICES = [
    ("128x128 (rápido, menos VRAM)", "128x128"),
    ("256x256 (recomendado)", "256x256"),
    ("512x512 (máxima calidad, más VRAM)", "512x512"),
]


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
    ram_boost: bool = True                  # dimensiona buffers según la RAM libre

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
