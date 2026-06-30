"""Carga y *parcheo* de workflows de ComfyUI (formato API).

Filosofía: en vez de construir el grafo a mano (frágil), cargamos una
**plantilla** JSON y la parcheamos **buscando nodos por ``class_type``** (no por
id). Así el mismo código funciona con nuestras plantillas y también con un
workflow que el usuario exporte desde ComfyUI ("Save (API Format)").

Cada función de parcheo es **tolerante**: si un tipo de nodo no aparece, lo
ignora (registra un aviso) en vez de fallar.
"""
from __future__ import annotations

import copy
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..utils.logging import get_logger
from .config import WORKFLOWS_DIR

log = get_logger(__name__)

Graph = Dict[str, dict]


# ----------------------------------------------------------------------------
# Carga
# ----------------------------------------------------------------------------
def load_workflow(name_or_path: str) -> Graph:
    """Carga una plantilla por nombre (``workflows/<name>.json``) o por ruta."""
    p = Path(name_or_path)
    if not p.exists():
        cand = WORKFLOWS_DIR / f"{name_or_path}.json"
        if cand.exists():
            p = cand
    if not p.exists():
        raise FileNotFoundError(f"No encuentro el workflow: {name_or_path}")
    graph = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(graph, dict):
        raise ValueError(
            f"{p.name} no está en 'API format'. En ComfyUI activa el modo dev y usa "
            "'Save (API Format)'."
        )
    # Descarta entradas que no sean nodos (p.ej. "_comment"): ComfyUI las
    # rechazaría al encolar.
    clean = {k: v for k, v in graph.items()
             if isinstance(v, dict) and "class_type" in v}
    if not clean:
        raise ValueError(
            f"{p.name} no contiene nodos en 'API format' (cada nodo necesita "
            "'class_type'). Re-expórtalo con 'Save (API Format)'."
        )
    return clean


# ----------------------------------------------------------------------------
# Búsqueda
# ----------------------------------------------------------------------------
def find_by_class(graph: Graph, *class_types: str) -> List[Tuple[str, dict]]:
    wanted = set(class_types)
    return [(nid, n) for nid, n in graph.items()
            if isinstance(n, dict) and n.get("class_type") in wanted]


def find_one(graph: Graph, *class_types: str) -> Optional[Tuple[str, dict]]:
    hits = find_by_class(graph, *class_types)
    return hits[0] if hits else None


def _ref_node(graph: Graph, ref) -> Optional[Tuple[str, dict]]:
    """Resuelve una referencia de input ``["<id>", idx]`` al nodo apuntado."""
    if isinstance(ref, list) and ref and isinstance(ref[0], (str, int)):
        nid = str(ref[0])
        node = graph.get(nid)
        if node:
            return nid, node
    return None


def _trace_back(graph: Graph, start_ref, target_classes: set, max_hops: int = 8
                ) -> Optional[Tuple[str, dict]]:
    """Sigue una cadena de inputs hacia atrás hasta dar con un ``class_type``."""
    seen = set()
    frontier = [start_ref]
    hops = 0
    while frontier and hops < max_hops:
        nxt = []
        for ref in frontier:
            resolved = _ref_node(graph, ref)
            if not resolved:
                continue
            nid, node = resolved
            if nid in seen:
                continue
            seen.add(nid)
            if node.get("class_type") in target_classes:
                return nid, node
            for v in (node.get("inputs") or {}).values():
                if isinstance(v, list):
                    nxt.append(v)
        frontier = nxt
        hops += 1
    return None


def _resolve_pos_neg(graph: Graph) -> Tuple[Optional[str], Optional[str]]:
    """Identifica los ``CLIPTextEncode`` positivo y negativo.

    Estrategia robusta: parte del nodo de muestreo/condicionado (WanImageToVideo
    o KSampler) y traza sus inputs ``positive`` / ``negative`` hacia el
    CLIPTextEncode. *Fallback*: por título (_meta) o por orden.
    """
    encodes = find_by_class(graph, "CLIPTextEncode")
    if len(encodes) == 1:
        return encodes[0][0], None
    if not encodes:
        return None, None

    anchor = find_one(graph, "WanImageToVideo", "KSamplerAdvanced", "KSampler", "SamplerCustomAdvanced")
    if anchor:
        inputs = anchor[1].get("inputs", {})
        pos = _trace_back(graph, inputs.get("positive"), {"CLIPTextEncode"})
        neg = _trace_back(graph, inputs.get("negative"), {"CLIPTextEncode"})
        pid = pos[0] if pos else None
        nid = neg[0] if neg else None
        if pid or nid:
            return pid, nid

    # Fallback por título.
    pos_id = neg_id = None
    for nid, node in encodes:
        title = str((node.get("_meta") or {}).get("title", "")).lower()
        if any(k in title for k in ("neg", "negativ")):
            neg_id = nid
        elif any(k in title for k in ("pos", "positiv", "prompt")):
            pos_id = nid
    if pos_id or neg_id:
        return pos_id or encodes[0][0], neg_id
    # Último recurso: el primero positivo, el segundo negativo.
    return encodes[0][0], encodes[1][0]


def _set(node: dict, key: str, value) -> None:
    node.setdefault("inputs", {})[key] = value


# ----------------------------------------------------------------------------
# Parcheo: Imagen -> Vídeo (Wan 2.2)
# ----------------------------------------------------------------------------
def patch_i2v(graph: Graph, *, image: str, positive: str, negative: str,
              width: int, height: int, length: int, fps: int,
              steps: int, cfg: float, seed: int, sampler: str, scheduler: str,
              shift: float, high_model: Optional[str] = None,
              low_model: Optional[str] = None, virtual_vram_gb: float = 0.0,
              filename_prefix: str = "fuser_i2v") -> Graph:
    """Devuelve una COPIA del grafo con todos los parámetros aplicados."""
    g = copy.deepcopy(graph)
    if seed is None or seed < 0:
        seed = random.randint(0, 2**31 - 1)

    # Imagen de entrada.
    for _, node in find_by_class(g, "LoadImage"):
        _set(node, "image", image)

    # Prompts.
    pos_id, neg_id = _resolve_pos_neg(g)
    if pos_id:
        _set(g[pos_id], "text", positive)
    if neg_id:
        _set(g[neg_id], "text", negative)

    # Tamaño / longitud del vídeo.
    for _, node in find_by_class(g, "WanImageToVideo"):
        _set(node, "width", int(width))
        _set(node, "height", int(height))
        _set(node, "length", int(length))
        _set(node, "batch_size", 1)

    # Shift del muestreo.
    for _, node in find_by_class(g, "ModelSamplingSD3"):
        _set(node, "shift", float(shift))

    # Cargadores GGUF (vídeo) — mapea high/low siguiendo qué sampler los usa.
    _patch_unet_loaders(g, high_model, low_model, virtual_vram_gb)

    # Samplers high/low.
    half = max(1, int(steps) // 2)
    for _, node in find_by_class(g, "KSamplerAdvanced"):
        ins = node.get("inputs", {})
        _set(node, "steps", int(steps))
        _set(node, "cfg", float(cfg))
        _set(node, "sampler_name", sampler)
        _set(node, "scheduler", scheduler)
        _set(node, "noise_seed", int(seed))
        # El experto de ALTO ruido es el que añade ruido (add_noise == enable).
        if str(ins.get("add_noise", "enable")) == "enable":
            _set(node, "start_at_step", 0)
            _set(node, "end_at_step", half)
        else:
            _set(node, "start_at_step", half)
            _set(node, "end_at_step", 10000)
    # KSampler simple (por si la plantilla usa uno solo).
    for _, node in find_by_class(g, "KSampler"):
        _set(node, "steps", int(steps))
        _set(node, "cfg", float(cfg))
        _set(node, "sampler_name", sampler)
        _set(node, "scheduler", scheduler)
        _set(node, "seed", int(seed))

    # Salida de vídeo.
    _patch_video_output(g, fps, filename_prefix)
    return g


def _patch_unet_loaders(g: Graph, high_model: Optional[str],
                        low_model: Optional[str], virtual_vram_gb: float) -> None:
    loader_classes = ("UnetLoaderGGUF", "UnetLoaderGGUFAdvanced",
                      "UnetLoaderGGUFDisTorch2MultiGPU",
                      "UnetLoaderGGUFAdvancedDisTorch2MultiGPU", "UNETLoader")
    loaders = find_by_class(g, *loader_classes)
    # virtual_vram para los cargadores DisTorch2.
    if virtual_vram_gb > 0:
        for _, node in loaders:
            if "DisTorch" in node.get("class_type", ""):
                _set(node, "virtual_vram_gb", float(virtual_vram_gb))

    if not (high_model or low_model) or not loaders:
        return

    # ¿Qué cargador alimenta al sampler de alto ruido y cuál al de bajo?
    role_of_loader: Dict[str, str] = {}
    for _, sampler in find_by_class(g, "KSamplerAdvanced"):
        ins = sampler.get("inputs", {})
        role = "high" if str(ins.get("add_noise", "enable")) == "enable" else "low"
        traced = _trace_back(g, ins.get("model"), set(loader_classes))
        if traced:
            role_of_loader[traced[0]] = role

    field = "unet_name"
    for nid, node in loaders:
        role = role_of_loader.get(nid)
        if role == "high" and high_model:
            _set(node, field, high_model)
        elif role == "low" and low_model:
            _set(node, field, low_model)
    # Si no se pudo trazar (1 solo cargador o grafo raro), usa el orden del fichero.
    if not role_of_loader and loaders:
        if high_model:
            _set(loaders[0][1], field, high_model)
        if low_model and len(loaders) > 1:
            _set(loaders[1][1], field, low_model)


def _patch_video_output(g: Graph, fps: int, filename_prefix: str) -> None:
    for _, node in find_by_class(g, "VHS_VideoCombine"):
        _set(node, "frame_rate", int(fps))
        _set(node, "filename_prefix", filename_prefix)
        _set(node, "save_output", True)
    for _, node in find_by_class(g, "CreateVideo"):
        _set(node, "fps", int(fps))
    for _, node in find_by_class(g, "SaveVideo", "SaveAnimatedWEBP", "SaveWEBM"):
        _set(node, "filename_prefix", filename_prefix)


# ----------------------------------------------------------------------------
# Parcheo: Texto -> Audio (Stable Audio Open)
# ----------------------------------------------------------------------------
def patch_audio(graph: Graph, *, prompt: str, negative: str, seconds: float,
                seed: int, steps: Optional[int] = None, cfg: Optional[float] = None,
                filename_prefix: str = "fuser_audio") -> Graph:
    g = copy.deepcopy(graph)
    if seed is None or seed < 0:
        seed = random.randint(0, 2**31 - 1)

    pos_id, neg_id = _resolve_pos_neg(g)
    if pos_id:
        _set(g[pos_id], "text", prompt)
    if neg_id:
        _set(g[neg_id], "text", negative)

    for _, node in find_by_class(g, "EmptyLatentAudio"):
        _set(node, "seconds", float(seconds))

    for _, node in find_by_class(g, "KSampler", "KSamplerAdvanced"):
        if "noise_seed" in (node.get("inputs") or {}):
            _set(node, "noise_seed", int(seed))
        else:
            _set(node, "seed", int(seed))
        if steps is not None:
            _set(node, "steps", int(steps))
        if cfg is not None:
            _set(node, "cfg", float(cfg))

    for _, node in find_by_class(g, "SaveAudio", "SaveAudioMP3", "SaveAudioOpus"):
        _set(node, "filename_prefix", filename_prefix)
    return g
