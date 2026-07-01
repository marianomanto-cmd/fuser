"""Pestaña *Imagen → Vídeo* (Wan 2.2 14B I2V) de la UI de Gradio.

Es una sección **nueva e independiente** del face swap: sube una imagen, escribe
un prompt y genera un vídeo corto (~480p, ~6 s, con audio) usando un servidor
**ComfyUI local** como motor. Si ComfyUI no está arrancado o faltan modelos, la
UI lo dice con instrucciones claras (no rompe el resto de la app).
"""
from __future__ import annotations

import gradio as gr

from ..i2v import config as i2vcfg
from ..i2v.comfy_client import ComfyUINotAvailable
from ..i2v.service import I2VGenerationError, I2VService
from ..utils.logging import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _parse_resolution(value: str) -> tuple[int, int]:
    if (value or "").strip().lower() == "auto":
        return 0, 0   # 0 = el servicio deriva el tamaño del aspecto de la imagen
    try:
        w, h = value.lower().split("x")
        return int(w), int(h)
    except Exception:
        return 0, 0


def _build_settings(
    comfy_url, offload_preset, resolution, length_frames, n_clips, steps, cfg, shift, seed,
    sampler, scheduler, high_model, low_model, text_encoder, vae,
    audio_enabled, audio_prompt, audio_steps, audio_cfg, prompt_boost, negative,
) -> i2vcfg.I2VSettings:
    w, h = _parse_resolution(resolution)
    return i2vcfg.I2VSettings(
        comfy_url=(comfy_url or i2vcfg.DEFAULT_COMFY_URL).strip(),
        offload_preset=offload_preset,
        width=w, height=h,
        length_frames=i2vcfg.snap_length_4nplus1(length_frames),
        n_clips=int(n_clips),
        steps=int(steps), cfg=float(cfg), shift=float(shift), seed=int(seed),
        sampler=sampler, scheduler=scheduler,
        high_noise_model=(high_model or "").strip(),
        low_noise_model=(low_model or "").strip(),
        text_encoder=(text_encoder or "").strip(),
        vae=(vae or "").strip(),
        audio_enabled=bool(audio_enabled),
        audio_prompt=(audio_prompt or "").strip(),
        audio_steps=int(audio_steps), audio_cfg=float(audio_cfg),
        prompt_boost=bool(prompt_boost),
    )


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------
def _on_check(comfy_url, offload_preset, audio_enabled) -> str:
    s = i2vcfg.I2VSettings(
        comfy_url=(comfy_url or i2vcfg.DEFAULT_COMFY_URL).strip(),
        offload_preset=offload_preset, audio_enabled=bool(audio_enabled),
    )
    try:
        return I2VService(s).validate_setup()["markdown"]
    except Exception as exc:  # pragma: no cover
        return f"❌ No se pudo comprobar ComfyUI: {exc}"


def _on_generate(image, prompt, negative, *control_values, progress=gr.Progress()):
    if not image:
        raise gr.Error("Sube una imagen de entrada.")
    if not (prompt or "").strip():
        raise gr.Error("Escribe un prompt que describa la escena/movimiento.")
    settings = _build_settings(*control_values, negative)
    service = I2VService(settings)

    def cb(frac, msg=""):
        progress(frac, desc=msg)

    # En 8 GB el swap y ComfyUI comparten VRAM: liberamos la del swap ANTES de
    # generar para no saturar (evita el thrashing que ralentiza ~50×).
    try:
        from .interface import free_swap_vram
        free_swap_vram()
    except Exception:  # pragma: no cover
        pass

    try:
        progress(0.0, desc="Conectando con ComfyUI…")
        result = service.generate(image, prompt, negative, progress=cb)
    except ComfyUINotAvailable as exc:
        raise gr.Error(str(exc))
    except I2VGenerationError as exc:
        raise gr.Error(str(exc))
    except gr.Error:
        raise
    except Exception as exc:  # pragma: no cover
        log.exception("Error en Imagen → Vídeo")
        raise gr.Error(f"Error al generar: {exc}")

    audio_txt = "con audio 🔊" if result["has_audio"] else "sin audio"
    status = (f"✅ ¡Vídeo generado! {result['resolution']} · ~{result['seconds']} s · {audio_txt}.  "
              "Descárgalo, o pulsa **➕ Extender** para continuarlo.")
    if result.get("note"):
        status += f"  ⚠️ {result['note']}"
    return result["video"], result["video"], status, result["video"]


def _on_extend(base_video, prompt, negative, *control_values, progress=gr.Progress()):
    """Continúa el ÚLTIMO vídeo generado: su último frame arranca clips nuevos que
    se pegan detrás. ``base_video`` viene del gr.State con la última salida."""
    if not base_video:
        raise gr.Error("Primero generá un vídeo; después podés extenderlo.")
    if not (prompt or "").strip():
        raise gr.Error("Escribe un prompt para la continuación (describe el movimiento).")
    settings = _build_settings(*control_values, negative)
    service = I2VService(settings)

    def cb(frac, msg=""):
        progress(frac, desc=msg)

    try:
        from .interface import free_swap_vram
        free_swap_vram()
    except Exception:  # pragma: no cover
        pass

    try:
        progress(0.0, desc="Extendiendo el vídeo…")
        result = service.extend(base_video, prompt, negative, progress=cb)
    except ComfyUINotAvailable as exc:
        raise gr.Error(str(exc))
    except I2VGenerationError as exc:
        raise gr.Error(str(exc))
    except gr.Error:
        raise
    except Exception as exc:  # pragma: no cover
        log.exception("Error al extender Imagen → Vídeo")
        raise gr.Error(f"Error al extender: {exc}")

    audio_txt = "con audio 🔊" if result["has_audio"] else "sin audio"
    status = (f"✅ ¡Vídeo extendido! {result['resolution']} · ~{result['seconds']} s · {audio_txt}.  "
              "Podés seguir pulsando **➕ Extender** para alargarlo más.")
    if result.get("note"):
        status += f"  ⚠️ {result['note']}"
    return result["video"], result["video"], status, result["video"]


def _on_offload_change(offload_preset: str) -> str:
    preset = i2vcfg.OFFLOAD_PRESETS.get(offload_preset, {})
    flags = " ".join(preset.get("comfy_flags", []))
    note = preset.get("note", "")
    return (f"### ⚙️ Arranque recomendado de ComfyUI\n"
            f"```bash\npython main.py --listen 127.0.0.1 --port 8188 {flags}\n```\n"
            f"_{note}_")


# ---------------------------------------------------------------------------
# Construcción de la pestaña
# ---------------------------------------------------------------------------
INTRO = """
## 🎞️ Imagen → Vídeo (Wan 2.2)

Convierte **una imagen + un prompt** en un **vídeo corto** animado. El motor es
**ComfyUI** en local (Wan 2.2 con cuantización **GGUF**). Por defecto usa el modelo
**5B** (cabe entero en 8 GB → **~3 min por clip**); el **14B** da algo más de calidad
pero es mucho más lento.

> ⚠️ **Importante en 8 GB:** **cerrá/no proceses face-swap mientras generás vídeo** —
> comparten los 8 GB de VRAM y si los dos cargan modelos, el vídeo va **50× más lento**.
> Arrancá ComfyUI con `--reserve-vram 0.4` (NO `--lowvram`).
>
> ⚙️ **Requiere ComfyUI arrancado** (proceso aparte). Pulsá **🔌 Comprobar ComfyUI**.
> Describí **el MOVIMIENTO y la cámara** (la imagen ya define la apariencia), **en
> inglés** y con **UNA acción por clip**: *"she walks toward the camera at a relaxed
> pace"*. Para secuencias ("sonríe, luego saluda") **encadená clips** con `||` o el
> botón **➕ Extender**. El 5B no sigue bien: varias acciones seguidas, cámara +
> sujeto moviéndose a la vez, acción rápida, o varias personas interactuando — para
> eso usá el **14B** (preset de Offload). Guía: [`docs/IMAGE_TO_VIDEO.md`](https://github.com/marianomanto-cmd/fuser/blob/main/docs/IMAGE_TO_VIDEO.md).
>
> ⚠️ **Uso responsable:** solo con consentimiento, sin suplantar a personas reales.
"""


def build_i2v_tab() -> None:
    """Construye los controles de la pestaña (dentro de un ``gr.Tab`` ya abierto)."""
    gr.Markdown(INTRO)

    with gr.Row():
        with gr.Column(scale=3):
            image_in = gr.Image(label="🖼️ Imagen de entrada", type="filepath", height=320)
            prompt = gr.Textbox(
                label="📝 Prompt (SOLO el movimiento, no una escena nueva)", lines=3,
                placeholder="Ej.: she turns her head slowly and smiles, gentle breeze in her hair, "
                            "slow push-in  ·  (describe el MOVIMIENTO/cámara, no otra escena)",
                info="La imagen YA define la apariencia: describí solo cómo se MUEVE (y quién es). "
                     "EN INGLÉS rinde bastante mejor. Con varios clips encadenados podés dar un "
                     "prompt POR CLIP separando con ||  (ej.: she smiles || she waves).",
            )
            prompt_boost = gr.Checkbox(
                value=True, label="✨ Potenciar prompt (recomendado)",
                info="Añade descriptores de calidad/movimiento al estilo del 'prompt extension' "
                     "oficial de Wan. Mejora adherencia y estabilidad sin cambiar tu escena.",
            )
            negative = gr.Textbox(
                label="🚫 Prompt negativo", value=i2vcfg.WAN_DEFAULT_NEGATIVE, lines=2,
                info="Lo que NO quieres ver. El valor por defecto es el negativo oficial de Wan.",
            )
        with gr.Column(scale=2):
            status_panel = gr.Markdown("Pulsa **🔌 Comprobar ComfyUI** para ver el estado.")
            check_btn = gr.Button("🔌 Comprobar ComfyUI", size="sm")

    with gr.Row():
        offload_preset = gr.Dropdown(
            choices=list(i2vcfg.OFFLOAD_LABELS.items()), value=i2vcfg.OFFLOAD_BALANCED,
            label="🧠 Offload / VRAM", info="Cómo repartir el modelo entre VRAM y RAM en 8 GB.",
        )
        resolution = gr.Dropdown(
            choices=i2vcfg.RESOLUTION_CHOICES, value="auto", label="📐 Resolución",
            info="AUTO = usa el aspecto de TU imagen (evita que Wan recorte y cambie la "
                 "escena). Solo fijá una manual si sabés lo que hacés.",
        )
        duration = gr.Dropdown(
            choices=i2vcfg.DURATION_CHOICES, value=33, label="⏱️ Duración (por clip)",
            info="A 16 fps. Menos frames = mucho más rápido y menos VRAM. 33 ≈ 2 s. "
                 "En 8 GB, ≥8 s en una sola pasada suele petar: mejor encadená clips →",
        )
        n_clips = gr.Slider(
            1, i2vcfg.MAX_N_CLIPS, value=1, step=1, label="🔗 Clips a encadenar",
            info="1 = un solo clip. >1 encadena clips (último frame → arranque del "
                 "siguiente) para vídeos largos SIN petar el VAE. Ej.: 3 clips de 3 s ≈ 9 s. "
                 "Podés dar un prompt por clip separando con ||.",
        )

    comfy_cmd_md = gr.Markdown(_on_offload_change(i2vcfg.OFFLOAD_BALANCED))

    with gr.Accordion("🔊 Audio", open=True):
        with gr.Row():
            audio_enabled = gr.Checkbox(
                value=True, label="Generar audio (Stable Audio Open) y mezclarlo",
                info="Wan no genera sonido: esto crea una pista de audio (Stable Audio Open) que "
                     "matchea la duración y la mezcla con ffmpeg. Si falla, el vídeo sale sin sonido.",
            )
            audio_prompt = gr.Textbox(
                label="Prompt de audio (vacío = usa el del vídeo)", lines=1,
                placeholder="Ej.: upbeat pop music, female vocals, energetic",
                info="Describe la música/sonido. Si lo dejas vacío, se reutiliza el prompt del vídeo.",
            )
        with gr.Row():
            audio_steps = gr.Slider(20, 100, value=50, step=5, label="Pasos de audio",
                                    info="Más pasos = mejor audio, algo más lento.")
            audio_cfg = gr.Slider(1.0, 10.0, value=4.5, step=0.5, label="CFG de audio",
                                  info="Cuánto se ciñe al prompt de audio.")

    with gr.Accordion("🎛️ Calidad y muestreo", open=False):
        with gr.Row():
            steps = gr.Slider(8, 40, value=20, step=1, label="Pasos",
                              info="20 = plantilla oficial del 5B (mejor adherencia al prompt). "
                                   "16 = más rápido con poca pérdida.")
            cfg = gr.Slider(1.0, 8.0, value=4.5, step=0.5, label="CFG (guidance)",
                            info="4.5 recomendado (oficial 5B = 5.0). La identidad ya la ancla el "
                                 "latente: si IGNORA el prompt subí a 5; si se ve 'quemado'/"
                                 "sobresaturado bajá a 3.5.")
            shift = gr.Slider(1.0, 12.0, value=8.0, step=0.5, label="Shift (ModelSamplingSD3)",
                              info="8.0 va bien. Afecta al ritmo del movimiento.")
        with gr.Row():
            sampler = gr.Dropdown(choices=i2vcfg.SAMPLER_CHOICES, value="uni_pc", label="Sampler")
            scheduler = gr.Dropdown(choices=i2vcfg.SCHEDULER_CHOICES, value="simple", label="Scheduler")
            seed = gr.Number(value=-1, precision=0, label="Semilla (-1 = aleatoria)",
                             info="Fija una semilla para reproducir un resultado.")

    with gr.Accordion("⚙️ Avanzado: nombres de modelos y conexión", open=False):
        comfy_url = gr.Textbox(value=i2vcfg.DEFAULT_COMFY_URL, label="URL de ComfyUI",
                               info="Donde escucha tu ComfyUI. Por defecto http://127.0.0.1:8188.")
        gr.Markdown(f"_{i2vcfg.QUANT_NOTE}_")
        gr.Markdown("_El **modelo se elige con el preset de Offload** de arriba: Equilibrado→5B, "
                    "Offload máx / Rendimiento→14B (más fidelidad, más lento). Deja estos campos "
                    "**vacíos** para automático; solo escribe un nombre si quieres forzar otro GGUF._")
        with gr.Row():
            high_model = gr.Textbox(value="", placeholder="(vacío = automático según el preset)",
                                    label="Modelo GGUF — override (experto ALTO ruido)")
            low_model = gr.Textbox(value="", placeholder="(vacío = automático según el preset)",
                                   label="Modelo GGUF — override (experto BAJO ruido, solo 14B)")
        with gr.Row():
            text_encoder = gr.Textbox(value="umt5_xxl_fp8_e4m3fn_scaled.safetensors",
                                      label="Codificador de texto (UMT5)")
            vae = gr.Textbox(value="", placeholder="(vacío = automático según el preset)",
                             label="VAE — override (5B=2.2, 14B=2.1)")

    generate_btn = gr.Button("🎬 Generar vídeo", variant="primary")
    gen_status = gr.Markdown("")
    with gr.Row():
        video_out = gr.Video(label="🎉 Resultado")
        file_out = gr.File(label="⬇️ Descargar vídeo")

    # Guarda la ruta del último vídeo para poder EXTENDERLO.
    last_video = gr.State(value=None)
    with gr.Row():
        extend_btn = gr.Button("➕ Extender (continuar desde el último frame)",
                               variant="secondary")
    gr.Markdown(
        "_**Extender**: toma el ÚLTIMO frame del vídeo de arriba y genera su continuación "
        "(usa el prompt y los ajustes actuales, más **🔗 Clips a encadenar**) y la pega "
        "detrás. Pulsalo varias veces para alargar más. Puede haber leve **deriva de color** "
        "clip a clip (limitación conocida de Wan)._"
    )

    # ----- Orden EXACTO de controles (debe coincidir con _build_settings) -----
    controls = [
        comfy_url, offload_preset, resolution, duration, n_clips, steps, cfg, shift, seed,
        sampler, scheduler, high_model, low_model, text_encoder, vae,
        audio_enabled, audio_prompt, audio_steps, audio_cfg, prompt_boost,
    ]
    # Falla RUIDOSAMENTE al construir la UI si controls y _build_settings se
    # desalinean (evita mapear en silencio steps->cfg, etc. al reordenar uno solo).
    import inspect as _inspect
    assert len(controls) + 1 == len(_inspect.signature(_build_settings).parameters), (
        "i2v: 'controls' y '_build_settings' desalineados (revisa orden/número)."
    )

    # ----- Wiring -----
    check_btn.click(_on_check, inputs=[comfy_url, offload_preset, audio_enabled],
                    outputs=status_panel)
    offload_preset.change(_on_offload_change, inputs=offload_preset, outputs=comfy_cmd_md)
    generate_btn.click(
        _on_generate,
        inputs=[image_in, prompt, negative, *controls],
        outputs=[video_out, file_out, gen_status, last_video],
    )
    extend_btn.click(
        _on_extend,
        inputs=[last_video, prompt, negative, *controls],
        outputs=[video_out, file_out, gen_status, last_video],
    )

    gr.Markdown(
        "💡 **Consejos:** empieza con el preset **Equilibrado** y 480p. Si ComfyUI da "
        "*out of memory*, cambia a **Offload máximo a RAM** y/o usa un GGUF más pequeño "
        "(Q3_K_M). La primera generación es la más lenta (carga modelos). Espera **varios "
        "minutos** por clip en 8 GB — ver tiempos en `docs/IMAGE_TO_VIDEO.md`."
    )
