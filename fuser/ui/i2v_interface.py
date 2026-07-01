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
    try:
        w, h = value.lower().split("x")
        return int(w), int(h)
    except Exception:
        return 832, 480


def _build_settings(
    comfy_url, offload_preset, resolution, length_frames, steps, cfg, shift, seed,
    sampler, scheduler, high_model, low_model, text_encoder, vae,
    audio_enabled, audio_prompt, audio_steps, audio_cfg, negative,
) -> i2vcfg.I2VSettings:
    w, h = _parse_resolution(resolution)
    return i2vcfg.I2VSettings(
        comfy_url=(comfy_url or i2vcfg.DEFAULT_COMFY_URL).strip(),
        offload_preset=offload_preset,
        width=w, height=h,
        length_frames=int(length_frames),
        steps=int(steps), cfg=float(cfg), shift=float(shift), seed=int(seed),
        sampler=sampler, scheduler=scheduler,
        high_noise_model=(high_model or "").strip(),
        low_noise_model=(low_model or "").strip(),
        text_encoder=(text_encoder or "").strip(),
        vae=(vae or "").strip(),
        audio_enabled=bool(audio_enabled),
        audio_prompt=(audio_prompt or "").strip(),
        audio_steps=int(audio_steps), audio_cfg=float(audio_cfg),
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

    audio_txt = "con audio 🔊" if result["has_audio"] else "sin audio (no se pudo generar)"
    status = (f"✅ ¡Vídeo generado! {result['resolution']} · ~{result['seconds']} s · {audio_txt}.  "
              "Descárgalo abajo.")
    return result["video"], result["video"], status


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
> Describí **el MOVIMIENTO y la cámara** (la imagen ya define la apariencia): *"slow
> push-in, gentle breeze, subtle breathing"*. Guía: [`docs/IMAGE_TO_VIDEO.md`](https://github.com/marianomanto-cmd/fuser/blob/main/docs/IMAGE_TO_VIDEO.md).
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
                label="📝 Prompt (qué pasa en el vídeo)", lines=3,
                placeholder="Ej.: a woman singing on a neon-lit stage, slow camera push-in, "
                            "hair gently moving, cinematic lighting",
                info="Describe el MOVIMIENTO y la escena. En inglés suele rendir mejor.",
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
            choices=i2vcfg.RESOLUTION_CHOICES, value="640x480", label="📐 Resolución",
            info="Wan prefiere lados múltiplos de 16. En 8 GB empezá pequeño (640×480).",
        )
        duration = gr.Dropdown(
            choices=i2vcfg.DURATION_CHOICES, value=33, label="⏱️ Duración",
            info="A 16 fps. Menos frames = mucho más rápido y menos VRAM. 33 ≈ 2 s.",
        )

    comfy_cmd_md = gr.Markdown(_on_offload_change(i2vcfg.OFFLOAD_BALANCED))

    with gr.Accordion("🔊 Audio", open=True):
        with gr.Row():
            audio_enabled = gr.Checkbox(
                value=False, label="Generar audio (Stable Audio Open) y mezclarlo",
                info="OFF por defecto (Stable Audio no está instalado en esta máquina). "
                     "Wan no genera sonido; esto crearía una pista aparte y la mezclaría con ffmpeg.",
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
            steps = gr.Slider(8, 40, value=16, step=1, label="Pasos",
                              info="5B: 16 va bien. Menos = más rápido, más artefactos.")
            cfg = gr.Slider(1.0, 8.0, value=5.0, step=0.5, label="CFG (guidance)",
                            info="5B sin LoRA: ~5. (El 14B-Lightning usa cfg 1.)")
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
        gr.Markdown("_Default = modelo **5B** (`wan22_ti2v_5b`, rápido). Para el 14B (lento, "
                    "más calidad) cambiá el **preset de Offload** a uno 14B y poné los GGUF "
                    "`Wan2.2-I2V-A14B-{High,Low}Noise-Q3_K_S.gguf` + VAE `wan_2.1_vae`._")
        with gr.Row():
            high_model = gr.Textbox(value="Wan2.2-TI2V-5B-Q4_K_M.gguf",
                                    label="Modelo (GGUF) — 5B usa el mismo en ambos")
            low_model = gr.Textbox(value="Wan2.2-TI2V-5B-Q4_K_M.gguf",
                                   label="Modelo experto BAJO ruido (solo 14B)")
        with gr.Row():
            text_encoder = gr.Textbox(value="umt5_xxl_fp8_e4m3fn_scaled.safetensors",
                                      label="Codificador de texto (UMT5)")
            vae = gr.Textbox(value="wan2.2_vae.safetensors", label="VAE (5B=2.2, 14B=2.1)")

    generate_btn = gr.Button("🎬 Generar vídeo", variant="primary")
    gen_status = gr.Markdown("")
    with gr.Row():
        video_out = gr.Video(label="🎉 Resultado")
        file_out = gr.File(label="⬇️ Descargar vídeo")

    # ----- Orden EXACTO de controles (debe coincidir con _build_settings) -----
    controls = [
        comfy_url, offload_preset, resolution, duration, steps, cfg, shift, seed,
        sampler, scheduler, high_model, low_model, text_encoder, vae,
        audio_enabled, audio_prompt, audio_steps, audio_cfg,
    ]

    # ----- Wiring -----
    check_btn.click(_on_check, inputs=[comfy_url, offload_preset, audio_enabled],
                    outputs=status_panel)
    offload_preset.change(_on_offload_change, inputs=offload_preset, outputs=comfy_cmd_md)
    generate_btn.click(
        _on_generate,
        inputs=[image_in, prompt, negative, *controls],
        outputs=[video_out, file_out, gen_status],
    )

    gr.Markdown(
        "💡 **Consejos:** empieza con el preset **Equilibrado** y 480p. Si ComfyUI da "
        "*out of memory*, cambia a **Offload máximo a RAM** y/o usa un GGUF más pequeño "
        "(Q3_K_M). La primera generación es la más lenta (carga modelos). Espera **varios "
        "minutos** por clip en 8 GB — ver tiempos en `docs/IMAGE_TO_VIDEO.md`."
    )
