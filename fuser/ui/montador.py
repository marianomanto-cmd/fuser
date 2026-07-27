"""🎬 Montador — pestaña dedicada a montar modelos .dfm entrenados sobre videos.

Reescrita DE CERO (2026-07-26) reemplazando el montaje que vivía embebido en la
pestaña Deep Swap. Principios de diseño:

- **Un solo camino**: 1 video o N videos son la misma cosa (una lista de
  trabajos). Sin generadores anidados ni delegación — un bucle plano.
- **Por partes**: cada video se corta en tramos exactos (ffmpeg segment) y cada
  parte terminada aparece al instante para controlar la calidad. Al final se
  unen y se reincrusta el audio original.
- **Reanudable de verdad**: cada parte se renderiza a un temporal y se publica
  atómicamente al terminar; al volver a montar, las partes válidas se saltean.
  La carpeta de trabajo lleva la huella del video + tamaño de parte: otro video
  u otro tamaño JAMÁS continúan un trabajo viejo.
- **Detener guarda**: la parada es cooperativa (bandera que el pipeline mira una
  vez por frame); lo hecho se une y queda en outputs/ como _PARCIAL.
- **La GPU primero se pregunta**: con un entrenamiento DFL vivo la VRAM está
  tomada y cargar modelos MATA el proceso (DirectML no lanza excepción) — la
  guarda corta antes, con mensaje claro.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from collections import deque
from pathlib import Path
from typing import List, Optional

import cv2
import gradio as gr

from .. import config
from ..core import face_library
from ..core import pipeline as pipeline_core
from ..utils import video as videoutil
from ..utils.logging import get_logger
from ..utils.system import ffmpeg_path
from .shared import NO_DFM, dfm_choices, ensure_gpu_libre, fmt_eta, get_pipeline

log = get_logger(__name__)

# Reintentos por video en una cola: el que falla va al final y se reintenta.
MAX_INTENTOS = 2

PARTES = [
    ("15 segundos (ver resultados muy seguido)", 15),
    ("30 segundos (equilibrado)", 30),
    ("60 segundos (menos uniones)", 60),
    ("Video entero (sin partes)", 0),
]

# Preset propio del Montador (no vive en EXPRESSION_PRESETS): igual que Máxima
# Identidad pero con GFPGAN en vez de CodeFormer. CodeFormer RESTAURA desde su
# codebook —de ahí la nitidez y también el look "retocado"—; GFPGAN es más
# conservador y suele leerse más natural, a costa de menos detalle en dientes.
# Con un .dfm la identidad ya está horneada, así que el enhancer solo decide el
# acabado: vale la pena tenerlo como A/B.
PRESET_GFPGAN = "max_identity_gfpgan"

PRESETS = [
    ("🎯 Máxima Identidad — para el MONTAJE FINAL", config.EXPR_MAXIDENTITY),
    ("🌿 Máxima Identidad + GFPGAN — acabado más natural, menos 'retocado'", PRESET_GFPGAN),
    ("⚡ Estándar — para TESTEOS rápidos (menos calidad)", config.EXPR_STANDARD),
]


# ---------------------------------------------------------------------------
# Ajustes del montaje
# ---------------------------------------------------------------------------
def _settings(dfm_id: str, preset: str, piel: float = 0.0,
              enhancer: Optional[float] = None,
              borde: str = config.EDGE_SUELTO) -> config.Settings:
    """Settings del montaje con .dfm: preset de pegado + VRAM/RAM al máximo.

    La IDENTIDAD vive en el modelo entrenado; el preset solo decide cómo se
    PEGA la cara (máscara, enhancer, estabilidad, QC).

    ``piel`` (0..1) = ``skin_detail``: reinyecta la ALTA FRECUENCIA del frame
    original (poros, grano) dentro de la cara. Es el anti-plástico de la app.
    No arrastra identidad —la identidad vive en la forma, que es baja y media
    frecuencia— así que subirlo no tira el parecido hacia la cara del video;
    solo devuelve textura de piel real bajo esa luz. Default 0 = como antes.

    ``enhancer`` (0..1) pisa ``enhancer_blend``: cuánto se mezcla la
    restauración de CodeFormer. Bajarlo deja pasar más del modelo crudo (menos
    "retocado", pero más suave si el modelo está poco entrenado).

    ``borde`` = ``ff_edge_fit``: cuánto se recorta el swap al contorno real de
    la cara. El preset de Máxima Identidad quita el parser a propósito (para que
    pase la geometría del modelo) y el precio es que NADA acota el pegado: se
    desborda al cuello, al pelo y bajo el mentón. Subirlo devuelve ese recorte.
    """
    s = config.Settings()
    # El preset GFPGAN deriva de Máxima Identidad: mismo modelo, misma máscara,
    # misma geometría — solo cambia el enhancer.
    es_gfpgan = preset == PRESET_GFPGAN
    clave = config.EXPR_MAXIDENTITY if es_gfpgan else preset
    base = config.EXPRESSION_PRESETS.get(clave, config.EXPRESSION_PRESETS[config.EXPR_MAXIDENTITY])
    for k, v in base.items():
        setattr(s, k, v)
    # expression_mode tiene que ser un modo REAL: el motor lo consulta para
    # decidir umbrales y flujo. Un valor inventado lo mandaría a los defaults.
    s.expression_mode = clave
    if es_gfpgan:
        s.enhancer_model = "gfpgan_1.4"
        # GFPGAN no tiene el "weight" de CodeFormer (no restaura desde codebook),
        # así que la fidelidad se controla solo con la mezcla.
        s.enhancer_blend = 0.8
    s.ff_deep_swapper_model = dfm_id
    s.chain_shape_then_texture = False   # el .dfm ya es forma+textura
    s.skin_detail = float(max(0.0, min(1.0, piel or 0.0)))
    if enhancer is not None:
        s.enhancer_blend = float(max(0.0, min(1.0, enhancer)))
    s.ff_edge_fit = borde if borde in config.EDGE_FIT_PRESETS else config.EDGE_SUELTO
    # El blending de perfiles mezcla mandíbula/oreja de vuelta hacia el VIDEO:
    # con un .dfm eso deshace la geometría entrenada. La máscara box ya funde.
    s.profile_blending_strength = 0.0
    s.ram_mode = config.RAM_MAX
    s.memory_mode = config.MODE_MAX_QUALITY
    s.gpu_mem_limit_gb = config.MEMORY_PRESETS[config.MODE_MAX_QUALITY]["gpu_mem_limit_gb"]
    s.output_quality = 12                # CRF x264 (menor = mejor)
    return s


# ---------------------------------------------------------------------------
# Trabajo por partes (corte, validación, unión)
# ---------------------------------------------------------------------------
def _fp(video: str) -> str:
    """Huella corta del archivo (ruta+tamaño+mtime). No lee el contenido."""
    p = Path(video)
    try:
        st = p.stat()
        raw = f"{p.resolve()}|{st.st_size}|{int(st.st_mtime)}"
    except OSError:
        raw = str(p)
    return hashlib.sha1(raw.encode("utf-8", "surrogateescape")).hexdigest()[:10]


def _dur(video: str) -> float:
    """Duración en segundos (los ilegibles van al final de la cola)."""
    try:
        info = videoutil.probe(video)
        return info.frame_count / (info.fps or 25.0)
    except Exception:
        return float("inf")


def _job_dir(dfm_id: str, video: str, secs: int) -> Path:
    """Carpeta de trabajo de UN montaje. La clave incluye la huella del video y
    el tamaño de parte: dos trabajos distintos nunca comparten carpeta."""
    stem = Path(video).stem[:40]
    tag = f"c{int(secs)}" if int(secs) > 0 else "entero"
    slug = face_library._slug(dfm_id.split("/")[-1])
    return config.OUTPUTS_DIR / f"montaje_{slug}_{stem}_{_fp(video)}_{tag}"


def _parte_ok(path: Path) -> bool:
    """True solo si la parte es un mp4 TERMINADO y con frames legibles.

    Un mp4 truncado por una caída pesa mucho más que cualquier umbral de bytes:
    hay que abrirlo y exigir frames de verdad, o la reanudación deja huecos.
    """
    try:
        if not path.is_file() or path.stat().st_size <= 1000:
            return False
        cap = cv2.VideoCapture(str(path))
        try:
            return bool(cap.isOpened()) and int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) > 0
        finally:
            cap.release()
    except Exception:
        return False


def _cortar(video: str, secs: int, out_dir: Path) -> List[Path]:
    """Corta el video en partes de ``secs`` con UNA pasada de ffmpeg (segment).

    Reencoda con keyframes forzados cada ``secs`` (el muxer segment solo corta
    en keyframes existentes: sin esto, pedir 3 s daba partes de 8). Reusa el
    corte SOLO si el manifiesto coincide (misma huella de video, mismo secs).
    """
    import subprocess

    ff = ffmpeg_path()
    if not ff:
        raise gr.Error("FFmpeg no disponible.")
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = out_dir / "corte.json"
    want = {"huella": _fp(video), "secs": int(secs)}
    existentes = sorted(out_dir.glob("in_*.mp4"))
    if existentes:
        try:
            if json.loads(manifest.read_text("utf-8")) == want:
                return existentes
        except Exception:
            pass
        for viejo in existentes:        # corte de OTRO trabajo: no sirve
            viejo.unlink(missing_ok=True)
    cmd = [ff, "-y", "-hide_banner", "-loglevel", "error", "-i", video,
           "-c:v", "libx264", "-crf", "16", "-preset", "veryfast",
           "-force_key_frames", f"expr:gte(t,n_forced*{int(secs)})",
           "-c:a", "aac", "-reset_timestamps", "1",
           "-f", "segment", "-segment_time", str(int(secs)),
           "-segment_time_delta", "0.05",
           str(out_dir / "in_%04d.mp4")]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=3600)
    except subprocess.CalledProcessError as exc:
        err = (exc.stderr or b"").decode("utf-8", "ignore")[-300:]
        raise gr.Error(f"FFmpeg falló al cortar el video: {err}")
    partes = sorted(out_dir.glob("in_*.mp4"))
    if not partes:
        raise gr.Error("El corte no produjo partes (¿video ilegible?).")
    manifest.write_text(json.dumps(want), "utf-8")
    return partes


def _unir(partes: List[str], out_stem: str, audio_de: str) -> Optional[str]:
    """Une las partes en outputs/<stem>.mp4 y reincrusta el audio original.

    Si el mux de audio sale bien, el intermedio MUDO se borra: si no, cada
    montaje dejaba en outputs/ una copia completa sin audio (gigas de basura).
    """
    if not partes:
        return None
    final = str(config.OUTPUTS_DIR / f"{out_stem}.mp4")
    if not videoutil.concat_videos(partes, final, drop_seam=False, crf=12):
        return None
    try:
        con_audio = str(config.OUTPUTS_DIR / f"{out_stem}_audio.mp4")
        if videoutil.mux_audio(final, audio_de, con_audio):
            Path(final).unlink(missing_ok=True)
            return con_audio
    except Exception:
        log.warning("No pude reincrustar el audio original", exc_info=True)
    return final


def _limpiar_parciales(stem: str) -> None:
    """Al completar un video, sus _PARCIAL de paradas anteriores ya no sirven."""
    for p in config.OUTPUTS_DIR.glob(f"{stem}_montaje_PARCIAL*.mp4"):
        try:
            p.unlink()
        except OSError:
            pass


# ---------------------------------------------------------------------------
# El montaje (1 video o cola: mismo bucle plano)
# ---------------------------------------------------------------------------
def _tag_ajustes(preset: str, piel: float, enhancer: Optional[float], borde: str) -> str:
    """Firma legible de los ajustes que CAMBIAN la imagen de salida.

    Nombra la carpeta de partes renderizadas: dos configuraciones distintas
    nunca comparten partes, así la reanudación no mezcla un video con tramos
    hechos con otros ajustes.
    """
    tag = f"{preset}_{borde}_p{int(round((piel or 0.0) * 100)):03d}"
    if enhancer is not None:
        tag += f"_e{int(round(enhancer * 100)):03d}"
    return tag


def _norm_files(files) -> List[str]:
    out = []
    for f in (files or []):
        p = f if isinstance(f, str) else getattr(f, "name", None)
        if p:
            out.append(p)
    # dedupe conservando el orden: el mismo archivo dos veces en la cola solo
    # duplicaría trabajo y entregas
    return list(dict.fromkeys(out))


def montar(dfm_id, files, preset, secs, piel=0.0, enhancer=None,
           borde=config.EDGE_SUELTO, progress=gr.Progress()):
    """Monta el modelo .dfm en 1..N videos, por partes y con vista en vivo.

    Generator de Gradio: emite ``(última parte, archivos, estado)``. Los videos
    van del más corto al más largo; el que falla va al final y se reintenta una
    vez; detener guarda lo hecho (el video en curso queda como _PARCIAL).
    """
    if not dfm_id or dfm_id == NO_DFM:
        raise gr.Error("Elegí un modelo .dfm (se crean en la pestaña 🧬 Deep Swap).")
    videos = _norm_files(files)
    if not videos:
        raise gr.Error("Subí al menos un video donde montar la cara.")
    ensure_gpu_libre("el montaje")
    pipeline_core.clear_stop()

    secs = int(secs or 0)
    videos = sorted(videos, key=_dur)            # victorias rápidas primero
    nv = len(videos)

    progress(0.01, desc="Cargando el modelo entrenado…")
    pipeline = get_pipeline(_settings(dfm_id, preset, piel, enhancer, borde),
                            progress=lambda f, m="": progress(0.01 + f * 0.05, desc=m))

    listos: List[str] = []       # finales terminados (cola)
    entregas: List[str] = []     # todo lo mostrado en "descargas" (partes + finales)
    descartados: List[str] = []
    trabajos = deque((v, 1) for v in videos)
    yield None, [], (f"▶️ **{nv} video(s)**, partes de {secs} s. Procesando…" if secs
                     else f"▶️ **{nv} video(s)**, sin partes. Procesando…")

    hechos = 0
    while trabajos:
        if pipeline_core.stop_requested():
            break
        video, intento = trabajos.popleft()
        nombre = Path(video).name
        etiqueta = f"[{hechos + 1}/{nv}] **{nombre}**" if nv > 1 else f"**{nombre}**"
        base = 0.06 + 0.94 * hechos / nv
        span = 0.94 / nv

        # --- preparar las partes de ESTE video --------------------------------
        try:
            if secs <= 0:
                partes_in = [Path(video)]
            else:
                progress(base, desc=f"{etiqueta} · cortando en partes…")
                partes_in = _cortar(video, secs, _job_dir(dfm_id, video, secs) / "entrada")
        except Exception as exc:
            log.exception("Montador: no pude cortar %s", nombre)
            # "hechos" cuenta videos que SALEN de la cola (completados o
            # descartados), nunca reintentos: si no, la barra pasa de 100% y la
            # etiqueta muestra [4/3] (hallazgo de la revisión).
            if intento < MAX_INTENTOS:
                trabajos.append((video, intento + 1))
                yield None, list(entregas), f"⚠️ {etiqueta} no se pudo preparar ({exc}); lo reintento al final."
            else:
                hechos += 1
                descartados.append(nombre)
                yield None, list(entregas), f"❌ {etiqueta} descartado ({exc}). Sigo…"
            continue

        # Partes renderizadas SEPARADAS POR TODOS los ajustes que cambian la
        # imagen (hallazgo de la revisión, extendido): sin esto, partes de un
        # test se "reanudaban" dentro de un montaje con otros ajustes y el video
        # final salía mezclado. El corte de ENTRADA sí se comparte: no depende
        # de ninguno de estos.
        listas_dir = _job_dir(dfm_id, video, secs) / f"listas_{_tag_ajustes(preset, piel, enhancer, borde)}"
        listas_dir.mkdir(parents=True, exist_ok=True)
        n = len(partes_in)
        hechas: List[str] = []
        parcial: Optional[str] = None
        fallo: Optional[Exception] = None
        t_gastado, t_computadas = 0.0, 0

        for i, parte in enumerate(partes_in):
            destino = listas_dir / f"parte_{i + 1:04d}.mp4"
            if _parte_ok(destino):
                hechas.append(str(destino))       # reanudación
                yield str(destino), list(entregas) + hechas, \
                    f"{etiqueta} · ⏩ parte **{i + 1}/{n}** ya estaba lista (reanudando)…"
                continue
            if pipeline_core.stop_requested():
                break

            def cb(frac, msg="", _i=i):
                progress(base + span * (0.05 + 0.85 * (_i + frac) / n),
                         desc=f"{etiqueta} · parte {_i + 1}/{n} · {msg}")

            # render a temporal + publicación atómica: una parte a medias jamás
            # se confunde con una terminada
            tmp = listas_dir / f"parte_{i + 1:04d}.enproceso.mp4"
            tmp.unlink(missing_ok=True)
            t0 = time.time()
            try:
                pipeline.process_video(str(parte), output_path=str(tmp), progress=cb)
            except Exception as exc:
                tmp.unlink(missing_ok=True)
                fallo = exc
                log.exception("Montador: falló la parte %d/%d de %s", i + 1, n, nombre)
                break
            t_gastado += time.time() - t0

            if pipeline.interrupted:              # botón detener a mitad de parte
                parcial = str(tmp.with_name(f"parte_{i + 1:04d}.parcial.mp4"))
                os.replace(tmp, parcial)
                if not _parte_ok(Path(parcial)):  # cortó tan temprano que no sirve
                    Path(parcial).unlink(missing_ok=True)
                    parcial = None
                break

            os.replace(tmp, destino)
            destino.with_name(f"parte_{i + 1:04d}.parcial.mp4").unlink(missing_ok=True)
            hechas.append(str(destino))
            t_computadas += 1
            por_parte = t_gastado / max(1, t_computadas)
            yield str(destino), list(entregas) + hechas, (
                f"{etiqueta} · ✅ parte **{i + 1}/{n}** lista — mirala arriba.\n\n"
                f"⏱️ {fmt_eta(por_parte)} por parte · faltan **{fmt_eta(por_parte * (n - i - 1))}** "
                f"({n - i - 1} partes). *Detener guarda lo hecho; al volver, continúa desde acá.*")

        # --- cierre de ESTE video ---------------------------------------------
        if fallo is not None:
            if intento < MAX_INTENTOS:
                trabajos.append((video, intento + 1))
                yield (hechas[-1] if hechas else None), list(entregas) + hechas, (
                    f"⚠️ {etiqueta}: falló una parte ({fallo}). Las {len(hechas)} anteriores "
                    f"quedaron; lo reintento al final de la cola…")
            else:
                hechos += 1
                descartados.append(nombre)
                yield (hechas[-1] if hechas else None), list(entregas) + hechas, (
                    f"❌ {etiqueta} falló {MAX_INTENTOS} veces; lo descarto. Sigo…")
            continue

        # stem con huella corta: dos videos de la cola con el MISMO nombre (de
        # carpetas distintas) escribirían el mismo outputs/<stem>_montaje.mp4 y
        # el segundo pisaría al primero en silencio.
        stem = f"{Path(video).stem[:40]}_{_fp(video)[:6]}"
        if pipeline_core.stop_requested():
            piezas = hechas + ([parcial] if parcial else [])
            progress(base + span, desc="Guardando lo procesado…")
            final = _unir(piezas, f"{stem}_montaje_PARCIAL", video)
            if final:
                entregas.extend(piezas + [final])
                # con partes, al volver se retoma; en "video entero" (n=1) NO hay
                # partes completas desde donde seguir — no prometer lo que no es
                cola_msg = ("Al volver a montar, sigue desde donde quedó." if n > 1 else
                            "OJO: en «video entero» no hay partes: al volver arranca "
                            "de cero (usá partes para poder retomar).")
                yield final, list(entregas), (
                    f"⏹️ **Detenido.** {etiqueta}: guardé el video hasta donde llegó "
                    f"({len(piezas)} de {n} partes) en `{final}`. {cola_msg}")
            else:
                entregas.extend(piezas)
                yield (piezas[-1] if piezas else None), list(entregas), (
                    f"⏹️ **Detenido** en {etiqueta} con {len(piezas)} de {n} partes. "
                    f"Quedaron en `{listas_dir}`.")
            break

        progress(base + span * 0.96, desc=f"{etiqueta} · uniendo las partes…")
        final = _unir(hechas, f"{stem}_montaje", video)
        hechos += 1
        if final:
            _limpiar_parciales(stem)
            listos.append(final)
            entregas.append(final)
            yield final, list(entregas), (
                f"🎉 {etiqueta} **completo** ({n} partes · {fmt_eta(t_gastado)} de proceso)."
                + (f" Sigo con el próximo ({nv - hechos} restantes)…" if trabajos else ""))
        else:
            entregas.extend(hechas)
            yield (hechas[-1] if hechas else None), list(entregas), (
                f"⚠️ {etiqueta}: las {n} partes están listas pero la unión falló; "
                f"están sueltas en `{listas_dir}`.")

    # --- resumen final ----------------------------------------------------------
    if pipeline_core.stop_requested():
        pend = len(trabajos)
        yield (entregas[-1] if entregas else None), list(entregas), (
            f"⏹️ **Montaje detenido** — {len(listos)}/{nv} videos completos en `outputs/`"
            + (f", {pend} sin empezar" if pend else "") + ". Volvé a darle y retoma solo.")
    else:
        cola = f" · descartados: {', '.join(descartados)}" if descartados else ""
        yield (entregas[-1] if entregas else None), list(entregas), (
            f"✅ **Montaje completo: {len(listos)}/{nv} video(s)** en `outputs/`{cola}.")


def previsualizar(dfm_id, files, preset, n_frames, piel=0.0, enhancer=None,
                  borde=config.EDGE_SUELTO, progress=gr.Progress()):
    """👁️ Muestra N frames sueltos ya montados, ANTES de lanzar el montaje entero.

    Toma frames repartidos a lo largo del video (keyframes equiespaciados, así
    caen poses distintas) y los procesa con LOS MISMOS ajustes del montaje. Con
    un video de 4 min a máxima calidad, esto son ~1 min contra varias horas:
    sirve para ver la identidad, la máscara y el color antes de comprometerse.

    Aproximación honesta: la preview corre en 1 pasada, sin la estabilización
    temporal ni el QC del montaje real (que necesitan la secuencia completa).
    Lo que ves por frame es representativo; la estabilidad entre frames, no.
    """
    if not dfm_id or dfm_id == NO_DFM:
        raise gr.Error("Elegí un modelo .dfm (se crean en la pestaña 🧬 Deep Swap).")
    videos = _norm_files(files)
    if not videos:
        raise gr.Error("Subí el video que querés previsualizar.")
    ensure_gpu_libre("la previsualización")
    pipeline_core.clear_stop()

    video = videos[0]        # siempre el PRIMERO de la lista: predecible
    n = int(n_frames or 10)
    try:
        progress(0.02, desc="Cargando el modelo entrenado…")
        pipeline = get_pipeline(_settings(dfm_id, preset, piel, enhancer, borde),
                                progress=lambda f, m="": progress(0.02 + f * 0.35, desc=m))
        frames = pipeline.preview(
            video, n_frames=n,
            progress=lambda f, m="": progress(0.37 + f * 0.63, desc=m))
    except gr.Error:
        raise
    except Exception as exc:  # pragma: no cover
        log.exception("Montador: falló la previsualización")
        raise gr.Error(f"Error al previsualizar: {exc}")

    nombre = Path(video).name
    extra = f" · los otros {len(videos) - 1} video(s) de la cola no se tocan" if len(videos) > 1 else ""
    ajustes = f"textura de piel **{piel:.2f}**"
    if enhancer is not None:
        ajustes += f" · enhancer **{enhancer:.2f}**"
    return frames, (
        f"✅ **{len(frames)} frames** de **{nombre}**{extra} · {ajustes}.\n\n"
        f"Mirá identidad, bordes de la máscara y color. *Es 1 pasada sin "
        f"estabilización temporal ni QC: el montaje final sale algo más estable.* "
        f"Si convence, dale a 🎬 MONTAR.")


def detener():
    """Pide la parada cooperativa: el frame en curso termina, lo hecho se guarda."""
    pipeline_core.request_stop()
    return ("⏹️ **Deteniendo…** termino el frame en curso, uno lo que ya está hecho "
            "y lo guardo en `outputs/`. Unos segundos.")


# ---------------------------------------------------------------------------
# La pestaña
# ---------------------------------------------------------------------------
def build_tab() -> dict:
    """Construye el contenido de la pestaña 🎬 Montador (llamar dentro del Tab).

    Devuelve los componentes que otras pestañas necesitan (p. ej. el desplegable
    de modelos, que Deep Swap refresca al exportar/importar un .dfm).
    """
    gr.Markdown(
        "### 🎬 Montar un modelo entrenado en videos\n"
        "Elegí el **modelo .dfm** (se crean en 🧬 Deep Swap), subí **uno o varios** "
        "videos y dale a MONTAR: van del más corto al más largo, por partes, y cada "
        "parte terminada aparece al instante."
    )
    with gr.Row():
        with gr.Column():
            with gr.Row():
                dfm_dd = gr.Dropdown(choices=dfm_choices(), value=NO_DFM, label="🧬 Modelo",
                                     scale=4)
                refresh_btn = gr.Button("🔄", scale=1, size="sm")
            preset_dd = gr.Dropdown(
                choices=PRESETS, value=config.EXPR_MAXIDENTITY, label="Preset de calidad",
                info="La identidad la pone el modelo; esto decide el pegado. Final → "
                     "Máxima Identidad. Probar encuadre/detección → Estándar (mucho más rápido).",
            )
            partes_dd = gr.Dropdown(
                choices=PARTES, value=30, label="✂️ Procesar por partes de",
                info="Cada parte terminada aparece abajo al instante.",
            )
            videos_in = gr.Files(
                label="🎬 Video(s) donde montar — 1 solo o una cola",
                file_count="multiple", file_types=["video"], type="filepath",
            )
            borde_dd = gr.Dropdown(
                choices=list(config.EDGE_FIT_LABELS.items()),
                value=config.EDGE_SUELTO, label="✂️ Ajuste del borde de la cara",
                info="Si el swap se DESBORDA (cuello, pelo, más allá del mentón), subilo. "
                     "El preset de Máxima Identidad quita el recorte a propósito para que "
                     "pase la geometría del modelo — y ese es justo el precio.",
            )
            with gr.Accordion("🎚️ Ajuste fino del realismo (contra el look plástico)",
                              open=False):
                piel_sl = gr.Slider(
                    0.0, 1.0, value=0.0, step=0.05, label="Textura de piel del video",
                    info="Reinyecta el GRANO y los POROS del frame original dentro de la "
                         "cara. No arrastra identidad (eso vive en la forma, no en la "
                         "micro-textura): devuelve piel real bajo esa luz. Contra la cara "
                         "de cera, probá 0,3–0,5. En 0 queda como antes.",
                )
                enh_sl = gr.Slider(
                    0.0, 1.0, value=0.7, step=0.05, label="Fuerza del enhancer (CodeFormer)",
                    info="Cuánto se mezcla la restauración. Alto = más nítido pero más "
                         "'retocado'; bajo = más crudo del modelo, más natural pero más "
                         "suave si está poco entrenado. 0,7 es el valor del preset.",
                )
            with gr.Group():
                gr.Markdown("#### 👁️ Probar antes de montar")
                n_frames = gr.Slider(
                    4, 16, value=10, step=1, label="Frames de muestra",
                    info="Se toman repartidos a lo largo del video (poses distintas) "
                         "y se procesan con los MISMOS ajustes del montaje.",
                )
                preview_btn = gr.Button("👁️ Previsualizar frames", variant="secondary")
            montar_btn = gr.Button("🎬 MONTAR", variant="primary")
            detener_btn = gr.Button("⏹️ Detener y guardar lo hecho", variant="stop")
            gr.Markdown(
                "*Detener no tira nada: une lo terminado y deja el video hasta ahí en "
                "`outputs/`. Al volver a montar, sigue desde donde quedó.*",
                elem_classes="fuser-soft",
            )
        with gr.Column():
            galeria = gr.Gallery(label="👁️ Frames de muestra (antes de montar)",
                                 columns=3, object_fit="contain", height=380)
            preview = gr.Video(label="🎬 Última parte terminada (mirá la calidad acá)")
            archivos = gr.Files(label="⬇️ Partes y videos completos (descargar)")
            estado = gr.Markdown("", elem_classes="fuser-soft")

    # concurrency_id="gpu": TODAS las acciones que cargan modelos comparten un
    # carril de a uno (también en Face Swap). Sin esto, dos eventos simultáneos
    # (montar + una preview) cargan DOS juegos de modelos en los 8 GB y DirectML
    # mata el proceso; además, el clear_stop de una acción nueva anulaba el
    # Detener de la que estaba corriendo. (Hallazgo de la revisión adversarial.)
    preview_btn.click(previsualizar,
                      inputs=[dfm_dd, videos_in, preset_dd, n_frames, piel_sl, enh_sl,
                              borde_dd],
                      outputs=[galeria, estado],
                      concurrency_id="gpu", concurrency_limit=1)
    montar_btn.click(montar,
                     inputs=[dfm_dd, videos_in, preset_dd, partes_dd, piel_sl, enh_sl,
                             borde_dd],
                     outputs=[preview, archivos, estado],
                     concurrency_id="gpu", concurrency_limit=1)
    # En su propio evento SIN cola: tiene que llegar mientras el montaje corre.
    detener_btn.click(detener, inputs=None, outputs=estado, queue=False)
    refresh_btn.click(lambda: gr.update(choices=dfm_choices()), inputs=None, outputs=dfm_dd)

    return {"dfm_dd": dfm_dd, "estado": estado}
