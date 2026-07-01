"""Cliente HTTP/WebSocket para un servidor **ComfyUI** local.

Implementado con la **librería estándar** (``urllib``) para no añadir
dependencias obligatorias a Fuser. El progreso en vivo (paso x/y) usa
``websocket-client`` **si está instalado**; si no, hace *fallback* a *polling*
de ``/history`` (progreso más grueso, pero funciona igual).

API de ComfyUI utilizada:
- ``POST /prompt``            encola un workflow (grafo en "API format").
- ``GET  /history/{id}``      resultado de un trabajo (outputs por nodo).
- ``POST /upload/image``      sube la imagen de entrada (multipart).
- ``GET  /view``             descarga un fichero de salida.
- ``GET  /object_info``       qué nodos/modelos hay instalados.
- ``GET  /system_stats``      *health check*.
- ``WS   /ws?clientId=...``   progreso en vivo.
- ``POST /interrupt``         cancela el trabajo en curso.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

from ..utils.logging import get_logger

log = get_logger(__name__)

ProgressCb = Optional[Callable[[float, str], None]]


class ComfyUIError(RuntimeError):
    """Error genérico al hablar con ComfyUI."""


class ComfyUINotAvailable(ComfyUIError):
    """ComfyUI no responde en la URL configurada."""

    def __init__(self, url: str, detail: str = ""):
        super().__init__(
            f"No se pudo conectar con ComfyUI en {url}.\n"
            "Arráncalo primero (en su carpeta):\n"
            "    python main.py --listen 127.0.0.1 --port 8188 --lowvram\n"
            "y comprueba la URL en los ajustes de la pestaña 'Imagen → Vídeo'.\n"
            f"{('Detalle: ' + detail) if detail else ''}"
        )
        self.url = url


def _summarize_comfy_error(body: str) -> str:
    """Extrae un motivo legible del cuerpo de error de ``/prompt`` (HTTP 400).

    ComfyUI devuelve JSON con ``error`` y/o ``node_errors``; de ahí sacamos qué
    nodo/valor falló (p.ej. un custom node no instalado) para decírselo al usuario
    en vez de un genérico "ComfyUI no responde".
    """
    if not body:
        return ""
    try:
        d = json.loads(body)
    except Exception:
        return body.strip()[:300]
    parts: List[str] = []
    err = d.get("error")
    if isinstance(err, dict):
        msg = err.get("message") or err.get("type") or ""
        det = err.get("details") or ""
        parts.append((f"{msg} — {det}" if det else msg).strip())
    for nid, ne in list((d.get("node_errors") or {}).items())[:4]:
        ct = (ne or {}).get("class_type", "")
        emsgs = "; ".join(
            (e.get("message", "") + (f" ({e.get('details')})" if e.get("details") else "")).strip()
            for e in ((ne or {}).get("errors") or []) if isinstance(e, dict)
        )
        parts.append(f"nodo {nid} [{ct}]: {emsgs}".strip(": "))
    out = " | ".join(p for p in parts if p).strip()
    if ("does not exist" in out.lower() or "not in list" in out.lower()
            or "cannot execute" in out.lower()):
        out += "  → parece faltar un custom node o un modelo; míralo en el botón "
        out += "'🔌 Comprobar ComfyUI' de la pestaña."
    return out[:500] if out else body.strip()[:300]


@dataclass
class OutputFile:
    """Un fichero producido por el workflow (vídeo, imagen o audio)."""

    filename: str
    subfolder: str
    type: str           # "output" | "temp" | "input"
    node_id: str
    media: str          # "video" | "image" | "audio"


class ComfyUIClient:
    """Cliente fino y robusto para ComfyUI."""

    def __init__(self, base_url: str, client_id: Optional[str] = None, timeout: int = 30):
        self.base_url = base_url.rstrip("/")
        self.client_id = client_id or uuid.uuid4().hex
        self.timeout = timeout

    # ---- HTTP helpers ------------------------------------------------------
    def _url(self, path: str, query: Optional[dict] = None) -> str:
        url = f"{self.base_url}{path}"
        if query:
            url += "?" + urllib.parse.urlencode(query)
        return url

    def _request(self, path: str, *, data: Optional[bytes] = None,
                 headers: Optional[dict] = None, query: Optional[dict] = None,
                 timeout: Optional[int] = None) -> bytes:
        req = urllib.request.Request(self._url(path, query), data=data,
                                     headers=headers or {})
        try:
            with urllib.request.urlopen(req, timeout=timeout or self.timeout) as resp:
                return resp.read()
        except urllib.error.HTTPError as exc:
            # El servidor RESPONDE pero rechaza la petición (400 = workflow inválido,
            # p.ej. falta un custom node). NO es "ComfyUI caído": propaga el motivo
            # real para no mandar al usuario a rearrancar ComfyUI en vano.
            body = ""
            try:
                body = exc.read().decode("utf-8", "replace")
            except Exception:
                pass
            detail = _summarize_comfy_error(body) or str(getattr(exc, "reason", "") or "")
            raise ComfyUIError(
                f"ComfyUI rechazó la petición ({exc.code} {exc.reason}) en {path}. {detail}".strip()
            ) from exc
        except urllib.error.URLError as exc:
            # Sin servidor escuchando / conexión rechazada: ahí sí, ComfyUI no está.
            raise ComfyUINotAvailable(self.base_url, str(getattr(exc, "reason", exc))) from exc

    def _get_json(self, path: str, query: Optional[dict] = None) -> dict:
        return json.loads(self._request(path, query=query).decode("utf-8"))

    def _post_json(self, path: str, payload: dict) -> dict:
        data = json.dumps(payload).encode("utf-8")
        raw = self._request(path, data=data, headers={"Content-Type": "application/json"})
        return json.loads(raw.decode("utf-8")) if raw else {}

    # ---- Salud / introspección --------------------------------------------
    def is_available(self) -> bool:
        try:
            self._request("/system_stats", timeout=min(self.timeout, 5))
            return True
        except Exception:
            return False

    def get_object_info(self) -> dict:
        return self._get_json("/object_info")

    def has_nodes(self, *class_types: str) -> Dict[str, bool]:
        info = self.get_object_info()
        return {c: (c in info) for c in class_types}

    def has_node(self, class_type: str) -> bool:
        """Comprueba UN nodo vía ``/object_info/<clase>`` (ligero, sin bajar todo).

        ComfyUI devuelve ``{}`` si la clase no existe. Usado para degradar con
        elegancia workflows que dependen de custom nodes (p.ej. DisTorch2).
        """
        try:
            d = self._get_json(f"/object_info/{class_type}")
            return isinstance(d, dict) and class_type in d
        except ComfyUIError:
            return False
        except Exception:  # pragma: no cover - respuesta rara
            return False

    # ---- Subida de imagen --------------------------------------------------
    def upload_image(self, image_path: str, subfolder: str = "fuser_i2v",
                     overwrite: bool = True) -> str:
        """Sube una imagen y devuelve el nombre a usar en el nodo ``LoadImage``."""
        path = Path(image_path)
        if not path.exists():
            raise ComfyUIError(f"No existe la imagen: {image_path}")
        body, content_type = _multipart(
            fields={"overwrite": "true" if overwrite else "false",
                    "type": "input", "subfolder": subfolder},
            file_field="image", file_path=path,
        )
        raw = self._request("/upload/image", data=body,
                            headers={"Content-Type": content_type})
        info = json.loads(raw.decode("utf-8"))
        name = info.get("name", path.name)
        sub = info.get("subfolder", "")
        return f"{sub}/{name}" if sub else name

    # ---- Encolar y esperar -------------------------------------------------
    def queue_prompt(self, graph: dict) -> str:
        resp = self._post_json("/prompt", {"prompt": graph, "client_id": self.client_id})
        if "prompt_id" not in resp:
            errors = resp.get("node_errors") or resp.get("error") or resp
            raise ComfyUIError(f"ComfyUI rechazó el workflow: {json.dumps(errors)[:1500]}")
        return resp["prompt_id"]

    def interrupt(self) -> None:
        try:
            self._request("/interrupt", data=b"", headers={"Content-Type": "application/json"})
        except Exception:
            pass

    def get_history(self, prompt_id: str) -> dict:
        try:
            return self._get_json(f"/history/{prompt_id}")
        except Exception:
            return {}

    def wait(self, prompt_id: str, *, progress: ProgressCb = None,
             timeout: int = 3600) -> dict:
        """Espera a que termine ``prompt_id`` y devuelve sus *outputs*.

        Usa WebSocket si ``websocket-client`` está disponible; si no, *polling*.
        """
        try:
            import websocket  # type: ignore  # websocket-client (opcional)
        except Exception:
            websocket = None

        if websocket is not None:
            try:
                return self._wait_ws(websocket, prompt_id, progress=progress, timeout=timeout)
            except ComfyUIError:
                raise
            except Exception as exc:  # pragma: no cover - si el WS falla, polling
                log.warning("WebSocket falló (%s); paso a polling.", exc)
        return self._wait_poll(prompt_id, progress=progress, timeout=timeout)

    def _wait_ws(self, websocket, prompt_id: str, *, progress: ProgressCb,
                 timeout: int) -> dict:
        ws_url = self.base_url.replace("http://", "ws://").replace("https://", "wss://")
        ws = websocket.WebSocket()
        ws.connect(f"{ws_url}/ws?clientId={self.client_id}", timeout=self.timeout)
        ws.settimeout(timeout)
        start = time.monotonic()
        try:
            while True:
                if time.monotonic() - start > timeout:
                    raise ComfyUIError("Tiempo de espera agotado generando el vídeo.")
                msg = ws.recv()
                if isinstance(msg, bytes):  # vistas previas binarias: ignorar
                    continue
                data = json.loads(msg)
                mtype, body = data.get("type"), data.get("data", {})
                if mtype == "progress" and progress:
                    mx = body.get("max") or 1
                    progress(min(0.99, body.get("value", 0) / mx),
                             f"Paso {body.get('value', 0)}/{mx}")
                elif mtype == "execution_error" and body.get("prompt_id") == prompt_id:
                    raise ComfyUIError(
                        f"ComfyUI falló: {body.get('exception_message', 'error desconocido')}"
                    )
                elif mtype == "executing" and body.get("prompt_id") == prompt_id:
                    if body.get("node") is None:  # node==None => terminó este prompt
                        break
        finally:
            try:
                ws.close()
            except Exception:
                pass
        return self._outputs_or_raise(prompt_id)

    def _wait_poll(self, prompt_id: str, *, progress: ProgressCb, timeout: int) -> dict:
        start = time.monotonic()
        while True:
            if time.monotonic() - start > timeout:
                raise ComfyUIError("Tiempo de espera agotado generando el vídeo.")
            hist = self.get_history(prompt_id)
            if prompt_id in hist:
                status = hist[prompt_id].get("status", {})
                if status.get("status_str") == "error":
                    raise ComfyUIError("ComfyUI marcó el trabajo como fallido (revisa su consola).")
                return hist[prompt_id].get("outputs", {})
            if progress:
                progress(0.5, "Generando… (sin progreso fino; instala 'websocket-client')")
            time.sleep(2.0)

    def _outputs_or_raise(self, prompt_id: str) -> dict:
        hist = self.get_history(prompt_id)
        if prompt_id not in hist:
            raise ComfyUIError("El trabajo terminó pero no aparece en el historial.")
        return hist[prompt_id].get("outputs", {})

    # ---- Recoger y descargar resultados -----------------------------------
    @staticmethod
    def collect_outputs(outputs: dict) -> List[OutputFile]:
        """Recorre los outputs y extrae ficheros de vídeo / imagen / audio."""
        found: List[OutputFile] = []
        # Claves donde ComfyUI/VHS publican ficheros, y cómo clasificarlas.
        media_keys = {"gifs": "video", "videos": "video", "images": "image", "audio": "audio"}
        for node_id, node_out in (outputs or {}).items():
            for key, default_media in media_keys.items():
                for entry in node_out.get(key, []) or []:
                    fn = entry.get("filename")
                    if not fn:
                        continue
                    media = _media_from_name(fn, default_media)
                    found.append(OutputFile(
                        filename=fn, subfolder=entry.get("subfolder", ""),
                        type=entry.get("type", "output"), node_id=str(node_id), media=media,
                    ))
        return found

    def pick_output(self, outputs: dict, media: str) -> Optional[OutputFile]:
        files = [f for f in self.collect_outputs(outputs) if f.media == media]
        return files[-1] if files else None

    def download(self, out: OutputFile, dest_path: str) -> str:
        raw = self._request("/view", query={
            "filename": out.filename, "subfolder": out.subfolder, "type": out.type,
        }, timeout=300)
        dest = Path(dest_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(raw)
        return str(dest)


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
_VIDEO_EXT = {".mp4", ".webm", ".mkv", ".mov", ".gif", ".avi"}
_AUDIO_EXT = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}
_IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp"}


def _media_from_name(filename: str, default: str) -> str:
    ext = Path(filename).suffix.lower()
    if ext in _VIDEO_EXT and ext != ".gif":
        return "video"
    if ext == ".gif":
        return "video"
    if ext in _AUDIO_EXT:
        return "audio"
    if ext in _IMAGE_EXT:
        return "image"
    return default


def _multipart(fields: Dict[str, str], file_field: str, file_path: Path) -> tuple[bytes, str]:
    """Construye un cuerpo multipart/form-data (sin dependencias externas)."""
    boundary = f"----fuser{uuid.uuid4().hex}"
    crlf = b"\r\n"
    buf = bytearray()
    for name, value in fields.items():
        buf += f"--{boundary}".encode() + crlf
        buf += f'Content-Disposition: form-data; name="{name}"'.encode() + crlf + crlf
        buf += str(value).encode() + crlf
    buf += f"--{boundary}".encode() + crlf
    buf += (f'Content-Disposition: form-data; name="{file_field}"; '
            f'filename="{file_path.name}"').encode() + crlf
    buf += b"Content-Type: application/octet-stream" + crlf + crlf
    buf += file_path.read_bytes() + crlf
    buf += f"--{boundary}--".encode() + crlf
    return bytes(buf), f"multipart/form-data; boundary={boundary}"
