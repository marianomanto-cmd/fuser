"""Desinstalador COMPLETO de Fuser (cross-platform, sin dependencias).

    python scripts/uninstall.py                 # plan + confirmación interactiva
    python scripts/uninstall.py --dry-run       # solo muestra qué borraría
    python scripts/uninstall.py --yes           # sin preguntas (deja tus datos)
    python scripts/uninstall.py --yes --purge-data --remove-repo   # NO deja nada

Quita todo lo que la instalación creó: el entorno `.venv`, el motor FaceFusion
vendorizado, los modelos `.onnx`, el entrenador DeepFaceLab (con su build y sus
preentrenados), las cachés/temporales, los accesos directos y la imagen Docker.

Corre con CUALQUIER Python (solo stdlib): funciona incluso si el `.venv` ya no
existe o está roto. No importa el paquete ``fuser`` a propósito — las rutas se
re-derivan aquí para no depender de dependencias que estamos borrando.

TRES REGLAS DE SEGURIDAD (por las que este script existe en vez de un `rm -rf`)
------------------------------------------------------------------------------
1. **Nunca atraviesa junctions.** ``vendor/facefusion/.assets/models`` es un
   junction a ``E:\\modelos\\facefusion`` en la máquina del usuario: un borrado
   recursivo ingenuo se lleva por delante el disco E:. Aquí se DESMONTAN todos
   los puntos de reanálisis (bottom-up) antes de tocar nada.
2. **Los datos del usuario no se borran salvo que lo pidas.** La Biblioteca de
   Caras, los videos de salida y los workspaces de entrenamiento se conservan
   por defecto; hay que pasar ``--purge-data`` (o decir que sí a la pregunta).
3. **Los `.dfm` entrenados se rescatan.** Un `.dfm` son días/semanas de GPU y
   vive DENTRO de las carpetas que borramos: antes de borrar se copian a
   ``../fuser_dfm_backup/`` (salvo con ``--purge-data``, donde ya pediste todo).

Además se niega a correr si hay un entrenamiento DFL vivo o la app levantada
(ver ``--force``): matarlos a media corrida corrompe el modelo en disco.
"""
from __future__ import annotations

import argparse
import os
import shutil
import socket
import stat
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Los .dfm rescatados van FUERA del árbol que se borra.
RESCUE_DIR = PROJECT_ROOT.parent / "fuser_dfm_backup"


# ---------------------------------------------------------------------------
# Rutas de la instalación (re-derivadas: NO importamos fuser.config)
# ---------------------------------------------------------------------------
def _env_dir(var: str, default: Path) -> Path:
    val = os.environ.get(var)
    return Path(val).expanduser() if val else default


def models_dir() -> Path:
    return _env_dir("FUSER_MODELS_DIR", PROJECT_ROOT / "models")


def outputs_dir() -> Path:
    return _env_dir("FUSER_OUTPUT_DIR", PROJECT_ROOT / "outputs")


def temp_dir() -> Path:
    return _env_dir("FUSER_TEMP_DIR", PROJECT_ROOT / "tmp")


def faces_dir() -> Path:
    return _env_dir("FUSER_FACES_DIR", PROJECT_ROOT / "faces")


def insightface_root() -> Path:
    return _env_dir("FUSER_INSIGHTFACE_ROOT", models_dir())


def vendor_dir() -> Path:
    return PROJECT_ROOT / "vendor" / "facefusion"


def ff_models_dir() -> Path:
    """Almacén de modelos de FaceFusion (suele ser un junction a E:)."""
    return vendor_dir() / ".assets" / "models"


def dfl_root() -> Path:
    """Misma lógica que ``fuser.core.dfm_trainer.dfl_root()``."""
    env = os.environ.get("FUSER_DFL_DIR")
    if env:
        return Path(env).expanduser()
    e_drive = Path("E:/modelos")
    if e_drive.is_dir():
        return e_drive / "deepfacelab"
    return PROJECT_ROOT / "dfl"


# ---------------------------------------------------------------------------
# Junctions / symlinks: detectar, leer destino, DESMONTAR sin seguirlos
# ---------------------------------------------------------------------------
def is_reparse(p: Path) -> bool:
    """True si es junction (Windows) o symlink (POSIX). NO sigue el enlace."""
    try:
        st = os.stat(p, follow_symlinks=False)
    except OSError:
        return False
    attrs = getattr(st, "st_file_attributes", None)
    if attrs is not None:
        return bool(attrs & stat.FILE_ATTRIBUTE_REPARSE_POINT)
    return stat.S_ISLNK(st.st_mode)


def link_target(p: Path) -> Optional[Path]:
    """Destino de un junction/symlink, o None."""
    try:
        return Path(os.readlink(p))
    except OSError:
        return None


def _unmount(p: Path) -> bool:
    """Desmonta un enlace dejando INTACTO su destino."""
    try:
        os.rmdir(p)          # junction de directorio (Windows)
        return True
    except OSError:
        pass
    try:
        p.unlink()           # symlink (POSIX)
        return True
    except OSError:
        return False


def unmount_reparse_points(root: Path) -> List[Path]:
    """Desmonta TODOS los enlaces dentro de ``root`` antes de un borrado.

    Sin esto, un ``rmtree`` sobre ``vendor/`` puede vaciar ``E:\\modelos``.
    """
    out: List[Path] = []
    if not root.exists() and not is_reparse(root):
        return out
    if is_reparse(root):
        if _unmount(root):
            out.append(root)
        return out
    for dirpath, dirnames, _files in os.walk(root, topdown=True, followlinks=False):
        for name in list(dirnames):
            p = Path(dirpath) / name
            if is_reparse(p):
                dirnames.remove(name)        # jamás descender por el enlace
                if _unmount(p):
                    out.append(p)
    return out


# ---------------------------------------------------------------------------
# Medición y borrado
# ---------------------------------------------------------------------------
def human(n: int) -> str:
    f = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if f < 1024 or unit == "TB":
            return f"{f:,.1f} {unit}".replace(",", ".") if unit != "B" else f"{int(f)} B"
        f /= 1024
    return f"{f:.1f} TB"


def dir_size(p: Path) -> int:
    """Tamaño en disco SIN atravesar enlaces (los junctions cuentan como 0)."""
    if is_reparse(p):
        return 0
    try:
        if p.is_file():
            return p.stat().st_size
    except OSError:
        return 0
    total = 0
    for dirpath, dirnames, filenames in os.walk(p, topdown=True, followlinks=False):
        for name in list(dirnames):
            if is_reparse(Path(dirpath) / name):
                dirnames.remove(name)
        for name in filenames:
            f = Path(dirpath) / name
            try:
                total += os.stat(f, follow_symlinks=False).st_size
            except OSError:
                pass
    return total


def _force_writable(func, path, _exc):
    """Reintenta un borrado que falló por atributo de solo-lectura."""
    try:
        os.chmod(path, stat.S_IWRITE)
        func(path)
    except OSError:
        pass


def _rmtree(p: Path) -> None:
    if sys.version_info >= (3, 12):
        shutil.rmtree(p, onexc=_force_writable)
    else:
        shutil.rmtree(p, onerror=_force_writable)


def remove(p: Path) -> Tuple[bool, str]:
    """Borra archivo o carpeta desmontando enlaces primero. -> (ok, detalle)."""
    try:
        if is_reparse(p):
            return (_unmount(p), "enlace desmontado (destino intacto)")
        if p.is_file():
            p.unlink()
            return True, ""
        if not p.exists():
            return True, "no existía"
        desmontados = unmount_reparse_points(p)
        _rmtree(p)
        detalle = f"{len(desmontados)} enlace(s) desmontado(s)" if desmontados else ""
        return (not p.exists()), detalle
    except OSError as exc:
        return False, str(exc)


def unsafe_reason(p: Path) -> Optional[str]:
    """Motivo por el que NO se debe borrar esta ruta (o None si es segura).

    Cinturón contra una env var mal puesta (``FUSER_MODELS_DIR=E:\\``).
    """
    if p.parent == p:
        return "es la raíz de un disco"
    try:
        if p == Path.home():
            return "es tu carpeta de usuario"
    except (OSError, RuntimeError):
        pass
    try:
        inside_project = p == PROJECT_ROOT or PROJECT_ROOT in p.parents
    except OSError:
        inside_project = False
    if not inside_project and len(p.parts) < 3:
        return "está demasiado cerca de la raíz del disco"
    return None


# ---------------------------------------------------------------------------
# Guardia: nada de desinstalar con la app o un entrenamiento vivos
# ---------------------------------------------------------------------------
def _pid_alive(pid: int) -> bool:
    if os.name == "nt":
        try:
            out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                                 capture_output=True, text=True, timeout=20)
            return str(pid) in (out.stdout or "")
        except (OSError, subprocess.SubprocessError):
            return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def live_trainings() -> List[str]:
    """Slugs con un entrenamiento DFL realmente vivo (train.pid + proceso)."""
    vivos: List[str] = []
    wsr = dfl_root() / "workspaces"
    if not wsr.is_dir():
        return vivos
    try:
        candidatos = sorted(wsr.iterdir())
    except OSError:
        return vivos
    for d in candidatos:
        pid_file = d / "train.pid"
        try:
            pid = int(pid_file.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            continue
        if _pid_alive(pid):
            vivos.append(f"{d.name} (PID {pid})")
    return vivos


def app_running(port: int = 7860) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


# ---------------------------------------------------------------------------
# Rescate de .dfm (días/semanas de GPU que viven dentro de lo que se borra)
# ---------------------------------------------------------------------------
def find_dfm() -> List[Path]:
    encontrados: List[Path] = []
    raices = [ff_models_dir(), dfl_root() / "workspaces", models_dir()]
    for raiz in raices:
        if not raiz.exists():
            continue
        # Aquí SÍ seguimos el junction a propósito: solo LEEMOS, para copiar.
        for dirpath, _dirnames, filenames in os.walk(raiz, followlinks=True):
            for name in filenames:
                if name.lower().endswith(".dfm"):
                    encontrados.append(Path(dirpath) / name)
    # Dedup por ruta real (el junction puede exponer el mismo archivo dos veces).
    vistos, unicos = set(), []
    for f in encontrados:
        try:
            clave = f.resolve()
        except OSError:
            clave = f
        if clave not in vistos:
            vistos.add(clave)
            unicos.append(f)
    return unicos


def rescue_dfm(archivos: List[Path], dry_run: bool) -> int:
    if not archivos:
        return 0
    print(f"\n💾 Rescatando {len(archivos)} modelo(s) .dfm entrenado(s) → {RESCUE_DIR}")
    if dry_run:
        for f in archivos:
            print(f"   · {f.name}  ({human(dir_size(f))})")
        return 0
    RESCUE_DIR.mkdir(parents=True, exist_ok=True)
    copiados = 0
    for f in archivos:
        destino = RESCUE_DIR / f.name
        n = 1
        while destino.exists():
            destino = RESCUE_DIR / f"{f.stem}_{n}{f.suffix}"
            n += 1
        try:
            shutil.copy2(f, destino)
            copiados += 1
            print(f"   ✅ {destino.name}  ({human(destino.stat().st_size)})")
        except OSError as exc:
            print(f"   ⚠️  No pude copiar {f.name}: {exc}")
    return copiados


# ---------------------------------------------------------------------------
# Inventario de lo que se va a borrar
# ---------------------------------------------------------------------------
class Item:
    def __init__(self, path: Path, label: str, group: str):
        self.path = path
        self.label = label
        self.group = group
        self.size = dir_size(path)

    @property
    def is_link(self) -> bool:
        return is_reparse(self.path)


def _add(items: List[Item], path: Path, label: str, group: str) -> None:
    if path.exists() or is_reparse(path):
        items.append(Item(path, label, group))


def collect_cache_items(items: List[Item]) -> None:
    _add(items, temp_dir(), "Temporales", "Caché y temporales")
    _add(items, PROJECT_ROOT / "prueba", "Salida del demo (prueba/)", "Caché y temporales")
    _add(items, PROJECT_ROOT / ".insightface", "Caché de InsightFace", "Caché y temporales")
    _add(items, Path.home() / ".insightface", "Caché de InsightFace (perfil)", "Caché y temporales")
    _add(items, PROJECT_ROOT / ".gradio", "Caché de Gradio", "Caché y temporales")
    _add(items, PROJECT_ROOT / "gradio_cached_examples", "Ejemplos de Gradio", "Caché y temporales")
    _add(items, PROJECT_ROOT / "flagged", "Gradio flagged/", "Caché y temporales")
    for log in sorted(PROJECT_ROOT.glob("*.log")):
        _add(items, log, f"Log ({log.name})", "Caché y temporales")
    for pyc in sorted(PROJECT_ROOT.rglob("__pycache__")):
        if "vendor" not in pyc.parts and ".venv" not in pyc.parts:
            _add(items, pyc, f"__pycache__ ({pyc.parent.name})", "Caché y temporales")


def collect_shortcut_items(items: List[Item]) -> None:
    _add(items, PROJECT_ROOT / "Fuser.bat", "Lanzador generado (Fuser.bat)", "Accesos directos")
    _add(items, PROJECT_ROOT / "fuser.ico", "Icono generado", "Accesos directos")
    if os.name != "nt":
        return
    perfil = Path(os.environ.get("USERPROFILE", Path.home()))
    candidatos = [
        perfil / "Desktop" / "Fuser.lnk",
        perfil / "OneDrive" / "Desktop" / "Fuser.lnk",
        perfil / "AppData" / "Roaming" / "Microsoft" / "Windows"
        / "Start Menu" / "Programs" / "Fuser.lnk",
    ]
    for c in candidatos:
        _add(items, c, f"Acceso directo ({c.parent.name})", "Accesos directos")


def collect(purge_data: bool, remove_repo: bool) -> List[Item]:
    items: List[Item] = []

    # --- Programa e intérprete ---------------------------------------------
    _add(items, PROJECT_ROOT / ".venv", "Entorno virtual (.venv)", "Programa")
    _add(items, PROJECT_ROOT / "venv", "Entorno virtual (venv)", "Programa")

    # --- Motor FaceFusion + su almacén de modelos --------------------------
    ff_models = ff_models_dir()
    destino = link_target(ff_models) if is_reparse(ff_models) else None
    # Se borra vendor/ entero (está 100% gitignorado y solo aloja FaceFusion):
    # así no queda una carpeta huérfana vacía.
    _add(items, vendor_dir().parent, "Motor FaceFusion vendorizado (vendor/)", "Motor FaceFusion")
    if destino is not None:
        # El junction se desmonta con vendor/; su DESTINO real hay que
        # borrarlo aparte y a conciencia (es E:\modelos\facefusion).
        _add(items, destino, f"Modelos de FaceFusion (destino del junction: {destino})",
             "Motor FaceFusion")

    # --- Modelos ONNX de Fuser ---------------------------------------------
    _add(items, models_dir(), "Modelos .onnx de Fuser", "Modelos")
    ins = insightface_root()
    if ins != models_dir():
        _add(items, ins, "Modelos de InsightFace (buffalo_l)", "Modelos")

    # --- Entrenador DeepFaceLab --------------------------------------------
    raiz_dfl = dfl_root()
    if raiz_dfl.exists():
        if purge_data:
            _add(items, raiz_dfl, f"DeepFaceLab COMPLETO ({raiz_dfl})", "Entrenador DFL")
        else:
            # Software y descargas sí; los workspaces (tu entrenamiento) no.
            _add(items, raiz_dfl / "build", "Build de DeepFaceLab", "Entrenador DFL")
            _add(items, raiz_dfl / "pretrain", "Preentrenados (RTT/RTM)", "Entrenador DFL")
            _add(items, raiz_dfl / "downloads", "Descargas del entrenador", "Entrenador DFL")
            _add(items, raiz_dfl / "backend.txt", "Motor activo (backend.txt)", "Entrenador DFL")

    collect_cache_items(items)
    collect_shortcut_items(items)

    # --- Datos del usuario (solo con --purge-data) -------------------------
    if purge_data:
        _add(items, faces_dir(), "Biblioteca de Caras (TUS fotos)", "⚠️  Datos del usuario")
        _add(items, outputs_dir(), "Videos de salida (TUS resultados)", "⚠️  Datos del usuario")
        _add(items, raiz_dfl / "workspaces", "Workspaces de entrenamiento",
             "⚠️  Datos del usuario")

    if remove_repo:
        items.append(Item(PROJECT_ROOT, f"Código fuente del repo ({PROJECT_ROOT})", "Repositorio"))

    # Sin duplicados ni rutas contenidas en otra que ya se borra.
    unicos: List[Item] = []
    for it in sorted(items, key=lambda i: len(i.path.parts)):
        if any(pad.path == it.path or pad.path in it.path.parents for pad in unicos):
            continue
        unicos.append(it)
    return unicos


# ---------------------------------------------------------------------------
# Docker
# ---------------------------------------------------------------------------
def docker_cleanup(dry_run: bool) -> None:
    docker = shutil.which("docker")
    if not docker:
        return
    comandos = []
    if (PROJECT_ROOT / "docker-compose.yml").exists():
        comandos.append([docker, "compose", "down", "-v", "--remove-orphans"])
    comandos += [[docker, "rm", "-f", "fuser"], [docker, "rmi", "-f", "fuser:latest"]]
    print("\n🐳 Docker")
    for cmd in comandos:
        etiqueta = " ".join(cmd[1:])
        if dry_run:
            print(f"   · (dry-run) docker {etiqueta}")
            continue
        try:
            r = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True,
                               text=True, timeout=180)
            print(f"   {'✅' if r.returncode == 0 else '·'} docker {etiqueta}")
        except (OSError, subprocess.SubprocessError) as exc:
            print(f"   ⚠️  docker {etiqueta}: {exc}")


# ---------------------------------------------------------------------------
# Borrado del propio repo (auto-borrado)
# ---------------------------------------------------------------------------
def _schedule_self_delete() -> bool:
    """Programa el borrado de PROJECT_ROOT para DESPUÉS de que salgamos.

    No podemos borrarnos a nosotros mismos: Windows mantiene abierto el .bat que
    nos lanzó y el propio .py. Un proceso desacoplado espera unos segundos y
    remata la carpeta cuando ya nadie la tiene tomada.
    """
    ruta = str(PROJECT_ROOT)
    try:
        if os.name == "nt":
            DETACHED = 0x00000008 | 0x00000200        # DETACHED_PROCESS | NEW_GROUP
            subprocess.Popen(
                f'cmd /c timeout /t 3 /nobreak >nul & rmdir /s /q "{ruta}"',
                creationflags=DETACHED, close_fds=True)
        else:
            subprocess.Popen(["sh", "-c", f'sleep 2; rm -rf "{ruta}"'],
                             start_new_session=True, close_fds=True)
        return True
    except (OSError, subprocess.SubprocessError):
        return False


def remove_repo_tree() -> Tuple[bool, str]:
    """Vacía PROJECT_ROOT y programa la eliminación de la carpeta. Va al final."""
    yo = Path(__file__).resolve()
    unmount_reparse_points(PROJECT_ROOT)
    for hijo in sorted(PROJECT_ROOT.iterdir()):
        if hijo == yo.parent:            # scripts/ — contiene este archivo
            continue
        remove(hijo)
    # scripts/: todo menos este script y el lanzador que lo está ejecutando.
    en_uso = {yo, yo.with_suffix(".bat"), yo.with_suffix(".sh")}
    for hijo in sorted(yo.parent.iterdir()):
        if hijo not in en_uso:
            remove(hijo)
    try:
        os.chdir(PROJECT_ROOT.parent)    # no se puede borrar el directorio actual
    except OSError:
        pass
    if _schedule_self_delete():
        return True, ""
    return False, (f"quedó la carpeta {PROJECT_ROOT} — bórrala a mano: "
                   f"{'rmdir /s /q' if os.name == 'nt' else 'rm -rf'} \"{PROJECT_ROOT}\"")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _ask(pregunta: str, palabra: Optional[str] = None) -> bool:
    if not sys.stdin or not sys.stdin.isatty():
        return False
    try:
        resp = input(pregunta).strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return False
    if palabra:
        return resp == palabra
    return resp.lower() in ("s", "si", "sí", "y", "yes")


def main() -> int:
    p = argparse.ArgumentParser(
        description="Desinstala Fuser por completo (venv, modelos, FaceFusion, DFL, Docker).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Por defecto NO borra tus caras, tus videos de salida ni tus "
               "entrenamientos: para eso está --purge-data.")
    p.add_argument("--dry-run", action="store_true",
                   help="solo muestra el plan; no borra nada")
    p.add_argument("-y", "--yes", action="store_true",
                   help="no preguntar (asume que sí)")
    p.add_argument("--purge-data", action="store_true",
                   help="TAMBIÉN borra Biblioteca de Caras, salidas y workspaces DFL")
    p.add_argument("--remove-repo", action="store_true",
                   help="TAMBIÉN borra el código fuente (esta carpeta)")
    p.add_argument("--skip-docker", action="store_true",
                   help="no tocar imágenes/contenedores Docker")
    p.add_argument("--force", action="store_true",
                   help="desinstalar aunque la app o un entrenamiento estén corriendo")
    args = p.parse_args()

    print("=" * 70)
    print("  FUSER — desinstalación" + ("  [DRY-RUN: no se borra nada]" if args.dry_run else ""))
    print(f"  Proyecto: {PROJECT_ROOT}")
    print("=" * 70)

    # --- Guardia: procesos vivos ------------------------------------------
    if not args.force:
        entrenando = live_trainings()
        levantada = app_running()
        if entrenando or levantada:
            print("\n⛔ No puedo desinstalar ahora mismo:")
            for e in entrenando:
                print(f"   · Entrenamiento DFL EN CURSO: {e}")
            if levantada:
                print("   · La app está levantada en http://127.0.0.1:7860")
            print("\n   Cerrá la app y pausá/terminá el entrenamiento primero.")
            print("   (Matar un entrenamiento a media corrida puede corromper el modelo.)")
            print("   Si sabés lo que hacés: volvé a correr con --force")
            return 2

    # --- ¿Borrar también los datos del usuario? ---------------------------
    purge = args.purge_data
    if not purge and not args.yes and not args.dry_run:
        print("\nTus datos (Biblioteca de Caras, videos de salida y entrenamientos)")
        print("se CONSERVAN salvo que pidas lo contrario.")
        if _ask("¿Borrar TAMBIÉN tus datos? Escribí BORRAR para confirmar, o Enter para conservarlos: ",
                palabra="BORRAR"):
            purge = True
            print("   → se borrarán también tus datos.")
        else:
            print("   → tus datos se conservan.")

    items = collect(purge, args.remove_repo)
    if not items:
        print("\n✅ No encontré nada instalado. Ya está desinstalado.")
        return 0

    # --- Plan --------------------------------------------------------------
    total = sum(i.size for i in items)
    print("\nSe va a ELIMINAR:")
    grupo_actual = None
    for it in sorted(items, key=lambda i: (i.group, str(i.path))):
        if it.group != grupo_actual:
            grupo_actual = it.group
            print(f"\n  {grupo_actual}")
        motivo = unsafe_reason(it.path)
        marca = "🔗" if it.is_link else ("⛔" if motivo else "  ")
        detalle = f"  [SE OMITE: {motivo}]" if motivo else ""
        tam = "(enlace)" if it.is_link else human(it.size)
        print(f"   {marca} {it.label}")
        print(f"       {it.path}  —  {tam}{detalle}")
    print(f"\n  Espacio a liberar: ~{human(total)}")

    if not purge:
        conservados = [(d, etiqueta) for d, etiqueta in (
            (faces_dir(), "Biblioteca de Caras"),
            (outputs_dir(), "Videos de salida"),
            (dfl_root() / "workspaces", "Workspaces de entrenamiento")) if d.exists()]
        if conservados:
            print("\n  SE CONSERVAN (no se tocan):")
            for d, etiqueta in conservados:
                print(f"   · {etiqueta}: {d}")

    if args.dry_run:
        rescue_dfm(find_dfm() if not purge else [], dry_run=True)
        if not args.skip_docker:
            docker_cleanup(dry_run=True)
        print("\n(dry-run) No se borró nada.")
        return 0

    if not args.yes and not _ask("\n¿Confirmás la desinstalación? [s/N]: "):
        print("Cancelado. No se borró nada.")
        return 1

    # --- Rescate de .dfm ---------------------------------------------------
    if not purge:
        rescue_dfm(find_dfm(), dry_run=False)

    # --- Borrado -----------------------------------------------------------
    print("\n🧹 Borrando…")
    liberado, fallos = 0, []
    repo_item = None
    for it in sorted(items, key=lambda i: str(i.path)):
        if it.path == PROJECT_ROOT:
            repo_item = it                     # el repo, siempre al final
            continue
        motivo = unsafe_reason(it.path)
        if motivo:
            print(f"   ⛔ OMITIDO {it.path} — {motivo}")
            continue
        ok, detalle = remove(it.path)
        if ok:
            liberado += it.size
            extra = f" ({detalle})" if detalle else ""
            print(f"   ✅ {it.label}{extra}")
        else:
            fallos.append((it, detalle))
            print(f"   ⚠️  {it.label} — {detalle}")

    if not args.skip_docker:
        docker_cleanup(dry_run=False)

    # --- Resumen -----------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"  Liberado: ~{human(liberado)}")
    if RESCUE_DIR.exists():
        print(f"  Modelos .dfm rescatados en: {RESCUE_DIR}")
    if not purge:
        print("  Tus caras, salidas y entrenamientos siguen en disco.")
    if fallos:
        print(f"\n  ⚠️  {len(fallos)} elemento(s) no se pudieron borrar:")
        for it, detalle in fallos:
            print(f"     · {it.path} — {detalle}")
        print("  Suele ser un archivo en uso: cerrá la app/el explorador y repetí.")

    if repo_item is not None:
        print("\n🗑️  Borrando el código fuente…")
        ok, detalle = remove_repo_tree()
        if ok:
            print("  ✅ Carpeta del proyecto vaciada; termina de borrarse sola en "
                  "unos segundos, al cerrarse esta ventana.")
        else:
            print(f"  ⚠️  {detalle}")

    print("\n✅ Fuser desinstalado." + ("" if fallos else " No queda nada instalado."))
    print("=" * 70)
    return 0 if not fallos else 1


if __name__ == "__main__":
    raise SystemExit(main())
