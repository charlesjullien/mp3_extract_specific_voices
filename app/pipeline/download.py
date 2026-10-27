"""Téléchargement de l'audio YouTube, limité aux hôtes autorisés."""

from __future__ import annotations

from pathlib import Path

from app.validate import clean_title, validate_youtube_url


class _SilentLogger:
    def debug(self, msg: str) -> None:
        return None

    def info(self, msg: str) -> None:
        return None

    def warning(self, msg: str) -> None:
        return None

    def error(self, msg: str) -> None:
        return None


def download_audio(url: str, dest_dir: Path) -> tuple[Path, str]:
    validate_youtube_url(url)
    try:
        import yt_dlp
    except ImportError as exc:
        raise RuntimeError(
            "yt-dlp n'est pas installé. Exécutez : pip install -r requirements.txt"
        ) from exc

    dest_dir.mkdir(parents=True, exist_ok=True)
    options = {
        "format": "bestaudio/best",
        "outtmpl": str(dest_dir / "source.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "restrictfilenames": True,
        "cachedir": False,
        "retries": 3,
        "fragment_retries": 3,
        "socket_timeout": 30,
        "logger": _SilentLogger(),
        "writesubtitles": False,
        "writeautomaticsub": False,
    }
    with yt_dlp.YoutubeDL(options) as ydl:
        info = ydl.extract_info(url, download=True)
        if not info:
            raise RuntimeError("Impossible de lire cette vidéo.")
        filename = Path(ydl.prepare_filename(info))

    if not filename.exists():
        candidates = sorted(dest_dir.glob("source.*"))
        if not candidates:
            raise RuntimeError("Le téléchargement n'a pas produit de fichier audio.")
        filename = candidates[0]

    path = filename.resolve()
    if path.parent != dest_dir.resolve():
        raise RuntimeError("Le téléchargement a écrit hors du dossier de travail.")
    title = clean_title(str(info.get("title") or "Vidéo YouTube"))
    return path, title
