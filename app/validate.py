"""Contrôles des liens, des fichiers et des identifiants de voix."""

from __future__ import annotations

import re
from urllib.parse import parse_qs, urlparse

class ValidationError(ValueError):
    pass


_ALLOWED_HOSTS = {
    "youtube.com",
    "www.youtube.com",
    "m.youtube.com",
    "music.youtube.com",
    "www.music.youtube.com",
    "youtu.be",
}
_VIDEO_ID = re.compile(r"^[A-Za-z0-9_-]{6,32}$")
_SPEAKER_ID = re.compile(r"^[A-Za-z0-9_-]{1,32}$")
_ALLOWED_EXTENSIONS = {".mp3", ".wav", ".m4a"}


def validate_youtube_url(value: str) -> str:
    url = (value or "").strip()
    if not url or len(url) > 500:
        raise ValidationError("Collez un lien YouTube valide.")
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or parsed.username or parsed.password or parsed.port:
        raise ValidationError("Seuls les liens YouTube https sont acceptés.")
    host = (parsed.hostname or "").lower()
    if host not in _ALLOWED_HOSTS:
        raise ValidationError("Seuls les liens YouTube sont acceptés.")
    if _video_id(parsed) is None:
        raise ValidationError("Ce lien ne pointe pas vers une seule vidéo.")
    return url


def audio_extension(filename: str) -> str:
    suffix = ""
    if filename:
        dot = filename.rfind(".")
        if dot != -1:
            suffix = filename[dot:].lower()
    if suffix not in _ALLOWED_EXTENSIONS:
        raise ValidationError("Le fichier doit être un MP3, un WAV ou un M4A.")
    return suffix


def parse_num_speakers(value: str | None) -> int | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if not text.isdigit():
        raise ValidationError("Le nombre de personnes doit être un entier entre 1 et 20.")
    number = int(text)
    if number < 1 or number > 20:
        raise ValidationError("Le nombre de personnes doit être entre 1 et 20.")
    return number


def validate_speaker_id(value: str) -> str:
    if not _SPEAKER_ID.fullmatch(value or ""):
        raise ValidationError("Voix inconnue.")
    return value


def sanitize_speaker_id(raw: str, index: int, used: set[str]) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]", "", str(raw))[:32]
    if not cleaned:
        cleaned = f"voix{index}"
    base = cleaned
    suffix_n = 2
    while cleaned in used:
        suffix = f"-{suffix_n}"
        cleaned = f"{base[: 32 - len(suffix)]}{suffix}"
        suffix_n += 1
    used.add(cleaned)
    return cleaned


def clean_title(value: str) -> str:
    cleaned = re.sub(r"[\x00-\x1f]", " ", value or "").strip()
    return cleaned[:180] or "Audio"


def _video_id(parsed) -> str | None:
    host = (parsed.hostname or "").lower()
    if host == "youtu.be":
        parts = [part for part in parsed.path.split("/") if part]
        candidate = parts[0] if parts else ""
        return candidate if _VIDEO_ID.fullmatch(candidate) else None

    query = parse_qs(parsed.query)
    if "v" in query and query["v"] and query["v"][0]:
        candidate = query["v"][0]
        return candidate if _VIDEO_ID.fullmatch(candidate) else None

    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) >= 2 and parts[0] in {"shorts", "embed", "live", "v"}:
        candidate = parts[1]
        return candidate if _VIDEO_ID.fullmatch(candidate) else None
    return None
