import pytest

from app.validate import (
    ValidationError,
    audio_extension,
    parse_num_speakers,
    sanitize_speaker_id,
    validate_youtube_url,
)


def test_watch_url():
    url = validate_youtube_url("  https://www.youtube.com/watch?v=dQw4w9WgXcQ  ")
    assert url == "https://www.youtube.com/watch?v=dQw4w9WgXcQ"


def test_short_music_and_mobile():
    validate_youtube_url("https://youtu.be/dQw4w9WgXcQ")
    validate_youtube_url("https://music.youtube.com/watch?v=dQw4w9WgXcQ")
    validate_youtube_url("https://m.youtube.com/shorts/dQw4w9WgXcQ")


def test_rejects_other_hosts():
    with pytest.raises(ValidationError):
        validate_youtube_url("https://vimeo.com/123456789")


def test_rejects_lookalike():
    with pytest.raises(ValidationError):
        validate_youtube_url("https://www.youtube.com.evil.com/watch?v=dQw4w9WgXcQ")


def test_rejects_userinfo():
    with pytest.raises(ValidationError):
        validate_youtube_url("https://user@www.youtube.com/watch?v=dQw4w9WgXcQ")


def test_rejects_playlist_only():
    with pytest.raises(ValidationError):
        validate_youtube_url("https://www.youtube.com/playlist?list=PL1234567890")


def test_rejects_channel():
    with pytest.raises(ValidationError):
        validate_youtube_url("https://www.youtube.com/@someone")


def test_num_speakers():
    assert parse_num_speakers("") is None
    assert parse_num_speakers(None) is None
    assert parse_num_speakers("3") == 3
    with pytest.raises(ValidationError):
        parse_num_speakers("0")
    with pytest.raises(ValidationError):
        parse_num_speakers("21")
    with pytest.raises(ValidationError):
        parse_num_speakers("deux")


def test_audio_extension():
    assert audio_extension("voix.MP3") == ".mp3"
    assert audio_extension("note.wav") == ".wav"
    assert audio_extension("clip.m4a") == ".m4a"
    with pytest.raises(ValidationError):
        audio_extension("notes.txt")


def test_speaker_ids_stay_unique():
    used = set()
    first = sanitize_speaker_id("SPEAKER_00", 1, used)
    second = sanitize_speaker_id("SPEAKER_00", 2, used)
    assert first == "SPEAKER_00"
    assert second != first
    assert sanitize_speaker_id("../etc", 3, used) == "etc"
