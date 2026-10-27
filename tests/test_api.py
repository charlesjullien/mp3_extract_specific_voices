from fastapi.testclient import TestClient

from app.main import app


def test_health():
    with TestClient(app) as client:
        response = client.get("/api/health")
    assert response.status_code == 200
    assert "ffmpeg" in response.json()
    assert "hf_token_configured" in response.json()


def test_requires_input():
    with TestClient(app) as client:
        response = client.post("/api/jobs", data={})
    assert response.status_code == 400
    assert "YouTube" in response.json()["detail"]


def test_rejects_non_youtube():
    with TestClient(app) as client:
        response = client.post("/api/jobs", data={"url": "https://example.com/video"})
    assert response.status_code == 400


def test_rejects_bad_speaker_count_before_download():
    with TestClient(app) as client:
        response = client.post(
            "/api/jobs",
            data={
                "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                "num_speakers": "0",
            },
        )
    assert response.status_code == 400


def test_rejects_text_file():
    with TestClient(app) as client:
        response = client.post(
            "/api/jobs",
            files={"file": ("notes.txt", b"hello", "text/plain")},
        )
    assert response.status_code == 400


def test_rejects_link_and_file_together():
    with TestClient(app) as client:
        response = client.post(
            "/api/jobs",
            data={"url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"},
            files={"file": ("a.mp3", b"abc", "audio/mpeg")},
        )
    assert response.status_code == 400


def test_page_is_french():
    with TestClient(app) as client:
        response = client.get("/")
    assert response.status_code == 200
    assert "Extraction de voix" in response.text
    assert "Transcrire" in response.text
