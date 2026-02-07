from main import app


def test_health():
    # Import TestClient lazily to avoid importing httpx at module import time (compatibility with Python 3.14)
    from fastapi.testclient import TestClient

    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert "status" in data
    assert "model_trained" in data
    assert "transcription_available" in data


def test_upload_without_transcription():
    from fastapi.testclient import TestClient

    client = TestClient(app)
    # Create a small dummy audio file (content does not need to be valid audio because transcription may be disabled)
    files = {"file": ("test.wav", b"RIFF....", "audio/wav")}
    r = client.post("/upload-audio", files=files)
    # Depending on environment: either transcription is disabled (200), transcription fails (200 with error), or the service rejects invalid file types (400)
    assert r.status_code in (200, 400)
    data = r.json()

    if r.status_code == 200:
        assert data["success"] is True
        assert "transcription_available" in data
        if data.get("transcription_available"):
            # If transcription is available, the endpoint records success or failure explicitly
            assert "transcription_success" in data
            assert isinstance(data["transcription_success"], bool)
            if data["transcription_success"]:
                assert isinstance(data["transcript"], str)
            else:
                assert "transcription_error" in data
                assert isinstance(data["transcription_error"], str)
        else:
            assert data["transcription_available"] is False
    else:
        # HTTP 400 should include a clear detail message
        assert "detail" in data
        assert isinstance(data["detail"], str)
        assert data["detail"].lower().startswith("audio") or "audio" in data["detail"].lower()

