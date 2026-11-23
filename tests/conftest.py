import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app

ROOT = Path(__file__).resolve().parents[1]


class MockPredictor:
    """MOCK. A deterministic fake that exists only so the API can be tested offline. It is NOT a trained model."""
    version = "mock-test-fixture"
    backbone = "mock-backbone"

    def __init__(self, scores=None, fn=None, raises=None, raw=None):
        self.scores, self.fn, self.raises, self.raw = scores, fn, raises, raw
        self.calls = []

    def predict(self, originals, simplified, languages):
        self.calls.append((list(originals), list(simplified), list(languages)))
        if self.raises:
            raise self.raises
        if self.raw is not None:
            return self.raw
        if self.fn:
            return [self.fn(o, s) for o, s in zip(originals, simplified)]
        return list(self.scores)[:len(originals)]


def make_settings(tmp_path, **kw):
    return Settings(model_dir=tmp_path / "model", baseline_dir=tmp_path / "baselines",
                    results_dir=tmp_path / "results", **kw)


@pytest.fixture
def settings(tmp_path):
    return make_settings(tmp_path)


@pytest.fixture
def client(settings):
    with TestClient(create_app(settings)) as c:
        yield c


@pytest.fixture
def make_client(tmp_path):
    """Factory: make_client(predictor=None, baselines=None, **settings_overrides) -> started TestClient."""
    opened = []

    def _make(predictor=None, baselines=None, raise_server_exceptions=True, **kw):
        c = TestClient(create_app(make_settings(tmp_path, **kw), predictor=predictor, baselines=baselines),
                       raise_server_exceptions=raise_server_exceptions)
        c.__enter__()
        opened.append(c)
        return c

    yield _make
    for c in opened:
        c.__exit__(None, None, None)


PAIR = {"original_text": "The tenant must pay rent by the fifth day of each month.",
        "simplified_text": "The tenant may pay rent by the tenth day of each month."}


@pytest.fixture(scope="session")
def live_url(tmp_path_factory):
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    env = {"PATH": "/usr/bin:/bin:/usr/local/bin", "PYTHONPATH": str(ROOT), "HOME": str(tmp_path_factory.mktemp("home")),
           "JUDGEBERT_MODEL_DIR": str(tmp_path_factory.mktemp("m")), "JUDGEBERT_BASELINE_DIR": str(tmp_path_factory.mktemp("b")),
           "JUDGEBERT_RESULTS_DIR": str(tmp_path_factory.mktemp("r"))}
    proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "app.main:app", "--port", str(port), "--log-level", "warning"],
                            cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    url = f"http://127.0.0.1:{port}"
    try:
        for _ in range(60):
            try:
                if httpx.get(url + "/health", timeout=1).status_code == 200:
                    break
            except httpx.HTTPError:
                time.sleep(0.25)
        else:
            pytest.skip("could not start a local uvicorn server")
        yield url
    finally:
        proc.terminate()
        proc.wait(timeout=10)
