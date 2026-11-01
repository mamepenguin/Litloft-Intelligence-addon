"""Loading the text and image encoders at the same time on a cold process.

SPEC-ADDON-041 runs the image encoder beside ``/search``'s text encoder, and
importing transformers from two threads at once raises ImportError in the loser.
"""

import threading
import time

import pytest

from app.workers import clip, embedder, ml_imports


@pytest.fixture(autouse=True)
def cold(monkeypatch):
    monkeypatch.setattr(ml_imports, "_imported", False)


def test_spec_addon_041_concurrent_callers_import_one_at_a_time_and_once(monkeypatch):
    active = 0
    peak = 0
    calls = []
    guard = threading.Lock()

    def slow_import(name):
        nonlocal active, peak
        with guard:
            active += 1
            peak = max(peak, active)
            calls.append(name)
        time.sleep(0.01)
        with guard:
            active -= 1

    monkeypatch.setattr(ml_imports.importlib, "import_module", slow_import)
    threads = [threading.Thread(target=ml_imports.import_ml_libraries) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert peak == 1
    assert calls == list(ml_imports._MODULES)


def test_spec_addon_041_a_failed_import_is_retried_by_the_next_caller(monkeypatch):
    def fail(name):
        raise ImportError(name)

    monkeypatch.setattr(ml_imports.importlib, "import_module", fail)
    with pytest.raises(ImportError):
        ml_imports.import_ml_libraries()

    seen = []
    monkeypatch.setattr(ml_imports.importlib, "import_module", seen.append)
    ml_imports.import_ml_libraries()
    assert seen == list(ml_imports._MODULES)


class _ImportedFirst(Exception):
    pass


@pytest.mark.parametrize(
    ("module", "state", "load"),
    [
        (embedder, {"_loaded": False, "_model": None}, embedder._ensure_loaded),
        (clip, {"_loaded": False, "_model": None}, clip._ensure_loaded),
        (clip, {"_siglip_text_loaded": False, "_siglip_text_model": None}, clip._ensure_siglip_text_loaded),
    ],
    ids=["text encoder", "open_clip encoder", "siglip text encoder"],
)
def test_spec_addon_041_every_encoder_loader_goes_through_the_shared_import(monkeypatch, module, state, load):
    for name, value in state.items():
        monkeypatch.setattr(module, name, value)

    def stop():
        raise _ImportedFirst

    monkeypatch.setattr(ml_imports, "import_ml_libraries", stop)
    with pytest.raises((_ImportedFirst, RuntimeError)) as caught:
        load()
    assert isinstance(caught.value, _ImportedFirst) or isinstance(caught.value.__cause__, _ImportedFirst)
