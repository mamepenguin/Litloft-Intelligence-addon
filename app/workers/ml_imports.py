"""One-at-a-time first import of the libraries the encoders are built on.

transformers initialises lazily. A second thread that asks for one of its names during the
first thread's initialisation gets an ImportError instead of waiting, so loaders that run
concurrently must not each import it on their own.
"""

import importlib
import threading

_MODULES = ("transformers", "sentence_transformers")

_lock = threading.Lock()
_imported = False


def import_ml_libraries() -> None:
    global _imported
    with _lock:
        if _imported:
            return
        for name in _MODULES:
            importlib.import_module(name)
        _imported = True
