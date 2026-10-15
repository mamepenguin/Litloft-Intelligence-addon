"""Where the SPEC-ADDON-021 tests reach the IDF term backfill.

The spec names the behavior but not these entry points, so every name the
tests depend on is bound here and nowhere else. Renaming one in the
implementation means editing this file only; no assertion lives here.
"""

from __future__ import annotations

from app import database, tfidf


def run_pass() -> None:
    """One backfill pass, synchronously, on the calling thread."""
    tfidf.run_idf_backfill_pass()


async def run_loop() -> None:
    """The per-process backfill loop the lifespan starts."""
    await tfidf.idf_backfill_loop()


def request() -> None:
    tfidf.request_idf_backfill()


def request_pending() -> bool:
    return tfidf.idf_backfill_requested()


def reset_state() -> None:
    """Forget the request flag and the files held after failing on their own,
    as a fresh process would."""
    tfidf.reset_idf_backfill_state()


def rules_version() -> str:
    return tfidf.IDF_TERM_RULES_VERSION


def check_rules() -> None:
    """The startup rule-version check the lifespan runs."""
    database.check_idf_term_rules()


def set_timings(
    monkeypatch,
    *,
    child_timeout: float | None = None,
    coalesce: float | None = None,
    retry: float | None = None,
) -> None:
    """Shorten the 300 s child limit, the 5 s coalescing wait and the 60 s
    retry wait."""
    if child_timeout is not None:
        monkeypatch.setattr(tfidf, "_IDF_BACKFILL_CHILD_TIMEOUT_SECONDS", child_timeout)
    if coalesce is not None:
        monkeypatch.setattr(tfidf, "_IDF_BACKFILL_COALESCE_SECONDS", coalesce)
    if retry is not None:
        monkeypatch.setattr(tfidf, "_IDF_BACKFILL_RETRY_SECONDS", retry)
