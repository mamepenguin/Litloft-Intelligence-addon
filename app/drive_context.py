"""Drive context helpers for drive-scoped routes.

Every public route in this addon operates within a single Litloft
drive, identified by the ``X-HV-Drive`` request header. The header is
set by Litloft's Generic Addon Proxy when the request arrives via
``/drive/{drive}/addons/intelligence/...`` and validated against the
caller's accessible drive set there.

Routers should call ``require_drive(x_hv_drive)`` once and then pass
the canonical drive name to every downstream call.
"""

from typing import Annotated
from urllib.parse import unquote

from fastapi import Header, HTTPException


def require_drive(
    x_hv_drive: Annotated[str | None, Header(alias="X-HV-Drive")] = None,
) -> str:
    """Return the canonical drive name from the X-HV-Drive header.

    Raises 400 if the header is absent. Header values are restricted to
    ISO-8859-1, so the frontend percent-encodes drive names; we decode
    once here so downstream code sees the original UTF-8 name.
    """
    if not x_hv_drive:
        raise HTTPException(status_code=400, detail="Drive context required")
    return unquote(x_hv_drive)


def assert_file_in_drive(file_drive: str, drive: str) -> None:
    """Raise 404 when a file's drive doesn't match the request drive.

    Returns 404 (not 403) so we never leak the existence of files in
    other drives — matches the host's drive_access pattern.
    """
    if file_drive != drive:
        raise HTTPException(status_code=404, detail="File not found")
