"""SPEC-ADDON-071 I11: a non-finite number never turns a 422 into a 500.

The file-question route's ``position_seconds`` is held in
``test_spec_addon_072_position.py``. These rows use the two drive-wide routes,
which share nothing with it but the addon's ``RequestValidationError`` handler.
The numeric field is left as a raw JSON token because the Python encoder
cannot spell the non-finite values.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from app.main import app

DRIVE = "family"
ASK = "/ask"
FIND = "/find"

BODIES = {
    ASK: '{"query": "what is on the table", "top_k": %s}',
    FIND: '{"question": "what is on the table", "limit": %s}',
}

ROWS = [
    (ASK, "NaN", "ask-nan"),
    (ASK, "Infinity", "ask-infinity"),
    (ASK, "-Infinity", "ask-negative-infinity"),
    (ASK, '"many"', "ask-wrong-type"),
    (FIND, "NaN", "find-nan"),
    (FIND, "Infinity", "find-infinity"),
    (FIND, "-Infinity", "find-negative-infinity"),
    (FIND, '"many"', "find-wrong-type"),
]


@pytest.fixture()
def post():
    client = TestClient(app, raise_server_exceptions=False)

    def _post(route: str, value: str):
        return client.post(
            route,
            content=(BODIES[route] % value).encode(),
            headers={"Content-Type": "application/json", "X-Lit-Drive": DRIVE},
        )

    return _post


def _json(response):
    """Parse strictly: ``NaN`` and ``Infinity`` are not JSON tokens."""
    return json.loads(
        response.text,
        parse_constant=lambda constant: pytest.fail(
            f"{constant} is not JSON"
        ),
    )


@pytest.mark.parametrize(("route", "value"), [(r, v) for r, v, _ in ROWS],
                         ids=[name for _, _, name in ROWS])
def test_a_non_finite_number_is_422_and_an_ordinary_error_keeps_its_shape(
    post, route, value,
):
    """SPEC-ADDON-071 I11: 422 everywhere, and the other errors are untouched."""
    response = post(route, value)

    assert response.status_code == 422, response.text
    detail = _json(response)["detail"]
    assert isinstance(detail, list)
    assert detail
    for error in detail:
        assert {"loc", "msg", "type"} <= set(error)
