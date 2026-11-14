"""SPEC-ADDON-109: an answer about a video with no text cites its frames by number."""

from __future__ import annotations

import pytest

from tests.file_question_vision_doubles import (  # noqa: F401 - fixtures
    addon,
    ask,
    by_kind,
    call,
    db,
    llm,
    vision_client,
)
from tests.vec_partition_db import search_index  # noqa: F401 - the fixture
from tests.video_frames_doubles import (
    FFmpeg,
    listed_in_order,
    ms,
    scenes,
    seed_text_less_video,
    sent_keys,
    user_prompt,
)

VIDEO = "vid000000109"


def _answering(text: str):
    def script(body):
        return 200, text
    return script


def _cited(response) -> list[tuple[int, dict | None]]:
    entries = by_kind(response).get("citations", [])
    return [(entry["n"], entry["position"]) for entry in entries]


class TestFrameCitations:
    @pytest.mark.parametrize(
        ("answer", "expected"),
        [
            ("A [1]. B [3]. C [1]. D [9]. E [0].", [(1, 10.0), (3, 30.0)]),
            ("Only [2] here.", [(2, 20.0)]),
            ("Nothing cited.", []),
            ("Out of range [4] and [0].", []),
        ],
        ids=["first-named-order-once-in-range", "one", "none", "only-out-of-range"],
    )
    async def test_each_named_frame_is_cited_once_with_its_time(
        self, answer, expected, db, addon, llm, call, tmp_path, monkeypatch,
    ):
        """SPEC-ADDON-109 I4 (053 I3 revised): ``1 <= n <= m``, first-named order, frame time."""
        seed_text_less_video(db, VIDEO, path=tmp_path / "v.mp4", duration=None)
        scenes(monkeypatch, [10.0, 20.0, 30.0], duration=None)
        FFmpeg().install(monkeypatch)
        vision_client(llm, script=_answering(answer))

        response = await ask(call, VIDEO)

        assert by_kind(response)["meta"]["image"] == "attached"
        assert _cited(response) == [(n, {"seconds": t}) for n, t in expected]

    async def test_a_dropped_frame_shifts_the_numbers_to_the_frames_sent(
        self, db, addon, llm, call, tmp_path, monkeypatch,
    ):
        """SPEC-ADDON-109 I4: numbers follow the kept frames, so ``[2]`` is the third chosen."""
        seed_text_less_video(db, VIDEO, path=tmp_path / "v.mp4", duration=None)
        scenes(monkeypatch, [10.0, 20.0, 30.0], duration=None)
        FFmpeg(fail={ms(20.0)}).install(monkeypatch)
        _, wire = vision_client(llm, script=_answering("X [2]. Y [3]. Z [1]."))

        response = await ask(call, VIDEO)

        assert sent_keys(wire) == [ms(10.0), ms(30.0)]
        assert _cited(response) == [(2, {"seconds": 30.0}), (1, {"seconds": 10.0})]

    async def test_a_frame_time_is_rounded_to_the_millisecond_everywhere(
        self, db, addon, llm, call, tmp_path, monkeypatch,
    ):
        """SPEC-ADDON-109 / 108 I2: cache key, prompt stamp and ``position.seconds`` agree."""
        seed_text_less_video(db, VIDEO, path=tmp_path / "v.mp4", duration=None)
        scenes(monkeypatch, [12.3456, 3725.9], duration=None)
        FFmpeg().install(monkeypatch)
        _, wire = vision_client(llm, script=_answering("See [2] then [1]."))

        response = await ask(call, VIDEO)

        assert sent_keys(wire) == [12346, 3725900]
        assert listed_in_order(user_prompt(wire), ["0:12", "1:02:05"])
        assert _cited(response) == [
            (2, {"seconds": 3725.9}),
            (1, {"seconds": 12.346}),
        ]

    async def test_a_cited_position_frame_carries_the_position(
        self, db, addon, llm, call, tmp_path, monkeypatch,
    ):
        """SPEC-ADDON-109 I4: the position frame is numbered in time order like the rest."""
        seed_text_less_video(db, VIDEO, path=tmp_path / "v.mp4", duration=None)
        scenes(monkeypatch, [10.0, 30.0], duration=None)
        FFmpeg().install(monkeypatch)
        _, wire = vision_client(llm, script=_answering("At that moment [2]."))

        response = await ask(call, VIDEO, position=21.5)

        assert sent_keys(wire) == [ms(10.0), ms(21.5), ms(30.0)]
        assert _cited(response) == [(2, {"seconds": 21.5})]
