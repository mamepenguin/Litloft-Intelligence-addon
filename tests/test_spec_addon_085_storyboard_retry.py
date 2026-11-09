"""SPEC-ADDON-085: an explicit reindex retries a failed YouTube storyboard.

The storyboard leg runs through the real ``yt-dlp``/sheet doubles of
``tests.youtube_frame_doubles`` and the real Pillow of
``tests.test_spec_addon_078_youtube_storyboard``; the reindex route is reached
through the addon's FastAPI app with a queue double, so every assertion is on a
lookup count, an HTTP answer, a row, a tile or a flag.
"""

from __future__ import annotations

import threading
import time

import httpx
import pytest

from app.frame_cache import frame_cache_dir
from app.models import IndexedFile
from tests.search_endpoint_db import patch_everywhere
from tests.test_spec_addon_078_youtube_storyboard import (  # noqa: F401 - fixtures
    add_loft,
    flags_of,
    new_id,
    pil,
    rows_of,
    run,
    serve_storyboard,
    storyboard_fps,
    world,
)
from tests.test_spec_addon_079_scene_tiles import tile_names
from tests.youtube_frame_doubles import (  # noqa: F401 - fixtures
    describe,
    youtube,
)

DRIVE = "family"
OTHER_DRIVE = "elsewhere"
LOOKUP_FAILURE = "ERROR: [youtube] x: Video unavailable"

ALLOWED = "metadata, clip, whisper, text, ocr, frame_ocr, page_ocr"
EMPTY_TASKS = (
    "Request body must include a non-empty 'tasks' array. "
    f"Allowed: {ALLOWED}"
)
UNKNOWN_TASK = f"Unknown task: 'bogus'. Allowed: {ALLOWED}"
OCR_ON_LOFT = (
    "Task 'ocr' applies only to pictures and is unavailable "
    "while features.image_ocr is false"
)
OVERWRITE_NOT_BOOL = "overwrite_corrections must be a boolean"


class Queue:
    def __init__(self) -> None:
        self.enqueued: list[tuple[str, str]] = []
        self.queued_pairs: set[tuple[str, str]] = set()

    def is_queued(self, file_id: str, task: str) -> bool:
        return (file_id, task) in self.queued_pairs

    async def enqueue_task_for_file(self, file_id: str, task: str) -> None:
        self.enqueued.append((file_id, task))


class Api:
    def __init__(self, client: httpx.AsyncClient, queue: Queue) -> None:
        self.client = client
        self.queue = queue

    async def reindex(self, file_id: str, body: dict, *, drive: str = DRIVE):
        return await self.client.post(
            f"/files/{file_id}/reindex",
            json=body,
            headers={"X-Lit-Drive": drive},
        )


@pytest.fixture()
async def api(world, monkeypatch):
    import app.config

    test_settings = app.config.settings
    from app.dependencies import get_index_manager
    from app.main import app

    # The app is imported after ``world`` installed its settings; modules that
    # were not loaded yet read the test ones from ``app.config`` already.
    patch_everywhere(monkeypatch, "settings", test_settings)

    queue = Queue()
    app.dependency_overrides[get_index_manager] = lambda: queue
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    client = httpx.AsyncClient(transport=transport, base_url="http://intelligence")
    try:
        yield Api(client, queue)
    finally:
        await client.aclose()
        app.dependency_overrides.pop(get_index_manager, None)


def accepted(file_id: str, tasks: list[str]) -> dict:
    return {"status": "accepted", "file_id": file_id, "tasks_reset": tasks}


def already_queued(file_id: str) -> dict:
    return {"status": "already_queued", "file_id": file_id, "tasks_reset": []}


def move_drive(world, file_id: str, drive: str) -> None:
    with world.Session() as session:
        session.query(IndexedFile).filter_by(file_id=file_id).update(
            {"drive": drive}
        )
        session.commit()


def mark_corrected(world, file_id: str, *, redo: bool = False) -> None:
    with world.Session() as session:
        session.query(IndexedFile).filter_by(file_id=file_id).update(
            {"transcript_corrected": True, "transcript_redo": redo}
        )
        session.commit()


def install_successful_storyboard(world, youtube, *, tiles: int = 9):
    world.configure(min_interval=10, max_frames=500)
    seconds = tiles * 10.0
    board, per_tile = serve_storyboard(
        youtube, world.pil, tiles=tiles, fps=storyboard_fps(seconds),
    )
    youtube.set_ytdlp(describe=describe([board], duration=seconds))
    return board, per_tile


def fail_leg(youtube, file_id: str, *, before: int = 0) -> None:
    youtube.set_ytdlp(fail=LOOKUP_FAILURE)
    assert run(file_id) is False
    assert len(youtube.lookups()) == before + 1


def failed_loft(world, youtube, *, drive: str = DRIVE, before: int = 0) -> str:
    file_id = new_id()
    add_loft(world, file_id, clip_thumbnail_indexed=True)
    if drive != DRIVE:
        move_drive(world, file_id, drive)
    fail_leg(youtube, file_id, before=before)
    return file_id


def wait_for(predicate, timeout: float = 15.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("the running leg never reached its lookup")


class TestOneRequestLetsTheNextLegCallYouTube:
    async def test_a_clip_request_after_a_failure_makes_the_next_leg_look_up_once(
        self, world, youtube, api,
    ):
        file_id = failed_loft(world, youtube)
        install_successful_storyboard(world, youtube)

        response = await api.reindex(file_id, {"tasks": ["clip"]})

        assert response.status_code == 200, response.text
        assert response.json() == accepted(file_id, ["clip"])
        before = len(youtube.lookups())
        assert run(file_id) is True
        assert len(youtube.lookups()) == before + 1

    async def test_an_already_queued_clip_request_makes_the_next_leg_look_up_once(
        self, world, youtube, api,
    ):
        file_id = failed_loft(world, youtube)
        install_successful_storyboard(world, youtube)
        api.queue.queued_pairs.add((file_id, "clip"))

        response = await api.reindex(file_id, {"tasks": ["clip"]})

        assert response.status_code == 202, response.text
        assert response.json() == already_queued(file_id)
        assert api.queue.enqueued == []
        before = len(youtube.lookups())
        assert run(file_id) is True
        assert len(youtube.lookups()) == before + 1


class TestARequestWhileTheLegRuns:
    async def test_the_next_leg_looks_up_after_the_running_leg_fails(
        self, world, youtube, api,
    ):
        file_id = new_id()
        add_loft(world, file_id, clip_thumbnail_indexed=True)
        install_successful_storyboard(world, youtube)
        youtube.set_ytdlp(fail=LOOKUP_FAILURE, delay=3.0)
        outcome: dict = {}

        def leg() -> None:
            outcome["result"] = run(file_id)

        thread = threading.Thread(target=leg, daemon=True)
        thread.start()
        wait_for(lambda: len(youtube.lookups()) == 1)

        response = await api.reindex(file_id, {"tasks": ["clip"]})

        assert response.status_code == 200, response.text
        assert response.json() == accepted(file_id, ["clip"])
        assert thread.is_alive(), "the request must be answered while the leg runs"
        thread.join(timeout=30)
        assert not thread.is_alive()
        assert outcome["result"] is False

        install_successful_storyboard(world, youtube)
        before = len(youtube.lookups())
        assert run(file_id) is True
        assert len(youtube.lookups()) == before + 1


class TestNoRequestKeepsTheGuard:
    async def test_a_leg_without_a_request_after_a_failure_makes_no_lookup(
        self, world, youtube,
    ):
        file_id = failed_loft(world, youtube)
        install_successful_storyboard(world, youtube)

        before = len(youtube.lookups())
        assert run(file_id) is False
        assert len(youtube.lookups()) == before

    async def test_a_file_with_no_request_stays_guarded_while_another_is_retried(
        self, world, youtube, api,
    ):
        first = failed_loft(world, youtube)
        second = failed_loft(world, youtube, before=len(youtube.lookups()))
        install_successful_storyboard(world, youtube)

        response = await api.reindex(first, {"tasks": ["clip"]})

        assert response.status_code == 200, response.text
        assert run(first) is True
        before = len(youtube.lookups())
        assert run(second) is False
        assert len(youtube.lookups()) == before


class TestOneRequestPermitsOneLeg:
    async def test_a_retried_leg_that_fails_again_leaves_the_next_leg_guarded(
        self, world, youtube, api,
    ):
        file_id = failed_loft(world, youtube)
        response = await api.reindex(file_id, {"tasks": ["clip"]})
        assert response.status_code == 200, response.text
        youtube.set_ytdlp(fail=LOOKUP_FAILURE)

        before = len(youtube.lookups())
        assert run(file_id) is False
        assert len(youtube.lookups()) == before + 1
        assert run(file_id) is False
        assert len(youtube.lookups()) == before + 1

    async def test_a_retried_leg_that_raises_leaves_the_next_leg_guarded(
        self, world, youtube, api,
    ):
        file_id = failed_loft(world, youtube)
        install_successful_storyboard(world, youtube)
        response = await api.reindex(file_id, {"tasks": ["clip"]})
        assert response.status_code == 200, response.text
        cache = frame_cache_dir(file_id)
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text("a file where the tile directory belongs")

        before = len(youtube.lookups())
        with pytest.raises(OSError):
            run(file_id)
        assert len(youtube.lookups()) == before + 1
        assert run(file_id) is False
        assert len(youtube.lookups()) == before + 1


class TestRequestsThatRecordNoRetry:
    async def test_a_request_without_clip_records_no_retry(
        self, world, youtube, api,
    ):
        file_id = failed_loft(world, youtube)
        install_successful_storyboard(world, youtube)

        response = await api.reindex(file_id, {"tasks": ["text"]})

        assert response.status_code == 200, response.text
        before = len(youtube.lookups())
        assert run(file_id) is False
        assert len(youtube.lookups()) == before

    @pytest.mark.parametrize(
        "body",
        [
            {"tasks": []},
            {"tasks": ["bogus"]},
            {"tasks": ["clip", "ocr"]},
            {"tasks": ["clip"], "overwrite_corrections": "yes"},
        ],
        ids=["empty", "unknown-task", "ocr-on-loft", "overwrite-not-bool"],
    )
    async def test_a_refused_request_records_no_retry(
        self, body, world, youtube, api,
    ):
        file_id = failed_loft(world, youtube)
        install_successful_storyboard(world, youtube)

        response = await api.reindex(file_id, body)

        assert response.status_code == 422, response.text
        before = len(youtube.lookups())
        assert run(file_id) is False
        assert len(youtube.lookups()) == before

    async def test_a_request_for_another_drives_file_records_no_retry(
        self, world, youtube, api,
    ):
        file_id = failed_loft(world, youtube, drive=OTHER_DRIVE)
        install_successful_storyboard(world, youtube)

        response = await api.reindex(file_id, {"tasks": ["clip"]})

        assert response.status_code == 404, response.text
        before = len(youtube.lookups())
        assert run(file_id) is False
        assert len(youtube.lookups()) == before

    async def test_a_corrected_file_without_confirmation_records_no_retry(
        self, world, youtube, api,
    ):
        file_id = failed_loft(world, youtube)
        mark_corrected(world, file_id)
        install_successful_storyboard(world, youtube)

        response = await api.reindex(file_id, {"tasks": ["clip", "whisper"]})

        assert response.status_code == 409, response.text
        before = len(youtube.lookups())
        assert run(file_id) is False
        assert len(youtube.lookups()) == before

    async def test_a_busy_corrected_file_records_no_retry(
        self, world, youtube, api,
    ):
        file_id = failed_loft(world, youtube)
        mark_corrected(world, file_id, redo=True)
        install_successful_storyboard(world, youtube)

        response = await api.reindex(
            file_id,
            {"tasks": ["clip", "whisper"], "overwrite_corrections": True},
        )

        assert response.status_code == 409, response.text
        before = len(youtube.lookups())
        assert run(file_id) is False
        assert len(youtube.lookups()) == before


class TestARetriedLegThatSucceeds:
    async def test_it_writes_the_same_rows_tiles_and_flags_as_a_first_success(
        self, world, youtube, api,
    ):
        retried = failed_loft(world, youtube)
        first = new_id()
        add_loft(world, first, clip_thumbnail_indexed=True)
        _, per_tile = install_successful_storyboard(world, youtube)

        response = await api.reindex(retried, {"tasks": ["clip"]})

        assert response.status_code == 200, response.text
        assert run(retried) is True
        assert run(first) is True

        rows = rows_of(world, retried)
        assert [ts for _, ts in rows] == pytest.approx(
            [n * per_tile for n in range(9)]
        )
        assert [ts for _, ts in rows] == [ts for _, ts in rows_of(world, first)]
        assert len({embedding_id for embedding_id, _ in rows}) == len(rows)
        assert tile_names(retried) == sorted(
            f"{int(round(ts * 1000))}.webp" for _, ts in rows
        )
        assert tile_names(retried) == tile_names(first)
        assert flags_of(world, retried)[0] is True


class TestTheRouteAnswersExactlyAsBefore:
    async def test_the_accepted_answer_is_unchanged(self, world, api):
        file_id = new_id()
        add_loft(world, file_id, clip_thumbnail_indexed=True)

        response = await api.reindex(file_id, {"tasks": ["clip"]})

        assert response.status_code == 200, response.text
        assert response.json() == accepted(file_id, ["clip"])
        assert api.queue.enqueued == [(file_id, "clip")]
        assert flags_of(world, file_id)[0] is False

    async def test_the_already_queued_answer_is_unchanged(self, world, api):
        file_id = new_id()
        add_loft(world, file_id, clip_indexed=True, clip_thumbnail_indexed=True)
        api.queue.queued_pairs.add((file_id, "clip"))

        response = await api.reindex(file_id, {"tasks": ["clip"]})

        assert response.status_code == 202, response.text
        assert response.json() == already_queued(file_id)
        assert api.queue.enqueued == []
        assert flags_of(world, file_id)[0] is True

    async def test_a_request_without_clip_is_answered_as_before(self, world, api):
        file_id = new_id()
        add_loft(world, file_id, clip_thumbnail_indexed=True)

        response = await api.reindex(file_id, {"tasks": ["text"]})

        assert response.status_code == 200, response.text
        assert response.json() == accepted(file_id, ["text"])

    @pytest.mark.parametrize(
        ("body", "detail"),
        [
            ({"tasks": []}, EMPTY_TASKS),
            ({"tasks": ["bogus"]}, UNKNOWN_TASK),
            ({"tasks": ["clip", "ocr"]}, OCR_ON_LOFT),
            ({"tasks": ["clip"], "overwrite_corrections": "yes"}, OVERWRITE_NOT_BOOL),
        ],
        ids=["empty", "unknown-task", "ocr-on-loft", "overwrite-not-bool"],
    )
    async def test_the_validation_answers_are_unchanged(self, body, detail, world, api):
        file_id = new_id()
        add_loft(world, file_id, clip_thumbnail_indexed=True)

        response = await api.reindex(file_id, body)

        assert response.status_code == 422, response.text
        assert response.json() == {"detail": detail}

    async def test_the_not_found_answer_is_unchanged(self, api):
        response = await api.reindex(new_id(), {"tasks": ["clip"]})

        assert response.status_code == 404, response.text
        assert response.json() == {"detail": "File not indexed"}

    async def test_the_cross_drive_answer_is_unchanged(self, world, api):
        file_id = new_id()
        add_loft(world, file_id, clip_thumbnail_indexed=True)
        move_drive(world, file_id, OTHER_DRIVE)

        response = await api.reindex(file_id, {"tasks": ["clip"]})

        assert response.status_code == 404, response.text
        assert response.json() == {"detail": "File not found"}

    @pytest.mark.parametrize(
        ("redo", "body", "detail"),
        [
            (
                False,
                {"tasks": ["clip", "whisper"]},
                {"code": "corrected"},
            ),
            (
                True,
                {"tasks": ["clip", "whisper"], "overwrite_corrections": True},
                {"code": "busy"},
            ),
        ],
        ids=["corrected", "busy"],
    )
    async def test_the_corrected_answers_are_unchanged(
        self, redo, body, detail, world, api,
    ):
        file_id = new_id()
        add_loft(world, file_id, clip_thumbnail_indexed=True)
        mark_corrected(world, file_id, redo=redo)

        response = await api.reindex(file_id, body)

        assert response.status_code == 409, response.text
        assert response.json() == {"detail": detail}
