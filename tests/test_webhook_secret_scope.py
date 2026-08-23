"""The webhook secret gates webhooks, and must not gate the queue routes.

Both halves matter and neither is obvious from reading one file:

- **Webhooks keep the gate.** It is opt-in (a no-op when
  `SEARCH_WEBHOOK_SECRET` is unset) but core supports it end to end: a
  listener declaring `secret_env` makes core send `X-Webhook-Secret`. A
  hand-wired deployment relies on it, and `/webhook/files-purged` can
  permanently delete this addon's index data, so quietly dropping the gate
  is not a cleanup.

- **Queue routes must not have it.** They are called from the browser
  through core's addon proxy, which never attaches that header. Guarding
  them meant setting the variable returned 403 for the admin queue
  controls. Their authorization is the proxy's `pre_check: admin`.
"""

import json
from pathlib import Path

import pytest
from fastapi import Depends

from app.dependencies import verify_webhook_secret
from app.routers import queue as queue_router
from app.routers import webhooks as webhooks_router

MANIFEST = Path(__file__).resolve().parents[1] / "manifest.json"


def _dependency_names(route) -> set[str]:
    return {d.call.__name__ for d in route.dependant.dependencies if d.call}


def _routes(router):
    return [r for r in router.router.routes if getattr(r, "dependant", None)]


class TestWebhooksAreGated:
    def test_every_webhook_route_verifies_the_secret(self):
        routes = _routes(webhooks_router)
        assert routes, "no webhook routes found — has the module moved?"
        for route in routes:
            assert "verify_webhook_secret" in _dependency_names(route), (
                f"{route.path} lost its secret check; a hand-wired "
                "deployment would silently become unauthenticated"
            )


class TestQueueIsNotGated:
    def test_no_queue_route_verifies_the_secret(self):
        for route in _routes(queue_router):
            assert "verify_webhook_secret" not in _dependency_names(route), (
                f"{route.path} is called from the browser via core's proxy, "
                "which never sends X-Webhook-Secret — this gate turns into a "
                "403 as soon as the secret is configured"
            )

    def test_queue_routes_are_admin_gated_in_the_manifest(self):
        """The proxy's pre_check is what actually authorizes them."""
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        by_path = {
            r["path"]: r for r in manifest.get("proxy", {}).get("routes", [])
        }
        queue_paths = [p for p in by_path if p.startswith("/queue/")]
        assert queue_paths, "no /queue/* routes declared in the manifest"
        for path in queue_paths:
            pre_check = by_path[path].get("pre_check") or {}
            assert pre_check.get("type") == "admin", (
                f"{path} has no admin pre_check; removing the addon-side "
                "gate would leave it unauthorized"
            )


class TestManifestDeclaresTheSecret:
    def test_every_listener_declares_secret_env(self):
        """Core only sends the header when the listener asks for it.

        Without this the gate cannot be switched on from the supported
        path: `configure.py` would set the variable in the container while
        core kept sending unauthenticated requests, and every webhook would
        403 with indexing silently stopped.
        """
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        hooks = manifest.get("event_hooks") or []
        assert hooks, "no event_hooks declared"
        for hook in hooks:
            assert hook.get("secret_env") == "SEARCH_WEBHOOK_SECRET", (
                f"{hook.get('event')} -> {hook.get('url')} does not declare "
                "secret_env, so core will never send X-Webhook-Secret to it"
            )


class TestGateSemantics:
    @pytest.mark.asyncio
    async def test_unset_secret_accepts_anything(self, monkeypatch):
        monkeypatch.setattr(
            "app.dependencies._WEBHOOK_SECRET", "", raising=False
        )
        assert await verify_webhook_secret(x_webhook_secret="") is None
        assert await verify_webhook_secret(x_webhook_secret="whatever") is None

    @pytest.mark.asyncio
    async def test_configured_secret_rejects_a_mismatch(self, monkeypatch):
        from fastapi import HTTPException

        monkeypatch.setattr(
            "app.dependencies._WEBHOOK_SECRET", "s3cret", raising=False
        )
        assert await verify_webhook_secret(x_webhook_secret="s3cret") is None
        with pytest.raises(HTTPException) as exc:
            await verify_webhook_secret(x_webhook_secret="wrong")
        assert exc.value.status_code == 403
