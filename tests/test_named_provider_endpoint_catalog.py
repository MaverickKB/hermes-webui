"""Named providers without allowlists discover their own endpoint catalog."""
import json
import sys
import threading
import types
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

import pytest

from api import config, profiles, routes


@pytest.fixture
def catalog(monkeypatch, tmp_path):
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            requests.append((self.path, self.headers.get("Authorization")))
            body = json.dumps({"data": [{"id": "auto"}, {"id": "org/local:fast"}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    endpoint = f"http://127.0.0.1:{server.server_port}/v1"
    fake_models = types.ModuleType("hermes_cli.models")
    fake_models.list_available_providers = lambda: []
    fake_models.provider_model_ids = lambda _pid: []
    fake_auth = types.ModuleType("hermes_cli.auth")
    fake_auth.get_auth_status = lambda _pid: {"key_source": "none"}
    monkeypatch.setitem(sys.modules, "hermes_cli.models", fake_models)
    monkeypatch.setitem(sys.modules, "hermes_cli.auth", fake_auth)
    monkeypatch.setattr(config, "_get_auth_store_path", lambda: tmp_path / "auth.json")
    monkeypatch.setattr(config, "_models_cache_path", tmp_path / "models.json")
    monkeypatch.setattr(config, "_LIVE_REBUILD_BUDGET_SECONDS", 0)
    monkeypatch.setattr(config, "cfg", {
        "model": {"provider": "openai-codex", "default": "gpt-5.5", "api_key": "other-provider-secret"},
        "providers": {"local_router": {"base_url": endpoint, "default_model": "retired-model"}},
    })
    config.invalidate_models_cache()
    routes._clear_live_models_cache()
    yield endpoint, requests
    routes._clear_live_models_cache()
    config.invalidate_models_cache()
    server.shutdown()
    server.server_close()
    worker.join()


def test_named_provider_catalog_and_selection(catalog):
    endpoint, requests = catalog
    result = config.get_available_models()
    groups = {group["provider_id"]: group for group in result["groups"]}
    assert "local-router" in groups
    models = groups["local-router"]["models"]
    assert {row["id"].removeprefix("@local-router:") for row in models} == {"auto", "org/local:fast"}
    for mid in ("auto", "org/local:fast"):
        assert config.resolve_model_provider(f"@local-router:{mid}") == (mid, "local_router", endpoint)
    assert requests == [("/v1/models", None)]


def test_named_provider_live_refresh(catalog, monkeypatch):
    _endpoint, requests = catalog
    monkeypatch.setattr(routes, "j", lambda _handler, payload: payload)
    result = routes._handle_live_models(None, urlparse("/api/models/live?provider=local-router"))
    assert {row["id"] for row in result["models"]} == {"auto", "org/local:fast"}
    assert requests == [("/v1/models", None)]


@pytest.mark.parametrize("key_field", ["key_env", "api_key_env", "api_key"])
def test_named_provider_uses_request_profile_key(catalog, monkeypatch, tmp_path, key_field):
    _endpoint, requests = catalog
    base = tmp_path / ".hermes"
    for name in ("work", "personal"):
        profile_home = base / "profiles" / name
        profile_home.mkdir(parents=True)
        (profile_home / ".env").write_text(f"LOCAL_ROUTER_KEY={name}-key\n")
    monkeypatch.setattr(profiles, "_DEFAULT_HERMES_HOME", base)
    monkeypatch.setenv("LOCAL_ROUTER_KEY", "process-secret")
    config.cfg["providers"]["local_router"][key_field] = (
        "${LOCAL_ROUTER_KEY}" if key_field == "api_key" else "LOCAL_ROUTER_KEY"
    )
    try:
        for name in ("work", "personal"):
            profiles.set_request_profile(name)
            with profiles.profile_env_for_active_request_readonly("test"):
                assert config._read_live_provider_model_ids("local-router") == ["auto", "org/local:fast"]
    finally:
        profiles.clear_request_profile()
    assert requests == [
        ("/v1/models", "Bearer work-key"),
        ("/v1/models", "Bearer personal-key"),
    ]


def test_named_provider_allowlist_and_offline_default(catalog, monkeypatch):
    _endpoint, requests = catalog
    entry = config.cfg["providers"]["local_router"]
    entry["models"] = ["curated-a", "curated-b"]
    assert config._read_live_provider_model_ids("local-router") == ["curated-a", "curated-b"]
    assert requests == []
    del entry["models"]
    monkeypatch.setattr("urllib.request.urlopen", lambda *_a, **_k: (_ for _ in ()).throw(OSError("offline")))
    assert config._read_live_provider_model_ids("local-router") == ["retired-model"]



@pytest.mark.parametrize("discover", [False, "false", "no", "0"])
def test_named_provider_discovery_opt_out(catalog, discover):
    _endpoint, requests = catalog
    config.cfg["providers"]["local_router"]["discover_models"] = discover
    assert config._read_live_provider_model_ids("local-router") == ["retired-model"]
    assert requests == []


@pytest.mark.parametrize("legacy_cache", [False, True])
def test_named_provider_survives_slow_catalog_rebuild(catalog, monkeypatch, legacy_cache):
    _endpoint, requests = catalog
    monkeypatch.setattr(config, "_LIVE_REBUILD_BUDGET_SECONDS", 0.01)
    if legacy_cache:
        config._save_models_cache_to_disk({
            "active_provider": "openai-codex", "default_model": "gpt-5.5",
            "configured_model_badges": {},
            "groups": [{"provider": "OpenAI Codex", "provider_id": "openai-codex",
                        "models": [{"id": "gpt-5.5", "label": "GPT"}]}],
            "aliases": {},
        })
        cache_path = config._get_models_cache_path()
        saved = json.loads(cache_path.read_text())
        # The released catalog schema omitted named endpoints on this path.
        saved["_schema_version"] = 3
        cache_path.write_text(json.dumps(saved))

    release_probe = threading.Event()
    def slow_provider_discovery():
        release_probe.wait()
        return []
    monkeypatch.setattr(sys.modules["hermes_cli.models"], "list_available_providers", slow_provider_discovery)
    try:
        result = config.get_available_models()
        groups = {group["provider_id"]: group for group in result["groups"]}
        assert [row["id"] for row in groups["local-router"]["models"]] == ["@local-router:retired-model"]
        assert requests == []
    finally:
        release_probe.set()
        with config._cache_build_cv:
            assert config._cache_build_cv.wait_for(lambda: not config._cache_build_in_progress, timeout=5)
