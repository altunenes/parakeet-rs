"""The optional example must verify the complete release before loading it."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
from huggingface_hub.errors import LocalEntryNotFoundError

MISSING = "missing"
SPEC = importlib.util.spec_from_file_location(
    "orukeet_example", Path(__file__).parents[1] / "scripts/download_orukeet.py"
)
assert SPEC is not None
assert SPEC.loader is not None
example = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(example)


@pytest.fixture
def release(tmp_path, monkeypatch):
    entries = {}
    for name in example.FILES:
        content = name.encode()
        (tmp_path / name).write_bytes(content)
        entries[name] = {
            "bytes": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
        }
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"files": entries}))
    monkeypatch.setattr(example, "MANIFEST_SHA256", example.sha256(manifest))
    calls = []

    def fetch(**kwargs):
        calls.append(kwargs)
        assert kwargs["repo_id"] == example.REPO_ID
        assert kwargs["revision"] == example.REVISION
        assert kwargs["filename"].startswith(example.SUBFOLDER + "/")
        return str(tmp_path / Path(kwargs["filename"]).name)

    monkeypatch.setattr(example, "hf_hub_download", fetch)
    return tmp_path, calls


def test_cached_release_checks_all_files(release):
    path, calls = release
    assert example.download_model(offline=True) == path
    assert len(calls) == len(example.FILES) + 1
    assert all(call["local_files_only"] for call in calls)


@pytest.mark.parametrize(
    "name", ["manifest.json", "encoder-model.int8.onnx", "config.json"]
)
def test_rejects_corrupt_release(release, name):
    path, _ = release
    (path / name).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum mismatch"):
        example.download_model()


def test_offline_never_retries_network(monkeypatch):
    calls = []

    def missing(**kwargs):
        calls.append(kwargs)
        raise LocalEntryNotFoundError(MISSING)

    monkeypatch.setattr(example, "hf_hub_download", missing)
    with pytest.raises(LocalEntryNotFoundError):
        example.download_model(offline=True)
    assert len(calls) == 1
    assert calls[0]["local_files_only"]


def test_uncached_release_downloads_then_verifies(release, monkeypatch):
    path, _ = release
    calls = []

    def fetch(**kwargs):
        calls.append(kwargs)
        if kwargs.get("local_files_only"):
            raise LocalEntryNotFoundError(MISSING)
        return str(path / Path(kwargs["filename"]).name)

    monkeypatch.setattr(example, "hf_hub_download", fetch)
    assert example.download_model() == path
    assert len(calls) == 2 * (len(example.FILES) + 1)
