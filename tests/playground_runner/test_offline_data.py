"""The kernel startup hook that serves cached datasets for their URLs.

Runs in this process, so every test restores pandas and urllib afterwards.
"""

import json
import urllib.request

import pandas as pd
import pytest
from playground_runner import offline_data
from playground_runner.offline_data import OfflineDatasetError

URL = "https://raw.githubusercontent.com/example/repo/main/transactions.csv"


@pytest.fixture
def restore_patches():
    saved = {name: getattr(pd, name) for name in offline_data.PANDAS_READERS}
    saved_urlopen = urllib.request.urlopen
    yield
    for name, fn in saved.items():
        setattr(pd, name, fn)
    urllib.request.urlopen = saved_urlopen


@pytest.fixture
def data_dir(tmp_path):
    root = tmp_path / "data" / "pg-1"
    (root / "files").mkdir(parents=True)
    (root / "files" / "transactions.csv").write_text("account,amount\nA,10\nB,25\n")
    (root / "files" / "rows.json").write_text(json.dumps([{"a": 1}, {"a": 2}]))
    (tmp_path / "secret.csv").write_text("x\n1\n")
    (root / "manifest.json").write_text(
        json.dumps(
            {
                URL: "files/transactions.csv",
                "https://example.org/rows.json": "files/rows.json",
                "https://example.org/escape.csv": "../../secret.csv",
                "https://example.org/missing.csv": "files/not-there.csv",
            }
        )
    )
    return tmp_path / "data"


def test_mapped_url_reads_the_cached_file(restore_patches, data_dir):
    offline_data.install(str(data_dir), "pg-1")
    df = pd.read_csv(URL)
    assert list(df.columns) == ["account", "amount"]
    assert df["amount"].sum() == 35
    # Keyword form, and a fragment does not defeat the lookup.
    assert len(pd.read_csv(filepath_or_buffer=URL + "#L1")) == 2
    assert pd.read_json("https://example.org/rows.json")["a"].tolist() == [1, 2]


def test_unmapped_url_fails_clearly(restore_patches, data_dir):
    offline_data.install(str(data_dir), "pg-1")
    with pytest.raises(OfflineDatasetError) as info:
        pd.read_csv("https://example.org/not-cached.csv")
    message = str(info.value)
    assert "runs offline" in message
    assert "https://example.org/not-cached.csv" in message
    assert "pg-1" in message
    # Still a URLError, so code that handles download failures catches it.
    assert isinstance(info.value, urllib.error.URLError)


def test_manifest_cannot_point_outside_its_playground(restore_patches, data_dir):
    offline_data.install(str(data_dir), "pg-1")
    with pytest.raises(OfflineDatasetError):
        pd.read_csv("https://example.org/escape.csv")


def test_mapped_entry_whose_file_is_missing_fails_clearly(restore_patches, data_dir):
    offline_data.install(str(data_dir), "pg-1")
    with pytest.raises(OfflineDatasetError, match="not in the dataset cache"):
        pd.read_csv("https://example.org/missing.csv")


def test_local_paths_and_buffers_pass_through(restore_patches, data_dir, tmp_path):
    offline_data.install(str(data_dir), "pg-1")
    local = tmp_path / "local.csv"
    local.write_text("a\n1\n2\n")
    assert pd.read_csv(local)["a"].sum() == 3
    assert pd.read_csv(str(local))["a"].sum() == 3
    import io

    assert pd.read_csv(io.StringIO("a\n5\n"))["a"].sum() == 5


def test_urlopen_and_urlretrieve_serve_the_cache(restore_patches, data_dir, tmp_path):
    offline_data.install(str(data_dir), "pg-1")
    with urllib.request.urlopen(URL) as response:
        assert response.read().startswith(b"account,amount")
        assert response.status == 200
    with urllib.request.urlopen(urllib.request.Request(URL)) as response:
        assert b"B,25" in response.read()
    target = tmp_path / "copy.csv"
    urllib.request.urlretrieve(URL, target)
    assert target.read_text().startswith("account,amount")
    with pytest.raises(OfflineDatasetError, match="runs offline"):
        urllib.request.urlopen("https://example.org/elsewhere")


def test_install_twice_does_not_stack_wrappers(restore_patches, data_dir):
    original = pd.read_csv
    offline_data.install(str(data_dir), "pg-1")
    offline_data.install(str(data_dir), "pg-1")
    assert pd.read_csv.__playground_offline__ is original


def test_a_playground_without_a_cache_refuses_every_url(restore_patches, tmp_path):
    offline_data.install(str(tmp_path), "never-imported")
    with pytest.raises(OfflineDatasetError, match="never-imported"):
        pd.read_csv(URL)
