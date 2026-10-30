"""Integration test for live IMO official problem retrieval."""

import urllib.request
import pytest
from mia.sources.imo import IMOSource


def _has_imo_connectivity() -> bool:
    try:
        req = urllib.request.Request("https://www.imo-official.org/", headers={"User-Agent": "MIA-Test"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status == 200
    except Exception:
        return False


@pytest.mark.integration
def test_live_imo_sync(tmp_path):
    """Retrieve actual problems from official IMO website."""
    if not _has_imo_connectivity():
        pytest.skip("No live network connectivity to https://www.imo-official.org")

    source = IMOSource(cache_dir=tmp_path)
    new_probs = source.sync(years=[2024], limit=3)
    if not new_probs and source.failures:
        pytest.skip(f"IMO live retrieval failed due to upstream network issue: {source.failures[-1].reason}")
    assert len(new_probs) > 0
    p1 = new_probs[0]
    assert p1.source == "imo"
    assert "imo-2024" in p1.problem_id
    assert len(p1.statement) > 20
