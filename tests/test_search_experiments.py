import json
import sys
from pathlib import Path

from askchem import db

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))


def test_search_experiment_config_tracks_channel_flags(monkeypatch):
    monkeypatch.setenv("CHEMTREE_DISABLE_TREE_RECALL", "1")
    monkeypatch.setenv("CHEMTREE_DISABLE_FTS", "0")
    monkeypatch.setenv("CHEMTREE_MAX_QUERY_VARIANTS", "2")

    config = dict(db._search_experiment_config())

    assert config["CHEMTREE_DISABLE_TREE_RECALL"] is True
    assert config["CHEMTREE_DISABLE_FTS"] is False
    assert config["CHEMTREE_MAX_QUERY_VARIANTS"] == "2"


def test_eval_probe_supports_search_options(tmp_path):
    from eval_common import load_probes

    probes_path = tmp_path / "probes.jsonl"
    probes_path.write_text(json.dumps({
        "id": "p1",
        "q": "Suzuki coupling",
        "family": "view_filter",
        "view": "by_reaction_type",
        "claim_type": "reaction",
        "mode": "phrase",
        "sort": "date",
    }) + "\n")

    probe = load_probes(probes_path)[0]

    assert probe.view == "by_reaction_type"
    assert probe.claim_type == "reaction"
    assert probe.mode == "phrase"
    assert probe.sort == "date"
