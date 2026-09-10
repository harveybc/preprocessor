"""P4: the two historical selectors have no split boundary.

Neither selector receives a split, so both fit their statistics on
whatever frame they are handed. The behaviour is preserved for
archival replay and LABELLED, so a reader can never mistake either
result for boundary-clean evidence. These tests pin the labels.
"""
from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PRE = REPO / "app/plugins/plugin_feature_selector_pre.py"
POST = REPO / "app/plugins/plugin_feature_selector_post.py"


def _flat(path: Path) -> str:
    """Source with runs of whitespace collapsed, so a wrapped
    sentence still matches the sentence it is."""
    return re.sub(r"\s+", " ", path.read_text())


def _const(path: Path, name: str) -> str:
    m = re.search(rf'{name} = "([A-Z_]+)"', path.read_text())
    assert m, f"{name} not declared in {path.name}"
    return m.group(1)


def test_pre_selector_is_labelled_non_authoritative():
    assert _const(PRE, "SELECTION_AUTHORITY") == \
        "LEGACY_NON_AUTHORITATIVE"
    text = _flat(PRE)
    assert "no split boundary" in text
    assert "fitted outside training" in text


def test_post_selector_is_labelled_non_authoritative():
    assert _const(POST, "SELECTION_AUTHORITY") == \
        "LEGACY_NON_AUTHORITATIVE"
    text = _flat(POST)
    assert "no split boundary" in text
    assert "fitted outside training" in text


def test_neither_selector_takes_a_split_argument():
    """The defect is structural: there is no parameter through
    which a caller could restrict the fit to training."""
    for path in (PRE, POST):
        sig = re.search(r"def process\(self[^)]*\)",
                        _flat(path)).group(0)
        for boundary_arg in ("split", "train_index", "fit_scope",
                             "train_only"):
            assert boundary_arg not in sig, (
                f"{path.name} grew a boundary argument — update "
                "this test and the label together")


def test_documented_but_absent_method_now_refuses():
    text = _flat(POST)
    assert 'UNIMPLEMENTED_DOCUMENTED_METHODS = ("cross_val",)' \
        in text
    assert "documented but not implemented" in text


def test_labels_are_visible_at_runtime_not_only_in_comments():
    for path in (PRE, POST):
        text = path.read_text()
        assert "print(f\"[WARNING]" in text, (
            f"{path.name} does not announce its label when it "
            "runs")
