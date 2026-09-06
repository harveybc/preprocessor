"""T0.2 invariant battery (work plan 43 §4) for the causal operator
protocol. Separate from the inherited suite; runs standalone."""
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import causal_operators as co  # noqa: E402

COLS = ["a", "b"]
RNG = np.random.default_rng(20260906)
TRAIN = RNG.normal(0.0, 1.0, (240, 2)).cumsum(axis=0)
SCORE = RNG.normal(0.0, 1.0, (120, 2)).cumsum(axis=0)


def _spec(kind="ewma", **params):
    defaults = {"identity": {}, "trailing_mean": {"window": 5},
                "trailing_median": {"window": 5},
                "ewma": {"alpha": 0.3},
                "local_level_kalman": {},
                "centered_mean_oracle": {"window": 5}}
    p = defaults[kind]
    p.update(params)
    oid = f"{kind}_t0"
    if kind == "centered_mean_oracle":
        oid = "NON_CAUSAL_ORACLE_ONLY_centered_mean"
    return {"schema": co.SCHEMA_VERSION, "operator_id": oid,
            "kind": kind, "version": "1", "params": p,
            "columns": list(COLS), "fit_role": "train",
            "lookback": p.get("window", 1),
            "availability_rule": "bar_close"}


ALL_KINDS = ("identity", "trailing_mean", "trailing_median",
             "ewma", "local_level_kalman")


def _fit(kind):
    return co.fit(_spec(kind), TRAIN, COLS, "train")


# 1. same prefix -> same bytes, batch vs incremental (multi-frag)
@pytest.mark.parametrize("kind", ALL_KINDS)
def test_inv1_batch_incremental_parity_multiple_fragmentations(kind):
    art = _fit(kind)
    batch = co.transform_batch(art, SCORE, COLS)
    for cuts in ([40, 80], [1, 2, 3, 60], [119], [7, 11, 90, 119]):
        state = co.init_state(art)
        outs = []
        prev = 0
        for c in cuts + [len(SCORE)]:
            out, state = co.transform_incremental(
                art, state, SCORE[prev:c], COLS)
            outs.append(out)
            prev = c
        frag = np.concatenate(outs)
        assert frag.tobytes() == batch.tobytes(), \
            f"{kind} parity broken at cuts {cuts}"


# 2. restart + artifact roundtrip -> same output
@pytest.mark.parametrize("kind", ALL_KINDS)
def test_inv2_restart_and_roundtrip(tmp_path, kind):
    art = _fit(kind)
    p = co.save_artifact(art, tmp_path, kind)
    art2 = co.load_artifact(p)
    assert art2["artifact_sha256"] == art["artifact_sha256"]
    b1 = co.transform_batch(art, SCORE, COLS)
    b2 = co.transform_batch(art2, SCORE, COLS)
    assert b1.tobytes() == b2.tobytes()


# 3. fit on a foreign role refuses
def test_inv3_foreign_fit_role_refuses():
    with pytest.raises(co.CausalOperatorError,
                       match="licenses only 'train'"):
        co.fit(_spec("ewma"), TRAIN, COLS, "validation")


# 4. missing/duplicate/reordered/extra columns refuse
def test_inv4_column_contract_refuses():
    art = _fit("ewma")
    for cols in (["a"], ["a", "a"], ["b", "a"], ["a", "b", "c"]):
        with pytest.raises(co.CausalOperatorError,
                           match="rejected"):
            co.transform_batch(art, SCORE[:, :len(cols)]
                               if len(cols) <= 2 else
                               np.hstack([SCORE, SCORE[:, :1]]),
                               cols)


# 5. NaN/inf/bool-as-number/string coercion refuse
def test_inv5_nonfinite_and_type_refusals():
    art = _fit("trailing_mean")
    bad = SCORE.copy()
    bad[5, 0] = np.nan
    with pytest.raises(co.CausalOperatorError, match="non-finite"):
        co.transform_batch(art, bad, COLS)
    bad[5, 0] = np.inf
    with pytest.raises(co.CausalOperatorError, match="non-finite"):
        co.transform_batch(art, bad, COLS)
    poisoned = _spec("trailing_mean")
    poisoned["params"]["window"] = True   # bool is never int
    with pytest.raises(co.CausalOperatorError, match="must be int"):
        co.validate_spec(poisoned)
    with pytest.raises(co.CausalOperatorError,
                       match="must be float"):
        co.validate_spec(_spec("ewma", alpha=1))
    with pytest.raises(co.CausalOperatorError,
                       match="must be float"):
        co.validate_spec(_spec("ewma", alpha="0.3"))


# 6. future availability refuses (oracle has no deployable path)
def test_inv6_centered_oracle_never_deployable():
    art = co.fit(_spec("centered_mean_oracle"), TRAIN, COLS, "train")
    state = co.init_state(art)
    with pytest.raises(co.CausalOperatorError,
                       match="NON_CAUSAL_ORACLE_ONLY"):
        co.transform_incremental(art, state, SCORE[:5], COLS)
    with pytest.raises(co.CausalOperatorError,
                       match="never look causal"):
        spec = _spec("centered_mean_oracle")
        spec["operator_id"] = "innocent_smoother"
        co.validate_spec(spec)


# 7. params/code/columns/order change the digest
def test_inv7_digest_sensitivity():
    base = co.spec_digest(_spec("ewma"))
    assert co.spec_digest(_spec("ewma", alpha=0.31)) != base
    other = _spec("ewma")
    other["columns"] = ["b", "a"]
    assert co.spec_digest(other) != base
    assert co.spec_digest(_spec("trailing_mean")) != base


# 8. cyclic or inapplicable composition never materializes
def test_inv8_dag_refusals():
    n = {"x": _spec("identity"), "y": _spec("ewma")}
    with pytest.raises(co.CausalOperatorError, match="cycle"):
        co.validate_dag(n, [("x", "y"), ("y", "x")], COLS)
    with pytest.raises(co.CausalOperatorError,
                       match="inapplicable"):
        co.validate_dag({"x": _spec("identity")}, [], ["z"])
    order = co.validate_dag(n, [("x", "y")], COLS)
    assert order == ["x", "y"]


# 9. the committed preprocessor pipeline is untouched
def test_inv9_committed_pipeline_untouched():
    import subprocess
    changed = subprocess.run(
        ["git", "diff", "--name-only", "HEAD"],
        cwd=str(Path(__file__).resolve().parents[1]),
        capture_output=True, text=True).stdout.split()
    touched_pipeline = [f for f in changed
                        if f.startswith("app/")
                        and "causal_operators" not in f]
    assert touched_pipeline == [], touched_pipeline


# spec hygiene: forbidden tokens
def test_spec_forbids_paths_and_callables():
    s = _spec("ewma")
    s["availability_rule"] = "/home/user/rule"
    with pytest.raises(co.CausalOperatorError, match="forbidden"):
        co.validate_spec(s)
    s = _spec("ewma")
    s["availability_rule"] = "__import__('os')"
    with pytest.raises(co.CausalOperatorError, match="forbidden"):
        co.validate_spec(s)


# identity is a mandatory exact no-op
def test_identity_is_exact_noop():
    art = _fit("identity")
    out = co.transform_batch(art, SCORE, COLS)
    assert out.tobytes() == np.asarray(SCORE, dtype=float).tobytes()


# artifact tamper refuses
def test_artifact_tamper_refuses(tmp_path):
    art = _fit("local_level_kalman")
    p = co.save_artifact(art, tmp_path, "k")
    doc = json.loads(p.read_text())
    doc["fitted"]["per_column"]["a"]["obs_var"] *= 2
    p.write_text(json.dumps(doc))
    with pytest.raises(co.CausalOperatorError,
                       match="does not re-derive"):
        co.load_artifact(p)
