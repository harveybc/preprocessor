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


def _tc(n, start=0.0):
    return co.make_bar_close_contract(n, start)


def _fit(kind):
    return co.fit(_spec(kind), TRAIN, COLS, "train")


# 1. same prefix -> same bytes, batch vs incremental (multi-frag)
@pytest.mark.parametrize("kind", ALL_KINDS)
def test_inv1_batch_incremental_parity_multiple_fragmentations(kind):
    art = _fit(kind)
    batch = co.transform_batch(art, SCORE, COLS, _tc(len(SCORE)))
    for cuts in ([40, 80], [1, 2, 3, 60], [119], [7, 11, 90, 119]):
        state = co.init_state(art)
        outs = []
        prev = 0
        for c in cuts + [len(SCORE)]:
            out, state = co.transform_incremental(
                art, state, SCORE[prev:c], COLS,
                _tc(c - prev, float(prev)))
            outs.append(out)
            prev = c
        frag = np.concatenate(outs)
        assert frag.tobytes() == batch.tobytes(), \
            f"{kind} parity broken at cuts {cuts}"


# 2. restart + artifact roundtrip -> same output
@pytest.mark.parametrize("kind", ALL_KINDS)
def test_inv2_restart_and_roundtrip(tmp_path, kind):
    art = _fit(kind)
    p = co.save_artifact(art, tmp_path)
    art2 = co.load_artifact(p)
    assert art2["artifact_sha256"] == art["artifact_sha256"]
    b1 = co.transform_batch(art, SCORE, COLS, _tc(len(SCORE)))
    b2 = co.transform_batch(art2, SCORE, COLS, _tc(len(SCORE)))
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
        mat = (SCORE[:, :len(cols)] if len(cols) <= 2
               else np.hstack([SCORE, SCORE[:, :1]]))
        with pytest.raises(co.CausalOperatorError,
                           match="rejected"):
            co.transform_batch(art, mat, cols, _tc(len(mat)))


# 5. NaN/inf/bool-as-number/string coercion refuse
def test_inv5_nonfinite_and_type_refusals():
    art = _fit("trailing_mean")
    bad = SCORE.copy()
    bad[5, 0] = np.nan
    with pytest.raises(co.CausalOperatorError, match="non-finite"):
        co.transform_batch(art, bad, COLS, _tc(len(bad)))
    bad[5, 0] = np.inf
    with pytest.raises(co.CausalOperatorError, match="non-finite"):
        co.transform_batch(art, bad, COLS, _tc(len(bad)))
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
        co.transform_incremental(art, state, SCORE[:5], COLS, _tc(5))
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


# spec hygiene: closed availability vocabulary (C1)
def test_spec_forbids_paths_and_callables():
    for rule in ("/home/user/rule", "__import__('os')",
                 "tomorrow_after_decision"):
        s = _spec("ewma")
        s["availability_rule"] = rule
        with pytest.raises(co.CausalOperatorError,
                           match="free-form|forbidden"):
            co.validate_spec(s)


# identity is a mandatory exact no-op
def test_identity_is_exact_noop():
    art = _fit("identity")
    out = co.transform_batch(art, SCORE, COLS, _tc(len(SCORE)))
    assert out.tobytes() == np.asarray(SCORE, dtype=float).tobytes()


# artifact tamper refuses
def test_artifact_tamper_refuses(tmp_path):
    art = _fit("local_level_kalman")
    p = co.save_artifact(art, tmp_path)
    doc = json.loads(p.read_text())
    doc["fitted"]["per_column"]["a"]["obs_var"] *= 2
    mutated = tmp_path / "mutated.json"
    mutated.write_text(json.dumps(doc))
    with pytest.raises(co.CausalOperatorError,
                       match="does not re-derive"):
        co.load_artifact(mutated)


# ---------------- C1-C4 correction battery ----------------
def test_c1_bool_and_string_matrices_refuse():
    art = _fit("ewma")
    with pytest.raises(co.CausalOperatorError, match="rejected"):
        co.transform_batch(art, np.array([[True], [False]]),
                           COLS[:1], _tc(2))
    with pytest.raises(co.CausalOperatorError, match="rejected"):
        co.transform_batch(art, np.array([["1.25"], ["2.5"]],
                                         dtype=object),
                           COLS[:1], _tc(2))
    with pytest.raises(co.CausalOperatorError, match="rejected"):
        co.fit(_spec("ewma"), np.array([[True, False]] * 30),
               COLS, "train")


def test_c1_future_or_unfinalized_refuses():
    art = _fit("ewma")
    n = 5
    tc = co.make_bar_close_contract(n)
    tc["as_of"] = 2.0                # rows 3,4 are in the future
    with pytest.raises(co.CausalOperatorError, match="future"):
        co.transform_batch(art, SCORE[:n], COLS, tc)
    tc2 = co.make_bar_close_contract(n)
    tc2["finalized_ts"][2] = tc2["as_of"] + 1.0   # unfinalized
    with pytest.raises(co.CausalOperatorError, match="future"):
        co.transform_batch(art, SCORE[:n], COLS, tc2)
    tc3 = co.make_bar_close_contract(n)
    tc3["finalized_ts"][2] = tc3["observation_ts"][2] - 1.0
    with pytest.raises(co.CausalOperatorError, match="impossible"):
        co.transform_batch(art, SCORE[:n], COLS, tc3)
    with pytest.raises(co.CausalOperatorError, match="timestamp"):
        co.transform_batch(art, SCORE[:n], COLS, None)


def test_c2_foreign_state_refuses():
    art3 = _fit("ewma")
    art9 = co.fit(_spec("ewma", alpha=0.9), TRAIN, COLS, "train")
    state = co.init_state(art3)
    _, state = co.transform_incremental(art3, state, SCORE[:10],
                                        COLS, _tc(10))
    with pytest.raises(co.CausalOperatorError,
                       match="foreign state"):
        co.transform_incremental(art9, state, SCORE[10:12], COLS,
                                 _tc(2, 10.0))


@pytest.mark.parametrize("kind", ALL_KINDS)
@pytest.mark.parametrize("cut", (7, 40, 99))
def test_c2_durable_restart_parity(tmp_path, kind, cut):
    """C2: uninterrupted A+B == A -> durable state -> fresh load ->
    B, byte for byte, at multiple cut points."""
    art = _fit(kind)
    full = co.transform_batch(art, SCORE, COLS, _tc(len(SCORE)))
    state = co.init_state(art)
    out_a, state = co.transform_incremental(
        art, state, SCORE[:cut], COLS, _tc(cut))
    sp = co.save_state(state, art, tmp_path, f"s{cut}")
    resumed = co.load_state(sp, art)          # fresh-process load
    out_b, _ = co.transform_incremental(
        art, resumed, SCORE[cut:], COLS,
        _tc(len(SCORE) - cut, float(cut)))
    assert np.concatenate([out_a, out_b]).tobytes() == \
        full.tobytes()


def test_c3_incompatible_edge_refuses():
    child = _spec("ewma")
    child["columns"] = ["z", "w"]
    with pytest.raises(co.CausalOperatorError,
                       match="schema-incompatible"):
        co.validate_dag({"root": _spec("identity"),
                         "kid": child},
                        [("root", "kid")], COLS)
    with pytest.raises(co.CausalOperatorError,
                       match="duplicate edge"):
        co.validate_dag({"a": _spec("identity"),
                         "b": _spec("ewma")},
                        [("a", "b"), ("a", "b")], COLS)
    with pytest.raises(co.CausalOperatorError,
                       match="disconnected"):
        co.validate_dag({"a": _spec("identity"),
                         "b": _spec("ewma")}, [], COLS)


def test_c3_graph_artifact_digest_binds(tmp_path):
    nodes = {"a": _spec("identity"), "b": _spec("ewma")}
    arts = {n: co.fit(s, TRAIN, COLS, "train")
            for n, s in nodes.items()}
    g = co.build_graph_artifact(nodes, [("a", "b")], COLS, arts)
    assert g["status"] == "VALIDATED_NOT_EXECUTABLE"
    g2 = dict(g)
    g2["topological_order"] = list(reversed(g["topological_order"]))
    import hashlib as h
    assert h.sha256(co._canonical(
        {k: g2[k] for k in g2 if k != "graph_sha256"})
    ).hexdigest() != g["graph_sha256"]


def test_c4_write_once_artifacts(tmp_path):
    art3 = _fit("ewma")
    art9 = co.fit(_spec("ewma", alpha=0.9), TRAIN, COLS, "train")
    p1 = co.save_artifact(art3, tmp_path)
    p2 = co.save_artifact(art3, tmp_path)    # idempotent identical
    assert p1 == p2
    p3 = co.save_artifact(art9, tmp_path)
    assert p3 != p1                          # content-addressed
    # a DIFFERENT artifact can never occupy an existing identity:
    forged = dict(art9)
    forged["artifact_sha256"] = art3["artifact_sha256"]
    with pytest.raises(co.CausalOperatorError):
        co.save_artifact(forged, tmp_path)


def test_c4_duplicate_key_and_nonfinite_load_refuse(tmp_path):
    art = _fit("ewma")
    p = co.save_artifact(art, tmp_path)
    evil = tmp_path / "evil.json"
    evil.write_text('{"schema": "x", "schema": "y"}')
    with pytest.raises(co.CausalOperatorError,
                       match="duplicate JSON key"):
        co.load_artifact(evil)
    evil.write_text('{"schema": NaN}')
    with pytest.raises(co.CausalOperatorError,
                       match="non-finite"):
        co.load_artifact(evil)
