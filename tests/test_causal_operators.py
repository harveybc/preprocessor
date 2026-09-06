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
            "lookback": co.derived_lookback(kind, p),
            "availability_rule": "bar_close"}


ALL_KINDS = ("identity", "trailing_mean", "trailing_median",
             "ewma", "local_level_kalman")


STREAM = "synthetic_bar_close"


def _tc(n, start=0.0):
    return co.make_bar_close_contract(n, start)


def _train_contract(matrix=None):
    m = TRAIN if matrix is None else matrix
    return co.make_train_contract(m, len(m))


def _fit(kind):
    return co.fit(_spec(kind), TRAIN, COLS, "train",
                  _train_contract())


def _init(art):
    return co.init_state(art, STREAM, 1.0)


def _load(snap_path, art):
    doc = json.loads(Path(snap_path).read_text())
    return co.load_state(snap_path, art, doc["snapshot_sha256"])


# 1. same prefix -> same bytes, batch vs incremental (multi-frag)
@pytest.mark.parametrize("kind", ALL_KINDS)
def test_inv1_batch_incremental_parity_multiple_fragmentations(kind):
    art = _fit(kind)
    batch = co.transform_batch(art, SCORE, COLS, _tc(len(SCORE)))
    for cuts in ([40, 80], [1, 2, 3, 60], [119], [7, 11, 90, 119]):
        state = _init(art)
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
        co.fit(_spec("ewma"), TRAIN, COLS, "validation",
               _train_contract())


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
    art = co.fit(_spec("centered_mean_oracle"), TRAIN, COLS,
                 "train", _train_contract())
    state = _init(art)
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
               COLS, "train", _train_contract(TRAIN[:30]))


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
    art9 = co.fit(_spec("ewma", alpha=0.9), TRAIN, COLS, "train",
                 _train_contract())
    state = _init(art3)
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
    state = _init(art)
    out_a, state = co.transform_incremental(
        art, state, SCORE[:cut], COLS, _tc(cut))
    sp = co.save_state(state, art, tmp_path)
    resumed = _load(sp, art)                  # fresh-process load
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
    arts = {n: co.fit(s, TRAIN, COLS, "train",
                      _train_contract())
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
    art9 = co.fit(_spec("ewma", alpha=0.9), TRAIN, COLS, "train",
                 _train_contract())
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


# ---------------- C13-C15 correction battery ----------------
def test_c13_pre_bypass_state_dies():
    """§4.1/§4.3: the PRE forgery (foreign version, fabricated
    rows_seen, arbitrary payload) refuses at every mutated field."""
    art = _fit("ewma")
    base = _init(art)
    _, base = co.transform_incremental(art, base, SCORE[:3], COLS,
                                       _tc(3))
    # standalone-detectable forgeries refuse at _check_state
    for mut, match in (
            ({"version": "foreign"}, "VERSION"),
            ({"stream_id": "other_stream"}, "stream"),
            ({"schema": "causal_operator_state.v2"}, "schema")):
        forged = copy.deepcopy(base)
        forged.update(mut)
        with pytest.raises(co.CausalOperatorError, match=match):
            co.transform_incremental(art, forged, SCORE[3:4],
                                     COLS, _tc(1, 3.0))
    smuggled = copy.deepcopy(base)
    smuggled["attacker"] = True
    with pytest.raises(co.CausalOperatorError,
                       match="exact v3 schema"):
        co.transform_incremental(art, smuggled, SCORE[3:4], COLS,
                                 _tc(1, 3.0))
    # a fresh-looking rows_seen=0 with a lived-in payload dies on
    # the coherence checks
    deep = copy.deepcopy(base)
    deep["rows_seen"] = 0
    with pytest.raises(co.CausalOperatorError,
                       match="incoherent|genesis"):
        co.transform_incremental(art, deep, SCORE[3:4], COLS,
                                 _tc(1, 3.0))
    # fresh state whose payload claims history refuses
    lied = _init(art)
    lied["payload"]["ewma"] = [-1986.0, -1986.0]
    with pytest.raises(co.CausalOperatorError,
                       match="incoherent"):
        co.transform_incremental(art, lied, SCORE[:1], COLS,
                                 _tc(1))


def test_c13_snapshot_authority_kills_value_forgeries(tmp_path):
    """§4.3: rows_seen/prefix/payload VALUE forgeries are killed by
    the snapshot authority chain — tampered bytes break the digest;
    re-signed bytes differ from the digest the run contract
    authorizes."""
    art = _fit("ewma")
    state = _init(art)
    _, state = co.transform_incremental(art, state, SCORE[:3],
                                        COLS, _tc(3))
    sp = co.save_state(state, art, tmp_path)
    authorized = json.loads(sp.read_text())["snapshot_sha256"]
    # tamper without re-signing -> digest does not re-derive
    doc = json.loads(sp.read_text())
    doc["state"]["rows_seen"] = 1000
    doc["state"]["prefix_sha256"] = "0" * 64
    evil = tmp_path / "evil.state.json"
    evil.write_text(json.dumps(doc, sort_keys=True, indent=1))
    with pytest.raises(co.CausalOperatorError,
                       match="does not re-derive"):
        co.load_state(evil, art, authorized)
    # re-sign coherently -> the digest is VALID but FOREIGN to the
    # authorizing run contract
    body = {"schema": doc["schema"], "state": doc["state"],
            "parent_snapshot_sha256":
                doc["parent_snapshot_sha256"]}
    import hashlib as _h
    doc["snapshot_sha256"] = _h.sha256(
        co._canonical(body)).hexdigest()
    evil.write_text(json.dumps(doc, sort_keys=True, indent=1))
    with pytest.raises(co.CausalOperatorError, match="authorizes"):
        co.load_state(evil, art, authorized)


def test_c13_replay_rewind_gap_fork_foreign():
    art = _fit("ewma")
    state = _init(art)
    _, state = co.transform_incremental(art, state, SCORE[:10],
                                        COLS, _tc(10))
    # replay/rewind
    with pytest.raises(co.CausalOperatorError, match="replay"):
        co.transform_incremental(art, copy.deepcopy(state),
                                 SCORE[5:7], COLS, _tc(2, 5.0))
    # gap refuses without the explicit policy, passes with it
    with pytest.raises(co.CausalOperatorError, match="gap"):
        co.transform_incremental(art, copy.deepcopy(state),
                                 SCORE[15:17], COLS, _tc(2, 15.0))
    out, _ = co.transform_incremental(
        art, copy.deepcopy(state), SCORE[15:17], COLS,
        _tc(2, 15.0), gap_policy="accept_declared_gap")
    assert out.shape == (2, 2)
    # irregular step INSIDE a chunk refuses
    tcx = _tc(2, 10.0)
    tcx["observation_ts"][1] = 13.5
    tcx["finalized_ts"][1] = 13.5
    tcx["as_of"] = 15.0
    with pytest.raises(co.CausalOperatorError, match="step"):
        co.transform_incremental(art, copy.deepcopy(state),
                                 SCORE[10:12], COLS, tcx)
    # foreign stream in the chunk contract
    tcf = _tc(2, 10.0)
    tcf["stream_id"] = "someone_elses_feed"
    with pytest.raises(co.CausalOperatorError, match="stream"):
        co.transform_incremental(art, copy.deepcopy(state),
                                 SCORE[10:12], COLS, tcf)
    # foreign stream at init refuses against the artifact binding
    with pytest.raises(co.CausalOperatorError, match="foreign"):
        co.init_state(art, "someone_elses_feed", 1.0)


def test_c13_fork_is_visible_in_the_snapshot_chain(tmp_path):
    """Two continuations from one snapshot are DETECTABLE: both
    children carry the same parent digest."""
    art = _fit("ewma")
    state = _init(art)
    _, state = co.transform_incremental(art, state, SCORE[:10],
                                        COLS, _tc(10))
    sp = co.save_state(state, art, tmp_path)
    parent = json.loads(sp.read_text())["snapshot_sha256"]
    kids = []
    for sl in (SCORE[10:20], SCORE[10:20] + 1.0):
        st = _load(sp, art)
        _, st = co.transform_incremental(art, st, sl, COLS,
                                         _tc(10, 10.0))
        kp = co.save_state(st, art, tmp_path,
                           parent_snapshot_sha256=parent)
        kids.append(json.loads(kp.read_text()))
    assert kids[0]["snapshot_sha256"] != kids[1]["snapshot_sha256"]
    assert kids[0]["parent_snapshot_sha256"] == \
        kids[1]["parent_snapshot_sha256"] == parent


def test_c13_load_requires_run_contract_authority(tmp_path):
    art = _fit("ewma")
    state = _init(art)
    sp = co.save_state(state, art, tmp_path)
    real = json.loads(sp.read_text())["snapshot_sha256"]
    with pytest.raises(co.CausalOperatorError, match="run contract"):
        co.load_state(sp, art, None)
    with pytest.raises(co.CausalOperatorError, match="authorizes"):
        co.load_state(sp, art, "f" * 64)
    assert co.load_state(sp, art, real)["rows_seen"] == 0


def test_c15_fit_contract_rejects_bad_observations():
    spec = _spec("ewma")
    good = _train_contract()
    # shuffled
    tc = copy.deepcopy(good)
    o = tc["time_contract"]["observation_ts"]
    f = tc["time_contract"]["finalized_ts"]
    o[0], o[1] = o[1], o[0]
    f[0], f[1] = f[1], f[0]
    with pytest.raises(co.CausalOperatorError,
                       match="STRICTLY increasing"):
        co.fit(spec, TRAIN, COLS, "train", tc)
    # duplicate
    tc = copy.deepcopy(good)
    tc["time_contract"]["observation_ts"][1] = \
        tc["time_contract"]["observation_ts"][0]
    tc["time_contract"]["finalized_ts"][1] = \
        tc["time_contract"]["finalized_ts"][0]
    with pytest.raises(co.CausalOperatorError,
                       match="STRICTLY increasing"):
        co.fit(spec, TRAIN, COLS, "train", tc)
    # future rows at the fit as_of
    tc = copy.deepcopy(good)
    tc["time_contract"]["as_of"] = 5.0
    with pytest.raises(co.CausalOperatorError, match="future"):
        co.fit(spec, TRAIN, COLS, "train", tc)
    # unfinalized
    tc = copy.deepcopy(good)
    tc["time_contract"]["finalized_ts"][3] = \
        tc["time_contract"]["as_of"] + 1.0
    with pytest.raises(co.CausalOperatorError, match="future"):
        co.fit(spec, TRAIN, COLS, "train", tc)
    # outside the declared interval
    tc = copy.deepcopy(good)
    tc["interval"] = [100.0, 400.0]
    with pytest.raises(co.CausalOperatorError, match="interval"):
        co.fit(spec, TRAIN, COLS, "train", tc)
    # wrong source prefix digest
    tc = copy.deepcopy(good)
    tc["source_prefix_sha256"] = "a" * 64
    with pytest.raises(co.CausalOperatorError,
                       match="prefix digest"):
        co.fit(spec, TRAIN, COLS, "train", tc)
    # data swapped under the same contract -> digest breaks
    with pytest.raises(co.CausalOperatorError,
                       match="prefix digest"):
        co.fit(spec, TRAIN + 1.0, COLS, "train", good)


def test_c15_derived_semantics_enforced():
    with pytest.raises(co.CausalOperatorError, match="0, 1"):
        co.validate_spec(_spec("ewma", alpha=2.0))
    with pytest.raises(co.CausalOperatorError, match="positive"):
        co.validate_spec(_spec("ewma", alpha=0.0))
    s = _spec("trailing_mean")
    s["lookback"] = 0                     # window=5 -> derived 4
    with pytest.raises(co.CausalOperatorError,
                       match="understated"):
        co.validate_spec(s)
    s2 = _spec("ewma")
    s2["lookback"] = 0                    # unbounded reach -> -1
    with pytest.raises(co.CausalOperatorError,
                       match="understated"):
        co.validate_spec(s2)
    assert co.derived_lookback("trailing_mean", {"window": 5}) == 4
    assert co.derived_lookback("ewma", {"alpha": 0.5}) == \
        co.LOOKBACK_UNBOUNDED
    assert co.derived_lookback("centered_mean_oracle",
                               {"window": 5}) == \
        co.LOOKBACK_NON_CAUSAL


def test_c14_snapshot_fsync_outcome_matrix(tmp_path, monkeypatch):
    """§4.4: both physical outcomes of a failed snapshot write are
    read by a FRESH descriptor and adjudicated from bytes."""
    import os as _os
    art = _fit("ewma")
    state = _init(art)
    real_fsync = _os.fsync
    # outcome A: bytes persisted although fsync raised
    monkeypatch.setattr(
        _os, "fsync",
        lambda fd: (_ for _ in ()).throw(OSError("injected")))
    with pytest.raises(co.StateWriteUncertain):
        co.save_state(state, art, tmp_path / "A")
    monkeypatch.setattr(_os, "fsync", real_fsync)
    files = list((tmp_path / "A").glob("*.state.json"))
    assert len(files) == 1
    sha = json.loads(files[0].read_text())["snapshot_sha256"]
    assert co.load_state(files[0], art, sha)["rows_seen"] == 0
    # outcome B: bytes did NOT persist -> nothing to load
    def eat_write(fd, payload):
        raise OSError("injected write loss")
    real_write = _os.write
    monkeypatch.setattr(_os, "write", eat_write)
    with pytest.raises(co.StateWriteUncertain):
        co.save_state(state, art, tmp_path / "B")
    monkeypatch.setattr(_os, "write", real_write)
    fb = list((tmp_path / "B").glob("*.state.json"))
    if fb:      # an empty husk must refuse on load
        with pytest.raises(co.CausalOperatorError):
            co.load_state(fb[0], art, "0" * 64)
    # outcome C: PARTIAL bytes -> digest refuses from a fresh read
    st2 = _init(art)
    _, st2 = co.transform_incremental(art, st2, SCORE[:4], COLS,
                                      _tc(4))
    sp = co.save_state(st2, art, tmp_path / "C")
    sha2 = json.loads(sp.read_text())["snapshot_sha256"]
    trunc = tmp_path / "C" / "truncated.state.json"
    trunc.write_bytes(sp.read_bytes()[: len(sp.read_bytes()) // 2])
    with pytest.raises(Exception):
        co.load_state(trunc, art, sha2)
    # a DIFFERENT snapshot can never overwrite an identity
    forged = sp.read_bytes().replace(b'"rows_seen": 4',
                                     b'"rows_seen": 9')
    victim = tmp_path / "C" / sp.name
    import stat as _stat
    _os.chmod(victim, 0o600)
    victim.write_bytes(forged)
    with pytest.raises(co.CausalOperatorError):
        co.load_state(victim, art, sha2)


def _mutant_co(old, new, name):
    import importlib.util as ilu
    import tempfile
    srcp = Path(co.__file__)
    text = srcp.read_text()
    assert old in text, "mutation anchor missing"
    mp = Path(tempfile.mkdtemp()) / f"{name}.py"
    mp.write_text(text.replace(old, new))
    spec = ilu.spec_from_file_location(name, mp)
    m = ilu.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_mut_c13_coherence_guard_bites():
    """Removing the payload/rows coherence check re-admits the PRE
    forgery — proving the C13 guard is what kills it."""
    m = _mutant_co(
        '''            if (v is None) != (rows == 0):
                raise CausalOperatorError(
                    "state ewma level incoherent with rows_seen — "
                    "fabricated history refused")''',
        '''            if False:
                raise CausalOperatorError("unreachable")''',
        "co_mut_c13")
    art = _fit("ewma")
    forged = _init(art)
    forged["payload"]["ewma"] = [-1986.0, -1986.0]
    with pytest.raises(co.CausalOperatorError):
        co.transform_incremental(art, copy.deepcopy(forged),
                                 SCORE[:1], COLS, _tc(1))
    # the same forgery under the mutant (its own code identity)
    art_m = m.fit(_spec("ewma"), TRAIN, COLS, "train",
                  _train_contract())
    forged_m = m.init_state(art_m, STREAM, 1.0)
    forged_m["payload"]["ewma"] = [-1986.0, -1986.0]
    out, _ = m.transform_incremental(
        art_m, forged_m, SCORE[:1], COLS, _tc(1))
    assert out[0, 0] < -500          # the mutant lets it steer


def test_mut_c15_lookback_guard_bites():
    m = _mutant_co(
        '''    want_lb = derived_lookback(spec["kind"], params)
    if spec["lookback"] != want_lb:''',
        '''    want_lb = derived_lookback(spec["kind"], params)
    if False:''',
        "co_mut_c15")
    s = _spec("trailing_mean")
    s["lookback"] = 0
    with pytest.raises(co.CausalOperatorError):
        co.validate_spec(s)
    assert m.validate_spec(dict(s))["lookback"] == 0
