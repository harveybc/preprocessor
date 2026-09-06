"""Causal transformation operator protocol (T0.1, work plan 43 §4).

A small reusable core, independent of the CLI: strict canonical
specs, train-role-only fit, immutable JSON artifacts bound by
SHA-256, deterministic batch/incremental parity with carried state,
typed refusals, mandatory identity control and validated DAG
composition. No pickle as authority; no callables, imports,
expressions or absolute paths inside a spec; artifacts never depend
on the working directory or machine topology.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

SCHEMA_VERSION = "causal_operator_spec.v1"
ARTIFACT_VERSION = "causal_operator_artifact.v2"

OPERATOR_KINDS = ("identity", "trailing_mean", "trailing_median",
                  "ewma", "local_level_kalman",
                  "centered_mean_oracle")
NON_CAUSAL_ORACLE_ONLY = ("centered_mean_oracle",)
FIT_ROLES = ("train",)

_PARAM_SCHEMA = {
    "identity": {},
    "trailing_mean": {"window": int},
    "trailing_median": {"window": int},
    "ewma": {"alpha": float},
    "local_level_kalman": {},
    "centered_mean_oracle": {"window": int},
}
_FORBIDDEN_SPEC_TOKENS = ("lambda", "import", "__", "eval(",
                          "exec(", "/home/", "/Users/", "C:\\")


class CausalOperatorError(Exception):
    """Typed refusal: every message starts with REFUSED."""

    def __init__(self, msg: str):
        super().__init__(f"REFUSED: {msg}")


class StateWriteUncertain(CausalOperatorError):
    """C14: a snapshot write whose durability is UNKNOWN (failed
    fsync/close). The caller must treat the pathname as suspect and
    re-verify from a fresh descriptor; recovery reads bytes, never
    this exception."""


LOOKBACK_UNBOUNDED = -1          # ewma / kalman: full-history reach
LOOKBACK_NON_CAUSAL = -2         # centered oracle: future reach


def derived_lookback(kind: str, params: dict) -> int:
    """C15: causal lookback DERIVES from operator semantics; a
    declared value is only accepted when it equals this."""
    if kind == "identity":
        return 0
    if kind in ("trailing_mean", "trailing_median"):
        return int(params["window"]) - 1
    if kind in ("ewma", "local_level_kalman"):
        return LOOKBACK_UNBOUNDED
    if kind == "centered_mean_oracle":
        return LOOKBACK_NON_CAUSAL
    raise CausalOperatorError(f"kind {kind!r} has no derived "
                              "lookback")


def _canonical(obj) -> bytes:
    return json.dumps(obj, sort_keys=True,
                      separators=(",", ":")).encode()


def code_identity() -> str:
    """Digest of THIS module's bytes — part of every spec digest."""
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


AVAILABILITY_RULES = ("bar_close",)


def _strict_numeric(x, where: str) -> np.ndarray:
    """C1: reject booleans, strings, objects, complex and implicit
    coercion BEFORE any conversion; accepted matrices must already
    hold a real numeric dtype with finite values."""
    arr = np.asarray(x)
    if arr.dtype.kind not in ("f", "i"):
        raise CausalOperatorError(
            f"{where} dtype {arr.dtype!r} is not an explicitly "
            "supported real numeric dtype — bool/str/object/complex "
            "inputs are rejected, never coerced")
    if arr.dtype.kind == "b":
        raise CausalOperatorError(
            f"boolean matrix rejected in {where}")
    out = arr.astype(float)
    if not np.isfinite(out).all():
        raise CausalOperatorError(
            f"non-finite value in {where} — NaN/inf inputs are "
            "rejected unless an explicit missingness policy licenses "
            "them")
    return out


def _check_time_contract(tc: dict, n_rows: int,
                         where: str) -> None:
    """C1/C13: every used observation must satisfy its availability
    rule at the decision `as_of`, and the contract names the STREAM
    the rows came from. Future or unfinalized observations refuse.
    Mandatory on every deployable call."""
    if not isinstance(tc, dict) or set(tc) != {
            "as_of", "observation_ts", "finalized_ts",
            "stream_id"}:
        raise CausalOperatorError(
            f"time contract in {where} must carry exactly as_of, "
            "observation_ts, finalized_ts and stream_id")
    if type(tc["stream_id"]) is not str or not tc["stream_id"]:
        raise CausalOperatorError(
            f"stream_id in {where} must be a nonempty string")
    as_of = tc["as_of"]
    if isinstance(as_of, bool) or not isinstance(
            as_of, (int, float)) or not math.isfinite(float(as_of)):
        raise CausalOperatorError("as_of must be a finite number")
    obs = _strict_numeric(tc["observation_ts"],
                          f"{where} observation_ts").reshape(-1)
    fin = _strict_numeric(tc["finalized_ts"],
                          f"{where} finalized_ts").reshape(-1)
    if len(obs) != n_rows or len(fin) != n_rows:
        raise CausalOperatorError(
            f"time contract length differs from the rows used in "
            f"{where}")
    if (fin < obs).any():
        raise CausalOperatorError(
            "finalization before observation is impossible")
    if (obs > float(as_of)).any() or (fin > float(as_of)).any():
        raise CausalOperatorError(
            "future or unfinalized observation at the decision "
            "as_of — refused")
    if n_rows > 1 and not (np.diff(obs) > 0).all():
        raise CausalOperatorError(
            f"observation_ts in {where} must be STRICTLY "
            "increasing — shuffled or duplicate observations "
            "refuse")


def _rolling_prefix(prev_hex: str, row_vals, obs_t: float,
                    fin_t: float) -> str:
    """C13: rolling digest of the exact consumed prefix — one hash
    per accepted row chained onto the previous prefix identity."""
    return hashlib.sha256(
        bytes.fromhex(prev_hex) + _canonical(
            [float(obs_t), float(fin_t),
             [float(v) for v in np.asarray(row_vals).reshape(-1)]]
        )).hexdigest()


def stream_genesis_prefix(stream_id: str) -> str:
    return hashlib.sha256(
        f"stream_genesis:{stream_id}".encode()).hexdigest()


def validate_spec(spec: dict) -> dict:
    """Strict canonical spec: exact keys, exact primitive types
    (bool is never a number), no callables/imports/paths."""
    required = {"schema", "operator_id", "kind", "version", "params",
                "columns", "fit_role", "lookback",
                "availability_rule"}
    if not isinstance(spec, dict) or set(spec) != required:
        raise CausalOperatorError(
            f"spec keys must be exactly {sorted(required)}")
    if spec["schema"] != SCHEMA_VERSION:
        raise CausalOperatorError("unknown spec schema")
    if spec["kind"] not in OPERATOR_KINDS:
        raise CausalOperatorError(
            f"unknown operator kind {spec['kind']!r}")
    for key in ("operator_id", "version", "availability_rule"):
        if type(spec[key]) is not str or not spec[key]:
            raise CausalOperatorError(f"{key} must be a nonempty "
                                      "string")
    if spec["availability_rule"] not in AVAILABILITY_RULES:
        raise CausalOperatorError(
            f"availability_rule must be one of "
            f"{AVAILABILITY_RULES} — free-form rules are rejected")
    if spec["fit_role"] not in FIT_ROLES:
        raise CausalOperatorError(
            "fit is limited to the declared training role")
    if type(spec["lookback"]) is not int:
        raise CausalOperatorError("lookback must be an int")
    cols = spec["columns"]
    if (not isinstance(cols, list) or not cols
            or any(type(c) is not str for c in cols)):
        raise CausalOperatorError("columns must be a nonempty list "
                                  "of strings")
    if len(set(cols)) != len(cols):
        raise CausalOperatorError("duplicate column in spec")
    pschema = _PARAM_SCHEMA[spec["kind"]]
    params = spec["params"]
    if not isinstance(params, dict) or set(params) != set(pschema):
        raise CausalOperatorError(
            f"params for {spec['kind']} must be exactly "
            f"{sorted(pschema)}")
    for k, t in pschema.items():
        v = params[k]
        if type(v) is not t:      # bool is not int; int is not float
            raise CausalOperatorError(
                f"param {k!r} must be {t.__name__}, got "
                f"{type(v).__name__}")
        if t in (int, float) and (isinstance(v, bool)
                                  or not math.isfinite(float(v))
                                  or v <= 0):
            raise CausalOperatorError(
                f"param {k!r} must be a finite positive "
                f"{t.__name__}")
    if spec["kind"] == "ewma" and not (0.0 < params["alpha"]
                                       <= 1.0):
        raise CausalOperatorError(
            "ewma alpha must lie in (0, 1] — its meaningful "
            "smoothing interval")
    want_lb = derived_lookback(spec["kind"], params)
    if spec["lookback"] != want_lb:
        raise CausalOperatorError(
            f"declared lookback {spec['lookback']} does not equal "
            f"the value derived from the operator semantics "
            f"({want_lb}) — causal reach cannot be understated")
    blob = json.dumps(spec)
    low = blob.lower()
    for tok in _FORBIDDEN_SPEC_TOKENS:
        if tok.lower() in low:
            raise CausalOperatorError(
                f"forbidden token {tok!r} inside the spec — no "
                "callables, imports, expressions or absolute paths")
    if spec["kind"] == "centered_mean_oracle" and \
            "NON_CAUSAL_ORACLE_ONLY" not in spec["operator_id"]:
        raise CausalOperatorError(
            "a centered filter must carry NON_CAUSAL_ORACLE_ONLY in "
            "its operator_id — it can never look causal")
    return spec


def spec_digest(spec: dict) -> str:
    validate_spec(spec)
    return hashlib.sha256(_canonical(
        {"spec": spec, "code_identity": code_identity()})
    ).hexdigest()


def _check_input(df_columns, spec: dict) -> None:
    cols = list(df_columns)
    if cols != list(spec["columns"]):
        raise CausalOperatorError(
            f"input columns {cols} do not match the spec's exact "
            f"columns/order {spec['columns']} — missing, duplicate, "
            "reordered or extra inputs are rejected")


# ------------------------------ fit -------------------------------
def fit(spec: dict, train_matrix: np.ndarray, columns: list,
        role: str, train_contract: dict = None) -> dict:
    """Fit on the TRAINING role only; returns the immutable artifact
    (JSON-only, digest-bound). C15: fitting binds an exact train
    interval, a timestamp/finalization contract and the source-
    prefix digest — `role` is a fact proven by the contract, never
    a caller label."""
    validate_spec(spec)
    if role != spec["fit_role"]:
        raise CausalOperatorError(
            f"fit called on role {role!r}; the spec licenses only "
            f"{spec['fit_role']!r}")
    _check_input(columns, spec)
    x = _strict_numeric(train_matrix, "train matrix")
    if x.ndim != 2 or x.shape[1] != len(spec["columns"]):
        raise CausalOperatorError("train matrix shape mismatch")
    if not isinstance(train_contract, dict) or set(
            train_contract) != {"stream_id", "interval",
                                "time_contract",
                                "source_prefix_sha256"}:
        raise CausalOperatorError(
            "fit requires a train contract with exactly stream_id, "
            "interval, time_contract and source_prefix_sha256 — "
            "an unbound training role is a label, not a fact")
    tc = train_contract["time_contract"]
    _check_time_contract(tc, x.shape[0], "train contract")
    if tc["stream_id"] != train_contract["stream_id"]:
        raise CausalOperatorError(
            "train contract stream_id differs from its own time "
            "contract")
    iv = train_contract["interval"]
    if (not isinstance(iv, (list, tuple)) or len(iv) != 2
            or not all(isinstance(v, (int, float))
                       and not isinstance(v, bool)
                       and math.isfinite(float(v)) for v in iv)
            or not float(iv[0]) <= float(iv[1])):
        raise CausalOperatorError(
            "train interval must be a finite [lo, hi] pair")
    obs = np.asarray(tc["observation_ts"], dtype=float)
    if obs[0] < float(iv[0]) or obs[-1] > float(iv[1]):
        raise CausalOperatorError(
            "train observations fall outside the declared train "
            "interval")
    want = train_contract["source_prefix_sha256"]
    if type(want) is not str or len(want) != 64:
        raise CausalOperatorError(
            "source_prefix_sha256 must be a 64-hex digest")
    fin = np.asarray(tc["finalized_ts"], dtype=float)
    prefix = stream_genesis_prefix(train_contract["stream_id"])
    for i in range(x.shape[0]):
        prefix = _rolling_prefix(prefix, x[i], obs[i], fin[i])
    if prefix != want:
        raise CausalOperatorError(
            "train source prefix digest does not re-derive from "
            "the fitted rows — the fit is not bound to the claimed "
            "data")
    fitted: dict = {}
    if spec["kind"] == "local_level_kalman":
        fitted["per_column"] = {}
        for j, c in enumerate(spec["columns"]):
            v = x[:, j]
            dv = np.diff(v)
            if len(dv) < 10:
                raise CausalOperatorError(
                    "train role too short for Kalman fit")
            # method-of-moments local level: Var(dv) = 2R + Q with
            # lag-1 autocov(dv) = -R  (MA(1) structure)
            g0 = float(np.var(dv, ddof=1))
            g1 = float(np.cov(dv[:-1], dv[1:], ddof=1)[0, 1])
            r = max(-g1, 1e-12)
            q = max(g0 - 2.0 * r, 1e-12)
            fitted["per_column"][c] = {"obs_var": r,
                                       "level_var": q,
                                       "init_level": float(v[0])}
    artifact = {"schema": ARTIFACT_VERSION,
                "spec": spec,
                "spec_sha256": spec_digest(spec),
                "fitted": fitted,
                "fit_rows": int(x.shape[0]),
                "train_binding": {
                    "stream_id": train_contract["stream_id"],
                    "interval": [float(iv[0]), float(iv[1])],
                    "source_prefix_sha256": want}}
    artifact["artifact_sha256"] = hashlib.sha256(_canonical(
        {k: artifact[k] for k in ("schema", "spec", "spec_sha256",
                                  "fitted", "fit_rows",
                                  "train_binding")})).hexdigest()
    return artifact


def verify_artifact(artifact: dict) -> dict:
    for k in ("schema", "spec", "spec_sha256", "fitted", "fit_rows",
              "train_binding", "artifact_sha256"):
        if k not in artifact:
            raise CausalOperatorError(f"artifact missing {k!r}")
    if artifact["schema"] != ARTIFACT_VERSION:
        raise CausalOperatorError("unknown artifact schema")
    if spec_digest(artifact["spec"]) != artifact["spec_sha256"]:
        raise CausalOperatorError(
            "artifact spec digest does not re-derive (code or spec "
            "changed)")
    tb = artifact["train_binding"]
    if not isinstance(tb, dict) or set(tb) != {
            "stream_id", "interval", "source_prefix_sha256"}:
        raise CausalOperatorError(
            "artifact train_binding must carry exactly stream_id, "
            "interval and source_prefix_sha256")
    recomputed = hashlib.sha256(_canonical(
        {k: artifact[k] for k in ("schema", "spec", "spec_sha256",
                                  "fitted", "fit_rows",
                                  "train_binding")})).hexdigest()
    if recomputed != artifact["artifact_sha256"]:
        raise CausalOperatorError("artifact digest does not "
                                  "re-derive")
    return artifact


def save_artifact(artifact: dict, root: Path,
                  name: str = None) -> Path:
    """C4: content-addressed write-once — the file NAME is the
    artifact digest, created O_EXCL|O_NOFOLLOW with private mode,
    fsynced (file+dir). A different artifact can never overwrite an
    existing identity; identical content at the same digest is
    idempotently accepted."""
    import os
    import stat as _stat
    verify_artifact(artifact)
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    p = root / f"{artifact['artifact_sha256']}.json"
    payload = json.dumps(artifact, sort_keys=True,
                         indent=1).encode()
    if p.exists():
        if p.is_symlink() or not p.is_file():
            raise CausalOperatorError(
                "artifact path is not a regular file")
        if p.read_bytes() == payload:
            return p               # idempotent identical content
        raise CausalOperatorError(
            "a DIFFERENT artifact already occupies this identity — "
            "write-once refused")
    try:
        fd = os.open(str(p),
                     os.O_CREAT | os.O_EXCL | os.O_WRONLY
                     | getattr(os, "O_NOFOLLOW", 0), 0o400)
    except FileExistsError:
        raise CausalOperatorError(
            "concurrent artifact creation — write-once refused")
    except OSError as exc:
        raise CausalOperatorError(
            f"uncertain artifact durability: {exc} — failing "
            "closed")
    try:
        st = os.fstat(fd)
        if not _stat.S_ISREG(st.st_mode):
            raise CausalOperatorError(
                "artifact descriptor is not a regular file")
        os.write(fd, payload)
        os.fsync(fd)
    finally:
        os.close(fd)
    dfd = os.open(str(root), os.O_RDONLY)
    try:
        os.fsync(dfd)
    finally:
        os.close(dfd)
    return p


def _strict_load_json(path: Path) -> dict:
    def _no_dupes(pairs):
        keys = [k for k, _ in pairs]
        if len(keys) != len(set(keys)):
            raise CausalOperatorError(
                "duplicate JSON key in artifact")
        return dict(pairs)
    p = Path(path)
    if p.is_symlink() or not p.is_file():
        raise CausalOperatorError(
            "artifact path is not a regular file")
    try:
        return json.loads(
            p.read_text(), object_pairs_hook=_no_dupes,
            parse_constant=lambda c: (_ for _ in ()).throw(
                CausalOperatorError(
                    f"non-finite literal {c!r} in artifact")))
    except json.JSONDecodeError as exc:
        raise CausalOperatorError(
            f"malformed JSON bytes ({exc.msg} at char {exc.pos}) — "
            "partial or corrupt persistence refuses typed")


def load_artifact(path: Path) -> dict:
    doc = _strict_load_json(path)
    allowed = {"schema", "spec", "spec_sha256", "fitted",
               "fit_rows", "train_binding", "artifact_sha256"}
    if set(doc) != allowed:
        raise CausalOperatorError(
            "artifact carries unknown or missing fields")
    return verify_artifact(doc)


# --------------------------- transforms ---------------------------
STATE_VERSION = "causal_operator_state.v3"
SNAPSHOT_VERSION = "causal_operator_state_snapshot.v1"
GAP_POLICIES = ("refuse_gap", "accept_declared_gap")


def init_state(artifact: dict, stream_id: str,
               expected_step: float) -> dict:
    """C13: runtime state carries its FULL identity — schema,
    artifact digest, operator kind/version, exact columns, the
    stream it consumes, the expected observation step, the exact
    rows seen, the last accepted observation/finalization times and
    a rolling digest of the exact consumed prefix. An empty state
    is legal only at an explicit stream start."""
    verify_artifact(artifact)
    if type(stream_id) is not str or not stream_id:
        raise CausalOperatorError("stream_id must be a nonempty "
                                  "string")
    if stream_id != artifact["train_binding"]["stream_id"]:
        raise CausalOperatorError(
            "state stream differs from the artifact's bound "
            "training stream — foreign stream refused")
    if (isinstance(expected_step, bool)
            or not isinstance(expected_step, (int, float))
            or not math.isfinite(float(expected_step))
            or float(expected_step) <= 0):
        raise CausalOperatorError(
            "expected_step must be a finite positive number")
    spec = artifact["spec"]
    kind = spec["kind"]
    n = len(spec["columns"])
    payload: dict = {}
    if kind in ("trailing_mean", "trailing_median"):
        payload["buffer"] = [[] for _ in range(n)]
    elif kind == "ewma":
        payload["ewma"] = [None] * n
    elif kind == "local_level_kalman":
        payload["level"] = [None] * n
        payload["var"] = [None] * n
    return {"schema": STATE_VERSION,
            "artifact_sha256": artifact["artifact_sha256"],
            "kind": kind, "version": spec["version"],
            "columns": list(spec["columns"]),
            "stream_id": stream_id,
            "expected_step": float(expected_step),
            "rows_seen": 0,
            "last_observation_ts": None,
            "last_finalized_ts": None,
            "prefix_sha256": stream_genesis_prefix(stream_id),
            "payload": payload}


_STATE_KEYS = {"schema", "artifact_sha256", "kind", "version",
               "columns", "stream_id", "expected_step",
               "rows_seen", "last_observation_ts",
               "last_finalized_ts", "prefix_sha256", "payload"}
_PAYLOAD_KEYS = {"identity": set(),
                 "trailing_mean": {"buffer"},
                 "trailing_median": {"buffer"},
                 "ewma": {"ewma"},
                 "local_level_kalman": {"level", "var"}}


def _num_or_none(v) -> bool:
    return v is None or (not isinstance(v, bool)
                         and isinstance(v, (int, float))
                         and math.isfinite(float(v)))


def _check_state(state: dict, artifact: dict) -> None:
    """C13: exact state schema per operator — version, payload keys,
    array lengths/types/finiteness, rows_seen, last times, stream
    identity and the rolling prefix digest. Foreign, fabricated or
    incoherent state refuses BEFORE any incremental step."""
    if not isinstance(state, dict) or set(state) != _STATE_KEYS:
        raise CausalOperatorError(
            "runtime state keys are not the exact v3 schema")
    if state["schema"] != STATE_VERSION:
        raise CausalOperatorError(
            "runtime state has no valid identity schema")
    if state["artifact_sha256"] != artifact["artifact_sha256"]:
        raise CausalOperatorError(
            "state was produced under a DIFFERENT artifact — "
            "foreign state refused")
    spec = artifact["spec"]
    if state["kind"] != spec["kind"] or             state["columns"] != list(spec["columns"]):
        raise CausalOperatorError("state operator/column binding "
                                  "mismatch")
    if state["version"] != spec["version"]:
        raise CausalOperatorError(
            "state operator VERSION differs from the artifact "
            "spec — foreign operator generation refused")
    if state["stream_id"] !=             artifact["train_binding"]["stream_id"]:
        raise CausalOperatorError(
            "state stream identity differs from the artifact "
            "binding — foreign stream refused")
    if (isinstance(state["expected_step"], bool)
            or not isinstance(state["expected_step"], (int, float))
            or not math.isfinite(float(state["expected_step"]))
            or float(state["expected_step"]) <= 0):
        raise CausalOperatorError("state expected_step invalid")
    rows = state["rows_seen"]
    if type(rows) is not int or rows < 0:
        raise CausalOperatorError("state rows_seen invalid")
    for k in ("last_observation_ts", "last_finalized_ts"):
        if not _num_or_none(state[k]):
            raise CausalOperatorError(f"state {k} invalid")
    if (state["last_observation_ts"] is None) != (rows == 0) or             (state["last_finalized_ts"] is None) != (rows == 0):
        raise CausalOperatorError(
            "state last-times are incoherent with rows_seen — a "
            "fabricated history refuses")
    pfx = state["prefix_sha256"]
    if type(pfx) is not str or len(pfx) != 64:
        raise CausalOperatorError("state prefix digest invalid")
    if rows == 0 and pfx != stream_genesis_prefix(
            state["stream_id"]):
        raise CausalOperatorError(
            "an empty state must carry the stream genesis prefix")
    pay = state["payload"]
    kind = spec["kind"]
    n = len(spec["columns"])
    if not isinstance(pay, dict) or set(pay) !=             _PAYLOAD_KEYS[kind]:
        raise CausalOperatorError(
            f"state payload keys are not the exact schema for "
            f"{kind}")
    if kind in ("trailing_mean", "trailing_median"):
        w = spec["params"]["window"]
        bufs = pay["buffer"]
        if (not isinstance(bufs, list) or len(bufs) != n
                or any(not isinstance(b, list) for b in bufs)):
            raise CausalOperatorError("state buffer shape invalid")
        want_len = min(rows, w)
        for b in bufs:
            if len(b) != want_len:
                raise CausalOperatorError(
                    "state buffer length is incoherent with "
                    "rows_seen — fabricated history refused")
            for v in b:
                if isinstance(v, bool) or not isinstance(
                        v, (int, float)) or not math.isfinite(
                        float(v)):
                    raise CausalOperatorError(
                        "non-finite/foreign value in state buffer")
    elif kind == "ewma":
        vals = pay["ewma"]
        if not isinstance(vals, list) or len(vals) != n:
            raise CausalOperatorError("state ewma shape invalid")
        for v in vals:
            if not _num_or_none(v):
                raise CausalOperatorError(
                    "non-finite/foreign value in state ewma")
            if (v is None) != (rows == 0):
                raise CausalOperatorError(
                    "state ewma level incoherent with rows_seen — "
                    "fabricated history refused")
    elif kind == "local_level_kalman":
        for key in ("level", "var"):
            vals = pay[key]
            if not isinstance(vals, list) or len(vals) != n:
                raise CausalOperatorError(
                    f"state {key} shape invalid")
            for v in vals:
                if not _num_or_none(v):
                    raise CausalOperatorError(
                        f"non-finite/foreign value in state {key}")
                if (v is None) != (rows == 0):
                    raise CausalOperatorError(
                        f"state {key} incoherent with rows_seen")


def save_state(state: dict, artifact: dict, root: Path,
               parent_snapshot_sha256: str = None) -> Path:
    """C14: durable CONTENT-ADDRESSED snapshot — exact JSON,
    canonical digest as the file name, O_EXCL|O_NOFOLLOW private
    create, fsync(file)+fsync(dir), descriptor-first checks. An
    identical snapshot is idempotent; a different one can never
    overwrite an identity. Durability doubt raises the typed
    StateWriteUncertain."""
    import os
    import stat as _stat
    _check_state(state, artifact)
    if parent_snapshot_sha256 is not None and (
            type(parent_snapshot_sha256) is not str
            or len(parent_snapshot_sha256) != 64):
        raise CausalOperatorError(
            "parent_snapshot_sha256 must be a 64-hex digest")
    body = {"schema": SNAPSHOT_VERSION, "state": state,
            "parent_snapshot_sha256": parent_snapshot_sha256}
    snap_sha = hashlib.sha256(_canonical(body)).hexdigest()
    doc = dict(body, snapshot_sha256=snap_sha)
    payload = json.dumps(doc, sort_keys=True, indent=1).encode()
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    p = root / f"{snap_sha}.state.json"
    if p.exists():
        if p.is_symlink() or not p.is_file():
            raise CausalOperatorError(
                "snapshot path is not a regular file")
        if p.read_bytes() == payload:
            return p               # idempotent identical content
        raise CausalOperatorError(
            "a DIFFERENT snapshot already occupies this identity — "
            "write-once refused")
    try:
        fd = os.open(str(p),
                     os.O_CREAT | os.O_EXCL | os.O_WRONLY
                     | getattr(os, "O_NOFOLLOW", 0), 0o400)
    except FileExistsError:
        raise CausalOperatorError(
            "concurrent snapshot creation — write-once refused")
    except OSError as exc:
        raise StateWriteUncertain(
            f"uncertain snapshot create: {exc} — failing closed")
    try:
        st = os.fstat(fd)
        if not _stat.S_ISREG(st.st_mode):
            raise CausalOperatorError(
                "snapshot descriptor is not a regular file")
        os.write(fd, payload)
        os.fsync(fd)
    except CausalOperatorError:
        raise
    except OSError as exc:
        raise StateWriteUncertain(
            f"uncertain snapshot durability: {exc} — the bytes may "
            "or may not have persisted; recovery must read them")
    finally:
        os.close(fd)
    try:
        dfd = os.open(str(root), os.O_RDONLY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
    except OSError as exc:
        raise StateWriteUncertain(
            f"uncertain snapshot directory durability: {exc}")
    return p


def load_state(path: Path, artifact: dict,
               expected_snapshot_sha256: str) -> dict:
    """C13/C14: state AUTHORITY comes from the consuming run
    contract — the caller must name the exact snapshot digest it
    expects. A producer checksum alone is integrity, never
    provenance."""
    if type(expected_snapshot_sha256) is not str or             len(expected_snapshot_sha256) != 64:
        raise CausalOperatorError(
            "load_state requires the expected snapshot digest from "
            "the consuming run contract")
    p = Path(path)
    if p.is_symlink() or not p.is_file():
        raise CausalOperatorError(
            "snapshot path is not a regular file")
    doc = _strict_load_json(p)
    if not isinstance(doc, dict) or set(doc) != {
            "schema", "state", "parent_snapshot_sha256",
            "snapshot_sha256"}:
        raise CausalOperatorError(
            "snapshot keys are not the exact schema")
    if doc["schema"] != SNAPSHOT_VERSION:
        raise CausalOperatorError("unknown snapshot schema")
    body = {"schema": doc["schema"], "state": doc["state"],
            "parent_snapshot_sha256":
                doc["parent_snapshot_sha256"]}
    if hashlib.sha256(_canonical(body)).hexdigest() !=             doc["snapshot_sha256"]:
        raise CausalOperatorError(
            "snapshot digest does not re-derive")
    if doc["snapshot_sha256"] != expected_snapshot_sha256:
        raise CausalOperatorError(
            "snapshot digest differs from the one the run contract "
            "authorizes — foreign state refused")
    state = doc["state"]
    _check_state(state, artifact)
    return state


def _step(kind: str, spec: dict, fitted: dict, state: dict,
          row: np.ndarray) -> np.ndarray:
    out = np.empty_like(row)
    pay = state["payload"]
    if kind == "identity":
        out[:] = row
    elif kind in ("trailing_mean", "trailing_median"):
        w = spec["params"]["window"]
        for j, v in enumerate(row):
            buf = pay["buffer"][j]
            buf.append(float(v))
            if len(buf) > w:
                buf.pop(0)
            out[j] = (float(np.mean(buf))
                      if kind == "trailing_mean"
                      else float(np.median(buf)))
    elif kind == "ewma":
        a = spec["params"]["alpha"]
        for j, v in enumerate(row):
            prev = pay["ewma"][j]
            cur = (float(v) if prev is None
                   else a * float(v) + (1.0 - a) * prev)
            pay["ewma"][j] = cur
            out[j] = cur
    elif kind == "local_level_kalman":
        for j, c in enumerate(spec["columns"]):
            f = fitted["per_column"][c]
            lvl, var = pay["level"][j], pay["var"][j]
            if lvl is None:
                lvl, var = f["init_level"], f["obs_var"]
            var = var + f["level_var"]
            k = var / (var + f["obs_var"])
            lvl = lvl + k * (float(row[j]) - lvl)
            var = (1.0 - k) * var
            pay["level"][j], pay["var"][j] = lvl, var
            out[j] = lvl
    else:
        raise CausalOperatorError(
            f"kind {kind!r} has no incremental path")
    state["rows_seen"] += 1
    return out


def transform_incremental(artifact: dict, state: dict,
                          chunk: np.ndarray, columns: list,
                          time_contract: dict = None,
                          gap_policy: str = "refuse_gap") -> tuple:
    """C13: continuation REQUIRES the chunk to extend the accepted
    prefix monotonically at the declared step — replay, rewind, gap
    and fork refuse (a gap only passes under the explicit
    accept_declared_gap policy). The rolling prefix digest and last
    accepted times advance with every row."""
    verify_artifact(artifact)
    spec = artifact["spec"]
    if spec["kind"] in NON_CAUSAL_ORACLE_ONLY:
        raise CausalOperatorError(
            "a NON_CAUSAL_ORACLE_ONLY operator has no incremental "
            "path — it is a diagnostic ceiling, never deployable")
    _check_state(state, artifact)
    _check_input(columns, spec)
    if gap_policy not in GAP_POLICIES:
        raise CausalOperatorError(
            f"gap_policy must be one of {GAP_POLICIES}")
    x = _strict_numeric(chunk, "incremental chunk")
    if x.ndim == 1:
        x = x.reshape(1, -1)
    if time_contract is None:
        raise CausalOperatorError(
            "a deployable transform call must consume a timestamp "
            "contract (as_of + per-row observation/finalization + "
            "stream identity)")
    _check_time_contract(time_contract, x.shape[0],
                         "incremental chunk")
    if time_contract["stream_id"] != state["stream_id"]:
        raise CausalOperatorError(
            "chunk stream differs from the state's stream — "
            "foreign stream refused")
    obs = np.asarray(time_contract["observation_ts"],
                     dtype=float).reshape(-1)
    fin = np.asarray(time_contract["finalized_ts"],
                     dtype=float).reshape(-1)
    step = float(state["expected_step"])
    tol = 1e-9 * max(1.0, abs(step))
    if state["rows_seen"] > 0:
        last = float(state["last_observation_ts"])
        if obs[0] <= last:
            raise CausalOperatorError(
                "chunk begins at or before the last accepted "
                "observation — replay/rewind refused")
        if abs(obs[0] - (last + step)) > tol and                 gap_policy != "accept_declared_gap":
            raise CausalOperatorError(
                f"observation gap: chunk starts at {obs[0]}, the "
                f"accepted prefix ends at {last} with step {step} "
                "— refused without an explicit gap policy")
    if x.shape[0] > 1 and (np.abs(np.diff(obs) - step)
                           > tol).any() and             gap_policy != "accept_declared_gap":
        raise CausalOperatorError(
            "irregular observation step INSIDE the chunk — refused "
            "without an explicit gap policy")
    outs = np.empty_like(x)
    for i in range(x.shape[0]):
        outs[i] = _step(spec["kind"], spec, artifact["fitted"],
                        state, x[i])
        state["prefix_sha256"] = _rolling_prefix(
            state["prefix_sha256"], x[i], obs[i], fin[i])
        state["last_observation_ts"] = float(obs[i])
        state["last_finalized_ts"] = float(fin[i])
    return outs, state


def make_bar_close_contract(n_rows: int,
                            start: float = 0.0,
                            stream_id: str =
                            "synthetic_bar_close") -> dict:
    """Convenience: a bar_close contract where every row finalized
    at its own index and the decision sits after the last row."""
    ts = [start + float(i) for i in range(n_rows)]
    return {"as_of": start + float(n_rows),
            "observation_ts": ts, "finalized_ts": list(ts),
            "stream_id": stream_id}


def make_train_contract(matrix, n_rows: int, start: float = 0.0,
                        stream_id: str =
                        "synthetic_bar_close") -> dict:
    """Convenience for tests/labs: the bound train contract whose
    prefix digest is derived from the exact rows being fitted."""
    tc = make_bar_close_contract(n_rows, start, stream_id)
    x = _strict_numeric(matrix, "train contract matrix")
    if x.ndim != 2 or x.shape[0] != n_rows:
        raise CausalOperatorError("train contract shape mismatch")
    prefix = stream_genesis_prefix(stream_id)
    for i in range(n_rows):
        prefix = _rolling_prefix(prefix, x[i],
                                 tc["observation_ts"][i],
                                 tc["finalized_ts"][i])
    return {"stream_id": stream_id,
            "interval": [float(start), float(start + n_rows - 1)],
            "time_contract": tc,
            "source_prefix_sha256": prefix}


def transform_batch(artifact: dict, matrix: np.ndarray,
                    columns: list,
                    time_contract: dict = None) -> np.ndarray:
    """Batch == incremental replay of the same accepted prefix, by
    construction: batch IS one incremental pass from fresh state."""
    verify_artifact(artifact)
    spec = artifact["spec"]
    _check_input(columns, spec)
    x = _strict_numeric(matrix, "batch matrix")
    if x.ndim != 2:
        raise CausalOperatorError("batch matrix must be 2-D")
    if time_contract is None:
        raise CausalOperatorError(
            "a deployable transform call must consume a timestamp "
            "contract (as_of + per-row observation/finalization)")
    _check_time_contract(time_contract, x.shape[0], "batch matrix")
    if spec["kind"] in NON_CAUSAL_ORACLE_ONLY:
        w = spec["params"]["window"]
        out = np.empty_like(x)
        for j in range(x.shape[1]):
            out[:, j] = np.convolve(x[:, j], np.ones(w) / w,
                                    mode="same")
        return out
    state = init_state(artifact, time_contract["stream_id"],
                       _batch_step(time_contract))
    out, _ = transform_incremental(artifact, state, x, columns,
                                   time_contract=time_contract)
    return out


def _batch_step(tc: dict) -> float:
    obs = np.asarray(tc["observation_ts"], dtype=float).reshape(-1)
    if len(obs) < 2:
        return 1.0
    return float(obs[1] - obs[0])


# ------------------------------ DAG -------------------------------
def validate_dag(nodes: dict, edges: list, input_columns: list
                 ) -> list:
    """C3: EVERY edge's exact output/input schema and order is
    validated (our operators preserve columns, so a child's columns
    must equal its parent's), duplicate edges refuse, more than one
    component refuses as disconnected ambiguity, cycles refuse, and
    roots must match the input contract. Invalid compositions are
    ABSENT, never scored."""
    for name, spec in nodes.items():
        validate_spec(spec)
    seen_edges = set()
    for a, b in edges:
        if (a, b) in seen_edges:
            raise CausalOperatorError(f"duplicate edge {a}->{b}")
        seen_edges.add((a, b))
        if a not in nodes or b not in nodes:
            raise CausalOperatorError(f"edge {a}->{b} names an "
                                      "unknown node")
        if list(nodes[a]["columns"]) != list(nodes[b]["columns"]):
            raise CausalOperatorError(
                f"edge {a}->{b} is schema-incompatible: parent "
                f"outputs {nodes[a]['columns']} but the child "
                f"expects {nodes[b]['columns']}")
    if len(nodes) > 1:
        undirected = {}
        for a, b in edges:
            undirected.setdefault(a, set()).add(b)
            undirected.setdefault(b, set()).add(a)
        start = next(iter(nodes))
        stack, comp = [start], {start}
        while stack:
            n = stack.pop()
            for m in undirected.get(n, ()):  # noqa: E501
                if m not in comp:
                    comp.add(m)
                    stack.append(m)
        if comp != set(nodes):
            raise CausalOperatorError(
                "disconnected composition is ambiguous — refused")
    order, seen, doing = [], set(), set()
    children = {}
    for a, b in edges:
        children.setdefault(a, []).append(b)
    parents = {n: [a for a, b in edges if b == n] for n in nodes}

    def visit(n):
        if n in doing:
            raise CausalOperatorError("composition graph has a "
                                      "cycle")
        if n in seen:
            return
        doing.add(n)
        for c in children.get(n, []):
            visit(c)
        doing.discard(n)
        seen.add(n)
        order.append(n)

    for n in nodes:
        visit(n)
    order.reverse()
    for n in order:
        if not parents[n]:
            if list(nodes[n]["columns"]) != list(input_columns):
                raise CausalOperatorError(
                    f"root node {n!r} is inapplicable to the input "
                    "contract — absent, not scored")
    return order


def build_graph_artifact(nodes: dict, edges: list,
                         input_columns: list,
                         fitted_artifacts: dict) -> dict:
    """C3: the canonical graph artifact — node specs, fitted
    artifact digests, edges, topological order and the input/output
    contracts, all under one digest. Composed EXECUTION is not
    implemented in this correction: the artifact is labeled
    VALIDATED_NOT_EXECUTABLE and claims no reusable DAG runtime."""
    order = validate_dag(nodes, edges, input_columns)
    for name in nodes:
        if name not in fitted_artifacts:
            raise CausalOperatorError(
                f"graph node {name!r} has no fitted artifact")
        verify_artifact(fitted_artifacts[name])
        if fitted_artifacts[name]["spec"] != nodes[name]:
            raise CausalOperatorError(
                f"graph node {name!r} artifact does not match its "
                "spec")
    doc = {"schema": "causal_operator_graph.v1",
           "status": "VALIDATED_NOT_EXECUTABLE",
           "nodes": {n: nodes[n] for n in sorted(nodes)},
           "node_artifact_sha256": {
               n: fitted_artifacts[n]["artifact_sha256"]
               for n in sorted(nodes)},
           "edges": sorted([list(e) for e in edges]),
           "topological_order": order,
           "input_contract": list(input_columns),
           "output_contract": list(nodes[order[-1]]["columns"]),
           "code_identity": code_identity()}
    doc["graph_sha256"] = hashlib.sha256(_canonical(
        {k: doc[k] for k in doc if k != "graph_sha256"})
    ).hexdigest()
    return doc
