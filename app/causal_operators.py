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
ARTIFACT_VERSION = "causal_operator_artifact.v1"

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


def _canonical(obj) -> bytes:
    return json.dumps(obj, sort_keys=True,
                      separators=(",", ":")).encode()


def code_identity() -> str:
    """Digest of THIS module's bytes — part of every spec digest."""
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _check_finite_matrix(x: np.ndarray, where: str) -> None:
    if not np.isfinite(x).all():
        raise CausalOperatorError(
            f"non-finite value in {where} — NaN/inf inputs are "
            "rejected unless an explicit missingness policy licenses "
            "them")


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
    if spec["fit_role"] not in FIT_ROLES:
        raise CausalOperatorError(
            "fit is limited to the declared training role")
    if type(spec["lookback"]) is not int or spec["lookback"] < 0:
        raise CausalOperatorError("lookback must be a nonnegative "
                                  "int")
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
        role: str) -> dict:
    """Fit on the TRAINING role only; returns the immutable artifact
    (JSON-only, digest-bound)."""
    validate_spec(spec)
    if role != spec["fit_role"]:
        raise CausalOperatorError(
            f"fit called on role {role!r}; the spec licenses only "
            f"{spec['fit_role']!r}")
    _check_input(columns, spec)
    x = np.asarray(train_matrix, dtype=float)
    if x.ndim != 2 or x.shape[1] != len(spec["columns"]):
        raise CausalOperatorError("train matrix shape mismatch")
    _check_finite_matrix(x, "train matrix")
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
                "fit_rows": int(x.shape[0])}
    artifact["artifact_sha256"] = hashlib.sha256(_canonical(
        {k: artifact[k] for k in ("schema", "spec", "spec_sha256",
                                  "fitted", "fit_rows")})).hexdigest()
    return artifact


def verify_artifact(artifact: dict) -> dict:
    for k in ("schema", "spec", "spec_sha256", "fitted", "fit_rows",
              "artifact_sha256"):
        if k not in artifact:
            raise CausalOperatorError(f"artifact missing {k!r}")
    if artifact["schema"] != ARTIFACT_VERSION:
        raise CausalOperatorError("unknown artifact schema")
    if spec_digest(artifact["spec"]) != artifact["spec_sha256"]:
        raise CausalOperatorError(
            "artifact spec digest does not re-derive (code or spec "
            "changed)")
    recomputed = hashlib.sha256(_canonical(
        {k: artifact[k] for k in ("schema", "spec", "spec_sha256",
                                  "fitted", "fit_rows")})).hexdigest()
    if recomputed != artifact["artifact_sha256"]:
        raise CausalOperatorError("artifact digest does not "
                                  "re-derive")
    return artifact


def save_artifact(artifact: dict, root: Path, name: str) -> Path:
    verify_artifact(artifact)
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    p = root / f"{name}.json"
    p.write_text(json.dumps(artifact, sort_keys=True, indent=1))
    return p


def load_artifact(path: Path) -> dict:
    return verify_artifact(json.loads(Path(path).read_text()))


# --------------------------- transforms ---------------------------
def init_state(artifact: dict) -> dict:
    verify_artifact(artifact)
    spec = artifact["spec"]
    kind = spec["kind"]
    state: dict = {"rows_seen": 0}
    n = len(spec["columns"])
    if kind in ("trailing_mean", "trailing_median"):
        state["buffer"] = [[] for _ in range(n)]
    elif kind == "ewma":
        state["ewma"] = [None] * n
    elif kind == "local_level_kalman":
        state["level"] = [None] * n
        state["var"] = [None] * n
    return state


def _step(kind: str, spec: dict, fitted: dict, state: dict,
          row: np.ndarray) -> np.ndarray:
    out = np.empty_like(row)
    if kind == "identity":
        out[:] = row
    elif kind in ("trailing_mean", "trailing_median"):
        w = spec["params"]["window"]
        for j, v in enumerate(row):
            buf = state["buffer"][j]
            buf.append(float(v))
            if len(buf) > w:
                buf.pop(0)
            out[j] = (float(np.mean(buf))
                      if kind == "trailing_mean"
                      else float(np.median(buf)))
    elif kind == "ewma":
        a = spec["params"]["alpha"]
        for j, v in enumerate(row):
            prev = state["ewma"][j]
            cur = (float(v) if prev is None
                   else a * float(v) + (1.0 - a) * prev)
            state["ewma"][j] = cur
            out[j] = cur
    elif kind == "local_level_kalman":
        for j, c in enumerate(spec["columns"]):
            f = fitted["per_column"][c]
            lvl, var = state["level"][j], state["var"][j]
            if lvl is None:
                lvl, var = f["init_level"], f["obs_var"]
            var = var + f["level_var"]
            k = var / (var + f["obs_var"])
            lvl = lvl + k * (float(row[j]) - lvl)
            var = (1.0 - k) * var
            state["level"][j], state["var"][j] = lvl, var
            out[j] = lvl
    else:
        raise CausalOperatorError(
            f"kind {kind!r} has no incremental path")
    state["rows_seen"] += 1
    return out


def transform_incremental(artifact: dict, state: dict,
                          chunk: np.ndarray, columns: list
                          ) -> tuple:
    verify_artifact(artifact)
    spec = artifact["spec"]
    if spec["kind"] in NON_CAUSAL_ORACLE_ONLY:
        raise CausalOperatorError(
            "a NON_CAUSAL_ORACLE_ONLY operator has no incremental "
            "path — it is a diagnostic ceiling, never deployable")
    _check_input(columns, spec)
    x = np.asarray(chunk, dtype=float)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    _check_finite_matrix(x, "incremental chunk")
    outs = np.empty_like(x)
    for i in range(x.shape[0]):
        outs[i] = _step(spec["kind"], spec, artifact["fitted"],
                        state, x[i])
    return outs, state


def transform_batch(artifact: dict, matrix: np.ndarray,
                    columns: list) -> np.ndarray:
    """Batch == incremental replay of the same accepted prefix, by
    construction: batch IS one incremental pass from fresh state."""
    verify_artifact(artifact)
    spec = artifact["spec"]
    _check_input(columns, spec)
    x = np.asarray(matrix, dtype=float)
    if x.ndim != 2:
        raise CausalOperatorError("batch matrix must be 2-D")
    _check_finite_matrix(x, "batch matrix")
    if spec["kind"] in NON_CAUSAL_ORACLE_ONLY:
        w = spec["params"]["window"]
        out = np.empty_like(x)
        for j in range(x.shape[1]):
            out[:, j] = np.convolve(x[:, j], np.ones(w) / w,
                                    mode="same")
        return out
    state = init_state(artifact)
    out, _ = transform_incremental(artifact, state, x, columns)
    return out


# ------------------------------ DAG -------------------------------
def validate_dag(nodes: dict, edges: list, input_columns: list
                 ) -> list:
    """Validated composition: unknown nodes, cycles and inapplicable
    configurations are ABSENT (refused), never scored."""
    for name, spec in nodes.items():
        validate_spec(spec)
    for a, b in edges:
        if a not in nodes or b not in nodes:
            raise CausalOperatorError(f"edge {a}->{b} names an "
                                      "unknown node")
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
