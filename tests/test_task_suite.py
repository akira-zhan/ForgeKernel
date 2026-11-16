"""The forgekernel task suite of src/forgekernel/card_template/eval/leaderboard.py: input groups, the
judge, the calibrated tolerance and the calibration itself. CPU only (timing needs CUDA)."""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

REPO_ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "fk_template_leaderboard_task", REPO_ROOT / "src" / "forgekernel" / "card_template" / "eval" / "leaderboard.py")
lb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lb)

JUDGE = """import torch

def run(x, w):
    y = x.to(torch.float64)
    y = y * torch.rsqrt(y.pow(2).mean(dim=-1, keepdim=True) + 1e-6) * w.to(torch.float64)
    return y.to(x.dtype)

def make_inputs(axes, group, seed, device):
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(axes["m"], axes["n"], generator=g)
    if group == "spiky":
        x[:, 0] = 100.0
    return {"x": x.to(device), "w": torch.ones(axes["n"]).to(device)}
"""

DEFN = {"name": "rms", "axes": {"m": {"type": "var"}, "n": {"type": "var"}},
        "inputs": {"x": {"shape": ["m", "n"], "dtype": "float32"},
                   "w": {"shape": ["n"], "dtype": "float32"}},
        "outputs": {"y": {"shape": ["m", "n"], "dtype": "float32"}},
        "reference": JUDGE, "custom_inputs_entrypoint": "make_inputs"}

GROUPS = [{"name": "plain", "kind": "generated", "seed": 1},
          {"name": "spiky", "kind": "generated", "seed": 2}]


def _problem(tmp_path, shapes=(("small", 4, 16),), groups=GROUPS, tolerance=None) -> dict:
    wls = [{"uuid": n, "axes": {"m": m, "n": k}, "input_groups": [dict(g) for g in groups]}
           for n, m, k in shapes]
    p = {"suite": lb.SUITE_TASK, "definition": json.loads(json.dumps(DEFN)), "workload": wls[0],
         "dps": False, "timing": "device", "_dir": str(tmp_path),
         "reference_impl": {"kind": "definition"}, "time_reference": False}
    if len(wls) > 1:
        p["group"] = [{"index": i, "name": w["uuid"], "workload": w} for i, w in enumerate(wls)]
    if tolerance is not None:
        p["tolerance"] = {"outputs": tolerance}
    return p


def _rms(x, w):
    y = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-6) * w
    return y


def test_generated_groups_follow_the_signature(tmp_path):
    p = _problem(tmp_path)
    plain = lb.task_inputs(p, GROUPS[0], "cpu")
    spiky = lb.task_inputs(p, GROUPS[1], "cpu")
    assert [tuple(t.shape) for t in plain] == [(4, 16), (16,)]
    assert float(spiky[0][0, 0]) == 100.0 and float(plain[0][0, 0]) != 100.0


def test_a_file_group_supplies_every_input(tmp_path):
    import safetensors.torch as st
    st.save_file({"x": torch.zeros(4, 16), "w": torch.ones(16)}, str(tmp_path / "g.safetensors"))
    p = _problem(tmp_path)
    inp = lb.task_inputs(p, {"name": "zeros", "kind": "file", "path": "g.safetensors"}, "cpu")
    assert float(inp[0].abs().sum()) == 0.0


def test_an_input_off_the_signature_is_an_input_error(tmp_path):
    import safetensors.torch as st
    st.save_file({"x": torch.zeros(4, 8), "w": torch.ones(16)}, str(tmp_path / "g.safetensors"))
    p = _problem(tmp_path)
    with pytest.raises(ValueError, match="the signature says"):
        lb.task_inputs(p, {"name": "bad", "kind": "file", "path": "g.safetensors"}, "cpu")


def test_task_error_is_normalized_and_strict_about_non_finite():
    r = torch.tensor([1.0, -4.0, 2.0])
    assert lb.task_error(r.clone(), r, "float32") == (0.0, None)
    e, why = lb.task_error(r + torch.tensor([0.0, 0.04, 0.0]), r, "float32")
    assert why is None and math.isclose(e, 0.01, rel_tol=1e-6)
    e, why = lb.task_error(torch.tensor([1.0, float("nan"), 2.0]), r, "float32")
    assert math.isinf(e) and "non-finite" in why
    rn = torch.tensor([1.0, float("nan"), float("inf")])
    assert lb.task_error(rn.clone(), rn, "float32") == (0.0, None)
    e, why = lb.task_error(torch.tensor([1.0, 0.0, float("inf")]), rn, "float32")
    assert math.isinf(e)
    e, why = lb.task_error(torch.tensor([1, 2]), torch.tensor([1, 3]), "int64")
    assert math.isinf(e) and "exact" in why
    e, why = lb.task_error(torch.zeros(3, 2), torch.zeros(2, 3), "float32")
    assert "shape" in why


def test_the_tolerance_never_falls_below_the_dtype_floor(tmp_path):
    assert lb.task_tolerance(_problem(tmp_path)) == {"y": lb.TASK_FLOOR["float32"]}
    assert lb.task_tolerance(_problem(tmp_path, tolerance={"y": 1e-3})) == {"y": 1e-3}
    assert lb.task_tolerance(_problem(tmp_path, tolerance={"y": 0.0})) == {"y": lb.TASK_FLOOR["float32"]}


def test_correctness_runs_every_group_and_returns_them_for_timing(tmp_path):
    p = _problem(tmp_path, tolerance={"y": 1e-5})
    verdict, sets = lb.task_correctness(p, _rms, False, "cpu")
    assert verdict["pass"], verdict
    assert len(sets) == 2
    assert {c["group"] for c in verdict["checks"]} == {"plain", "spiky"}


def test_correctness_fails_past_the_tolerance(tmp_path):
    p = _problem(tmp_path, tolerance={"y": 1e-5})
    verdict, _ = lb.task_correctness(p, lambda x, w: _rms(x, w) * 1.01, False, "cpu")
    assert not verdict["pass"] and verdict["status"] == "INCORRECT_NUMERICAL"
    assert "input group plain output y" in verdict["error"]


def test_a_broken_input_generator_is_not_charged_to_the_kernel(tmp_path):
    p = _problem(tmp_path)
    p["definition"]["reference"] = JUDGE.replace('"w": torch.ones', '"v": torch.ones')
    verdict, _ = lb.task_correctness(p, _rms, False, "cpu")
    assert verdict["status"] == "INPUT_ERROR" and "no input 'w'" in verdict["error"]


def test_every_shape_is_a_view_with_its_own_groups(tmp_path):
    p = _problem(tmp_path, shapes=(("small", 4, 16), ("large", 64, 32)))
    views = lb._task_shapes(p)
    assert [v["_wl_name"] for v in views] == ["small", "large"]
    assert [len(lb.task_groups(v)) for v in views] == [2, 2]
    one = lb._task_shapes(_problem(tmp_path))
    assert [v["_wl_name"] for v in one] == ["small"]


def _calibration(capsys) -> dict:
    out = capsys.readouterr().out
    obj, _ = json.JSONDecoder().raw_decode(out.split("=== JSON ===", 1)[1].strip())
    return obj["calibration"]


def test_calibration_turns_the_reference_error_into_the_tolerance(tmp_path, capsys):
    p = _problem(tmp_path, shapes=(("small", 4, 16), ("large", 64, 32)))
    noisy = lambda x, w: _rms(x, w) * (1 + 3e-4)  # noqa: E731
    assert lb._task_calibrate(p, noisy, False, "cpu") == 0
    cal = _calibration(capsys)
    assert cal["match"] is True and not cal["problems"]
    assert math.isclose(cal["max_error"]["y"], 3e-4, rel_tol=0.05)
    assert cal["margin"] == lb.TASK_TOL_MARGIN == 2.0
    assert math.isclose(cal["tolerance"]["y"], 2 * 3e-4, rel_tol=0.05)
    assert {(r["shape"], r["group"]) for r in cal["rows"]} == {
        ("small", "plain"), ("small", "spiky"), ("large", "plain"), ("large", "spiky")}


def test_a_kernel_a_little_further_than_the_reference_still_passes(tmp_path, capsys):
    # Another correct summation order lands a little above the reference's own error;
    # the calibrated tolerance must not turn that into a failure.
    p = _problem(tmp_path)
    lb._task_calibrate(p, lambda x, w: _rms(x, w) * (1 + 3e-4), False, "cpu")
    tol = _calibration(capsys)["tolerance"]
    q = _problem(tmp_path, tolerance=tol)
    result, _ = lb.task_correctness(q, lambda x, w: _rms(x, w) * (1 + 4.5e-4), False, "cpu")
    assert result["pass"] is True
    result, _ = lb.task_correctness(q, lambda x, w: _rms(x, w) * (1 + 1e-3), False, "cpu")
    assert result["pass"] is False


def test_calibration_refuses_a_reference_that_computes_something_else(tmp_path, capsys):
    p = _problem(tmp_path)
    lb._task_calibrate(p, lambda x, w: x, False, "cpu")
    cal = _calibration(capsys)
    assert cal["match"] is False
    assert any("the judge and the reference disagree" in m for m in cal["problems"])


def test_calibration_of_an_exact_reference_lands_on_the_floor(tmp_path, capsys):
    p = _problem(tmp_path)
    judge, _ = lb._task_custom(p)
    lb._task_calibrate(p, judge, False, "cpu")
    cal = _calibration(capsys)
    assert cal["match"] and cal["tolerance"]["y"] == lb.TASK_FLOOR["float32"]


# --- [target] l2 = "cold": the rotation, and whether it got there -----------------------------

class _Timing:
    """Stands in for eval/timing.py (CUDA): keeps the callables it was handed."""

    def __init__(self):
        self.work = None

    def measure_windows_workloads(self, work, burn_in_s=None):
        self.work = work
        return {"n": len(work)}

    def workloads_performance(self, result):
        return {"workloads": [{"device_median_ms": 1.0, "ref_device_ms": 2.0}] * result["n"]}


def _cold_side(tmp_path, monkeypatch, l2_bytes, mode="cold"):
    import sys
    import types
    timing = _Timing()
    monkeypatch.setitem(sys.modules, "timing", timing)
    monkeypatch.setattr(lb.torch.cuda, "get_device_properties",
                        lambda device: types.SimpleNamespace(L2_cache_size=l2_bytes))
    problem = _problem(tmp_path)
    if mode is not None:
        problem["l2"] = mode
    sets = [[torch.zeros(4, 16), torch.ones(16)] for _ in GROUPS]      # 320 B per input set
    block = lb._task_side(problem, sets, _rms, False, _rms, False, device="cpu")
    return block["leaderboard"]["protocol"], timing.work


def test_cold_timing_rotates_until_one_sweep_exceeds_twice_the_l2(tmp_path, monkeypatch):
    proto, work = _cold_side(tmp_path, monkeypatch, l2_bytes=64 * 1024)
    assert proto["l2"] == "cold" and proto["l2_reached"] is True
    # 320 B inputs against a 64 KiB L2: 410 copies -- far over the old cap of 64, which left
    # the sweep at 20 KiB (all of it inside L2) under a protocol that still said "cold"
    assert [r["copies"] for r in proto["l2_rotation"]] == [410, 410]
    assert all(r["copies"] * r["input_bytes"] >= 2 * r["l2_bytes"] for r in proto["l2_rotation"])
    for name, kernel_fns, ref_fns in work:
        assert len(kernel_fns) == len(ref_fns) == 410            # one callable per copy, both sides


def test_cold_timing_says_so_when_the_sweep_cannot_reach_twice_the_l2(tmp_path, monkeypatch, capsys):
    proto, work = _cold_side(tmp_path, monkeypatch, l2_bytes=1 << 20)   # needs 6554 copies > the cap
    assert proto["l2"] == "cold" and proto["l2_reached"] is False
    assert [r["copies"] for r in proto["l2_rotation"]] == [lb.L2_COLD_MAX_COPIES] * 2
    err = capsys.readouterr().err
    assert err.count('WARNING l2 = "cold" not reached') == 2 and "this reading is L2-warm" in err
    # a device that reports no L2 size: one copy, and the same warning instead of silence
    proto, work = _cold_side(tmp_path, monkeypatch, l2_bytes=0)
    assert proto["l2_reached"] is False and len(work[0][1]) == 1
    assert "the device reports no L2 size" in capsys.readouterr().err


def test_warm_timing_is_one_input_set_and_carries_no_rotation(tmp_path, monkeypatch, capsys):
    proto, work = _cold_side(tmp_path, monkeypatch, l2_bytes=64 * 1024, mode=None)
    assert proto["l2"] == "warm" and "l2_reached" not in proto and "l2_rotation" not in proto
    assert all(len(k) == 1 for _, k, _ in work) and capsys.readouterr().err == ""
