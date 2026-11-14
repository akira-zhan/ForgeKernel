"""An sm120 task through `forge new` / `forge run`: one architecture from task.toml to the loop.

The sm120 tests elsewhere stop at the tools (SASS pages, NCU rows). Nothing took a task whose
``arch = "sm120"`` through the task card, the job environment and the loop's configuration, so a
place that still assumed the architectures the harness started on could only show up in a real
run: a CUDA kernel file was built for sm_100a on an sm_120a device ("no kernel image is available
for execution on the device"), and a prompt told its reader the smem limit of another chip.

No GPU here. The agent, the calibration job and the loop body are stood in for exactly as in
test_launch.py; the loop's own start-up (configuration, KB tree gate, bindings) is the real one,
run with ``--dry-run``.
"""
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import re
import subprocess
import tomllib

import pytest

from forgekernel import launch, taskcard
from forgekernel.localenv import load_machine
from forgekernel.loop.cli import _kb_arch_problem, _resolve_kb_arch
from forgekernel.loop.kb_lint import lint_kb_dir
from forgekernel.runs import list_runs

from test_launch import REF_TOML, _loop_ends, _run_with_check, _start, _task, _Teacher, repo  # noqa: F401
from test_loop_driver import REPO

SM120_TOML = REF_TOML.replace('arch = "sm100"', 'arch = "sm120"')
SM120_CUDA_TOML = SM120_TOML.replace('language = "cutedsl"', 'language = "cuda"')


def _card_module(op_dir):
    spec = importlib.util.spec_from_file_location("sm120_card_lb", op_dir / "eval" / "leaderboard.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_toml_really_names_sm120():
    assert 'arch = "sm120"' in SM120_TOML and 'language = "cuda"' in SM120_CUDA_TOML


def test_forge_run_takes_an_sm120_task_to_the_loop_on_its_own_architecture(repo, monkeypatch):  # noqa: F811
    from forgekernel.loop import cli as loop_cli
    real_loop_main = loop_cli.main
    seen = {}

    def at_the_loop(argv):
        # the loop's real start-up on the card `forge run` just built, stopping before any agent
        seen["argv"] = argv
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            seen["rc"] = real_loop_main(list(argv) + ["--dry-run"])
        seen["plan"] = buf.getvalue()

    monkeypatch.setattr(loop_cli.RealBackend, "run_agent", _Teacher().method)
    monkeypatch.setattr(launch, "calibrate", lambda task, op_dir: {
        "match": True, "tolerance": {"y": 2e-6}, "problems": []})
    monkeypatch.setattr(loop_cli, "main", _loop_ends(at_the_loop))
    assert _start(_task(repo, toml=SM120_TOML)) == 0
    run = list_runs()[0]
    assert run.record()["status"] == "done"
    op_dir = run.operators_dir / "rmsnorm"

    # the card: the job's architecture, in the entry script and in the job environment
    entry = (op_dir / "gpu_entry.sh").read_text()
    assert 'CUTE_DSL_ARCH:-sm_120a' in entry and "sm_100a" not in entry
    sub = tomllib.loads((op_dir / "submit.toml").read_text())
    assert sub["env"]["CUTE_DSL_ARCH"] == "sm_120a"
    # ... whose NCU request, on this architecture only, adds the cp.async fill counters and the
    # sleeping stall (where GB202 counts an mbarrier wait)
    start = entry.index('NCU_METRICS="')
    snippet = entry[start:entry.index("esac", start) + 4] + '\necho "$NCU_METRICS"\n'
    metrics = subprocess.run(["bash", "-c", snippet], capture_output=True, text=True, check=True,
                             env={"CUTE_DSL_ARCH": sub["env"]["CUTE_DSL_ARCH"], "PATH": "/usr/bin:/bin"}
                             ).stdout.strip().split(",")
    assert "sm__sass_inst_executed_op_ldgsts.sum" in metrics
    assert "smsp__average_warps_issue_stalled_sleeping_per_issue_active.ratio" in metrics
    assert "dram__bytes_op_read.sum" in metrics

    # the loop's configuration: this architecture's KB tree, which exists and is lint-clean
    cfg = tomllib.loads((op_dir / "loop.toml").read_text())
    assert cfg["loop"]["kb_arch"] == "sm120" and cfg["loop"]["language"] == "cutedsl"
    assert _resolve_kb_arch(None, cfg) == "sm120"
    kb = REPO / "hw_kb" / "cutedsl"
    assert _kb_arch_problem(kb, "sm120") is None and lint_kb_dir(kb / "sm120") == []

    # the loop itself, started on that card: reads the sm120 tree and nothing else
    assert seen["rc"] == 0, seen["plan"]
    plan = seen["plan"]
    assert re.search(r"kb_arch\s+= sm120\s+\(reads/writes \S*hw_kb/cutedsl/sm120\)", plan), plan
    assert "MISSING" not in plan and "hw_kb/cutedsl/sm90" not in plan and "hw_kb/cutedsl/sm100" not in plan


def test_an_sm120_cuda_card_builds_its_kernel_for_sm120(repo, monkeypatch):  # noqa: F811
    run, task, check = _run_with_check(repo, toml=SM120_CUDA_TOML)
    machine = load_machine(repo, environ={})
    op_dir = taskcard.write_card(run, task, check, machine)
    assert (op_dir / "kernel.cu").is_file()
    sub = tomllib.loads((op_dir / "submit.toml").read_text())
    assert sub["env"]["CUTE_DSL_ARCH"] == "sm_120a"
    cfg = tomllib.loads(taskcard.write_loop_toml(run, task, op_dir, deny=[]).read_text())["loop"]
    assert cfg["kb_arch"] == "sm120"
    kb = REPO / "hw_kb" / "cuda"
    assert _kb_arch_problem(kb, "sm120") is None and lint_kb_dir(kb / "sm120") == []

    card = _card_module(op_dir)
    src = (op_dir / "kernel.cu").read_text()
    # inside a job: gpu_entry.sh has exported the card's architecture
    monkeypatch.setenv("CUTE_DSL_ARCH", sub["env"]["CUTE_DSL_ARCH"])
    assert card.cuda_kernel_flags(src)[-1] == "-gencode=arch=compute_120a,code=sm_120a"
    # a target the kernel file names itself still wins, and only the `=` spelling is a flag
    own = "// fk-cuda-cflags: -gencode=arch=compute_120a,code=sm_120a -Xptxas=-v\n" + src
    flags = card.cuda_kernel_flags(own)
    assert flags.count("-gencode=arch=compute_120a,code=sm_120a") == 1
    with pytest.raises(ValueError, match="not allowed"):
        card.cuda_kernel_flags("// fk-cuda-cflags: -gencode arch=compute_120a,code=sm_120a\n" + src)
    # outside a job (no architecture exported) and on the boards' cards: the boards' B200
    monkeypatch.delenv("CUTE_DSL_ARCH")
    assert card.cuda_kernel_flags(src)[-1] == card.CUDA_GENCODE == "-gencode=arch=compute_100a,code=sm_100a"
    for arch, want in (("sm_90a", "90a"), ("sm_103a", "103a")):
        assert card.cuda_gencode(arch) == f"-gencode=arch=compute_{want},code=sm_{want}"


def test_the_binary_reference_build_follows_the_cards_architecture(tmp_path, monkeypatch):
    """binref.py compiles a sealed solution inside a job; without a target of its own it must
    build for the card's architecture, as the kernel file does."""
    (tmp_path / "binref.json").write_text(json.dumps({"module": "m", "sources": []}))
    ns = {"__name__": "binref_tool_sm120", "__file__": str(tmp_path / "binref.py")}
    exec(compile((REPO / "src" / "forgekernel" / "binref_tool.py").read_text(), ns["__file__"], "exec"), ns)
    monkeypatch.setenv("CUTE_DSL_ARCH", "sm_120a")
    assert ns["default_gencode"]() == "-gencode=arch=compute_120a,code=sm_120a"
    monkeypatch.delenv("CUTE_DSL_ARCH")
    assert ns["default_gencode"]() == ns["BLACKWELL_GENCODE"]
