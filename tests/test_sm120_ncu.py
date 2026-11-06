"""sm_120 NCU extras: the cp.async (LDGSTS) shared-memory fill counters.

On GB202 an LDGSTS fill is invisible to every smem counter in ``KEY_METRICS`` and to the
hardware ``l1tex__..._op_ldgsts`` counters; only the SASS-patched counters see it, and a
conflicting fill shows there as more wavefronts per LDGSTS (measured on an RTX PRO 5000,
``probes/sm120/mechanisms/warp_mma_fp8_ldsm_check.py``). They cost NCU an extra replay
pass, so gpu_entry.sh requests them only when the card's CUTE_DSL_ARCH is sm_120*.
"""
import re
import subprocess

from forgekernel import leaderboard as lb
from forgekernel.inner.ncu import (ARCH_EXTRA_METRICS, KEY_METRICS, NCU_METRICS_ARG, key_metrics,
                                   parse_ncu_csv, render_ncu_trace)

SM120 = ARCH_EXTRA_METRICS["sm_120"]


def _csv(rows):
    head = '"ID","Kernel Name","Metric Name","Metric Unit","Metric Value"\n'
    return head + "".join(f'"0","k","{n}","{u}","{v}"\n' for n, u, v in rows)


def test_extras_are_parsed_but_not_requested_by_default():
    assert set(SM120) == {"sm__sass_l1tex_data_pipe_lsu_wavefronts_mem_shared_op_ldgsts.sum",
                          "sm__sass_inst_executed_op_ldgsts.sum"}
    # the default request (every arch) is unchanged
    assert not set(SM120) & set(KEY_METRICS)
    assert not set(SM120) & set(NCU_METRICS_ARG.split(","))
    # a GB202 profile of the conflicting-fill tile: 4096 fill wavefronts over 128 LDGSTS
    d = key_metrics(parse_ncu_csv(_csv([
        ("gpu__time_active.sum", "ns", "11,488"),
        ("l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum", "", "0"),
        ("sm__sass_l1tex_data_pipe_lsu_wavefronts_mem_shared_op_ldgsts.sum", "", "4,096"),
        ("sm__sass_inst_executed_op_ldgsts.sum", "inst", "128")])))
    assert d["smem_wavefronts_st"] == 0
    assert d["smem_wavefronts_ldgsts"] == 4096 and d["ldgsts_inst"] == 128
    assert d["smem_wavefronts_ldgsts"] / d["ldgsts_inst"] == 32      # 4 would be conflict-free


def test_a_profile_without_the_extras_has_no_such_keys():
    d = key_metrics(parse_ncu_csv(_csv([("gpu__time_active.sum", "ns", "1,000")])))
    assert "smem_wavefronts_ldgsts" not in d and "ldgsts_inst" not in d


def test_gpu_entry_appends_exactly_the_sm120_extras():
    """The gpu_entry.sh case block names the same metrics as ARCH_EXTRA_METRICS["sm_120"]."""
    block = re.search(r'case "\$CUTE_DSL_ARCH" in\n(.*?)\nesac', lb.GPU_ENTRY_TEMPLATE, re.S).group(1)
    m = re.search(r'sm_120\*\) NCU_METRICS="\$NCU_METRICS,([^"]+)"', block)
    assert m and set(m.group(1).split(",")) == set(SM120)


def test_gpu_entry_case_block_only_fires_on_sm120():
    """Run the template's NCU_METRICS lines in bash under three CUTE_DSL_ARCH values."""
    text = lb.GPU_ENTRY_TEMPLATE
    start = text.index('NCU_METRICS="')
    end = text.index("esac", start) + len("esac")
    snippet = text[start:end].replace("{{", "{").replace("}}", "}") + '\necho "$NCU_METRICS"\n'
    seen = {}
    for arch in ("sm_120a", "sm_100a", "sm_90a"):
        out = subprocess.run(["bash", "-c", snippet], env={"CUTE_DSL_ARCH": arch, "PATH": "/usr/bin:/bin"},
                             capture_output=True, text=True, check=True).stdout.strip().split(",")
        seen[arch] = out
    # the default part is identical for every arch; only sm_120 appends, and only the extras
    assert seen["sm_100a"] == seen["sm_90a"]
    assert seen["sm_120a"][:len(seen["sm_100a"])] == seen["sm_100a"]
    assert set(seen["sm_120a"][len(seen["sm_100a"]):]) == set(SM120)
    assert not set(SM120) & set(seen["sm_100a"])


def test_gpu_entry_default_list_is_exactly_key_metrics():
    """gpu_entry.sh's default NCU_METRICS (every arch) requests exactly what KEY_METRICS parses;
    it used to miss the two smem wavefront metrics, so smem_wavefronts_st/ld were never collected."""
    text = lb.GPU_ENTRY_TEMPLATE
    default = re.search(r'\nNCU_METRICS="([^"$]+)"', text).group(1).split(",")
    assert len(default) == len(set(default))
    assert set(default) == set(KEY_METRICS)


def test_trace_shows_the_fill_columns():
    from forgekernel.inner.ncu import _TRACE_COLUMNS
    assert {"smem_wavefronts_ldgsts", "ldgsts_inst"} <= set(_TRACE_COLUMNS)
    assert callable(render_ncu_trace)
