"""sm_120 NCU extras: the cp.async (LDGSTS) shared-memory fill counters and the sleeping stall.

On GB202 an LDGSTS fill is invisible to every smem counter in ``KEY_METRICS`` and to the
hardware ``l1tex__..._op_ldgsts`` counters; only the SASS-patched counters see it, and a
conflicting fill shows there as more wavefronts per LDGSTS (measured on an RTX PRO 5000,
``probes/sm120/mechanisms/warp_mma_fp8_ldsm_check.py``). They cost NCU an extra replay
pass, so gpu_entry.sh requests them only when the card's CUTE_DSL_ARCH is sm_120*.

The sleeping stall is where GB202 counts an mbarrier / TMA wait: the delivered FP8 decode GEMM
(TMA + mbarrier supply) reads 20-21 there against 3.5 long-scoreboard and 1.9 barrier, and the
stall ledgers of the sm120 runs put 97-100 % of the mbarrier.trywait samples in ``sleeping``.
"""
import re
import subprocess

from forgekernel import leaderboard as lb
from forgekernel.inner.ncu import (ARCH_EXTRA_METRICS, KEY_METRICS, NCU_METRICS_ARG, arch_key,
                                   key_metrics, parse_ncu_csv, render_ncu_trace, summarize_metrics)

SM120 = ARCH_EXTRA_METRICS["sm_120"]


def _csv(rows):
    head = '"ID","Kernel Name","Metric Name","Metric Unit","Metric Value"\n'
    return head + "".join(f'"0","k","{n}","{u}","{v}"\n' for n, u, v in rows)


def test_extras_are_parsed_but_not_requested_by_default():
    assert set(SM120) == {"sm__sass_l1tex_data_pipe_lsu_wavefronts_mem_shared_op_ldgsts.sum",
                          "sm__sass_inst_executed_op_ldgsts.sum",
                          "smsp__average_warps_issue_stalled_sleeping_per_issue_active.ratio"}
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


# --- the summary read on sm_120 ---------------------------------------------------------
# Readings of the sm120 FP8 decode GEMM runs (RTX PRO 5000): the TMA + mbarrier student at
# 68 % and at 77 % of peak DRAM, tensor pipe 7-8 %, 12 warps per SM (17 % of 64).
DECODE_68 = {"tensor_pct": 7.2, "warp_active_pct": 15.0, "dram_pct": 68.2, "stall_long_sb": 2.9,
             "stall_barrier": 0.4, "stall_mio": 0.0, "smem_wavefronts_ldgsts": 0, "ldgsts_inst": 0}
DECODE_77 = {"tensor_pct": 8.4, "warp_active_pct": 17.4, "dram_pct": 77.1, "stall_long_sb": 3.5,
             "stall_barrier": 1.9, "stall_mio": 0.0, "smem_wavefronts_ldgsts": 0, "ldgsts_inst": 0}


def _flag(s, key):
    return next((f for f in s["flags"] if f["key"] == key), None)


def test_arch_key_spellings():
    assert arch_key("sm120") == arch_key("sm_120a") == arch_key("sm_120") == "sm_120"
    assert arch_key("sm90") == "sm_90" and arch_key(None) is None and arch_key("b200") is None


def test_mbarrier_wait_is_read_as_supply_wait_not_a_starved_tensor_core():
    """A DRAM-streaming decode GEMM whose warps sleep on mbarrier waits is supply-bound; the
    tensor pipe is idle because of it. With DRAM under its flag level the verdict names the
    wait and quotes the DRAM headroom."""
    s = summarize_metrics({**DECODE_68, "stall_sleeping": 20.1}, arch="sm120")
    assert s["verdict"].startswith("supply-wait bound") and "68.2 % of peak" in s["verdict"]
    assert _flag(s, "stall_sleeping")["severity"] == "severe"
    assert "mbarrier" in _flag(s, "stall_sleeping")["note"]
    # the same kernel once DRAM is at its flag level: bandwidth is the bound
    assert summarize_metrics({**DECODE_77, "stall_sleeping": 20.1}, arch="sm120")["verdict"] \
        == "memory-bandwidth bound"
    # a record taken before the metric was requested keeps its old read
    assert summarize_metrics(DECODE_68)["verdict"] == "latency bound (tensor core starved)"
    # a compute-bound TMA kernel also sleeps a little on its barriers: tensor pipe busy, no flag
    busy = summarize_metrics({"tensor_pct": 86.2, "warp_active_pct": 18.2, "dram_pct": 16.9,
                              "stall_sleeping": 6.0}, arch="sm120")
    assert not busy["verdict"].startswith("supply-wait")


def test_cp_async_fill_conflict_is_flagged_from_wavefronts_per_ldgsts():
    clean = summarize_metrics({"smem_wavefronts_ldgsts": 512, "ldgsts_inst": 128})
    assert _flag(clean, "ldgsts_fill_wavefronts") is None           # 4 per LDGSTS: conflict-free
    bad = summarize_metrics({"smem_wavefronts_ldgsts": 4096, "ldgsts_inst": 128})
    f = _flag(bad, "ldgsts_fill_wavefronts")
    assert f["value"] == 32 and f["severity"] == "severe"
    assert bad["verdict"].startswith("cp.async fill-conflict bound")
    mild = summarize_metrics({"smem_wavefronts_ldgsts": 640, "ldgsts_inst": 128})
    assert _flag(mild, "ldgsts_fill_wavefronts")["severity"] == "elevated"
    # a TMA-fed kernel executes no LDGSTS: nothing to divide, nothing flagged
    assert _flag(summarize_metrics(DECODE_77), "ldgsts_fill_wavefronts") is None


def test_notes_do_not_name_wgmma_on_sm120():
    m = {"stall_barrier": 25, "stall_long_sb": 25, "tensor_pct": 5, "warp_active_pct": 5}
    sm120 = {f["key"]: f["note"] for f in summarize_metrics(m, arch="sm_120a")["flags"]}
    assert not any("WGMMA" in n or "wgmma wait_group (" in n for n in sm120.values())
    assert "stall_sleeping" in sm120["stall_long_sb"] and "bar.sync" in sm120["stall_barrier"]
    # a profile that carries the sm_120 extras is read as sm_120 without being told
    inferred = {f["key"]: f["note"] for f in summarize_metrics({**m, "ldgsts_inst": 0})["flags"]}
    assert inferred == sm120
    # every other architecture keeps the wording it had
    other = {f["key"]: f["note"] for f in summarize_metrics(m, arch="sm90")["flags"]}
    assert "WGMMA" in other["stall_barrier"] and "NANOSLEEP" in other["stall_long_sb"]
