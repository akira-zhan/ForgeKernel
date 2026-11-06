"""SM120 (sm_120a, RTX Blackwell) pages through the SASS / orchestration / stall tools.

sm_120a has neither wgmma nor tcgen05: its tensor cores are driven by warp-level
``mma.sync``, which ptxas spells HMMA (f16/bf16), QMMA (kind::f8f6f4, plus
``QMMA.SF`` for the mxf8f6f4 block-scaled form) and OMMA (``OMMA.SF``, mxf4nvf4).
Before these families were taught to the tools, a cp.async-fed FP8 GEMM failed
``sass_tools``' tensor-kernel self-check, its mainloop was labelled copy-producer,
its MMA stalls were costed as elementwise and the operand census read FAIL.

Fixtures (``tests/fixtures/sm120/``):
- ``cutlass_sm120_cpasync_fp8_gemm.sass`` / ``cutlass_sm120_tma_ws_fp8_gemm.sass``:
  two public CUTLASS FP8 GEMMs compiled for sm_120a (a cp.async multistage
  mainloop and an SM120 TMA warp-specialized mainloop), encodings stripped;
- ``qmma_*`` / ``omma_*``: the four minimal k-loops of ``sm120_mma_fixture.cu``
  (FP8 f32-acc, FP8 f16-acc, MXF8 block-scaled, NVF4 block-scaled).
"""
from pathlib import Path

import pytest

from forgekernel.loop.driver import _anchor_shares
from forgekernel.skills import orch_tools as OT
from forgekernel.skills import sass_census as SC
from forgekernel.skills import sass_tools
from forgekernel.skills import sched_live_tools as SL
from forgekernel.skills import stall_tools
from forgekernel.skills import struct_tools as ST
from forgekernel.skills.sass_tools import parse_sass

FIX = Path(__file__).parent / "fixtures" / "sm120"
CPASYNC = "cutlass_sm120_cpasync_fp8_gemm"
TMA_WS = "cutlass_sm120_tma_ws_fp8_gemm"


def _page(name: str) -> str:
    return (FIX / f"{name}.sass").read_text()


def _instrs(name: str):
    return list(parse_sass(_page(name)).instructions)


def _first(name: str, family: str):
    return next(i for i in _instrs(name) if i.opcode.split(".", 1)[0] == family)


@pytest.mark.parametrize("name", [CPASYNC, TMA_WS])
def test_sm120_gemm_passes_the_tensor_kernel_self_check(name, capsys):
    """No HGMMA / WARPGROUP on the page (the cp.async one has no TMA either): the
    family check must still recognise it as the tensor kernel."""
    assert sass_tools.analyze(_page(name), name, [(name, 1.0)])
    out = capsys.readouterr().out
    assert f"SASS-OK kernel={name}" in out
    assert "QMMA" in out.split("decisive families:")[1].splitlines()[0]


@pytest.mark.parametrize("name,n", [(CPASYNC, 64), (TMA_WS, 128)])
def test_sm120_mma_census_picks_the_mma_sync_grammar(name, n):
    c = SL.mma_isa_census(_instrs(name))
    assert (c.wgmma, c.tcgen05, c.mma_sync) == (0, 0, n)
    assert c.mode == "mma_sync" and c.fams["QMMA"] == n


@pytest.mark.parametrize("name", [CPASYNC, TMA_WS])
def test_sm120_sched_and_live_verdicts_open(name):
    lines = []
    ok_sched, ok_live = SL.report(_page(name), out=lines.append)
    text = "\n".join(lines)
    assert ok_sched and ok_live
    assert "isa=mma_sync" in text and "QMMA.16832.F32.E4M3.E4M3" in text


def test_qmma_register_widths():
    assert SL._mma_sync_widths("QMMA.16832.F32.E4M3.E4M3") == (4, 4, 2)
    # f16 accumulator packs two per register, as for HMMA
    assert SL._mma_sync_widths("QMMA.16832.F16.E4M3.E4M3") == (2, 4, 2)
    assert SL._mma_sync_widths("OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X") == (4, 4, 2)


def test_block_scaled_issue_uses_its_scale_factor_registers():
    """``QMMA.SF Rd, Ra, Rb, Rc, Rsfa, Rsfb, URZ``: the two scale registers are live
    into the issue; dropping them under-counts the liveness peak."""
    for name, prefix in (("qmma_mxf8_blockscaled_loop", "QMMA.SF.16832"),
                         ("omma_nvf4_blockscaled_loop", "OMMA.SF.16864")):
        ins = _first(name, prefix.split(".", 1)[0])
        assert ins.opcode.startswith(prefix)
        defs, uses, _ = SL.reg_accesses(ins)
        ops = [o.strip() for o in ins.operands.rstrip(" ;").split(",")]
        d, a, b, c, sfa, sfb = (int(o[1:]) for o in ops[:6])
        assert defs == set(range(d, d + 4))
        assert set(range(a, a + 4)) | set(range(b, b + 2)) | {sfa, sfb} <= uses
        assert ops[6] == "URZ"


@pytest.mark.parametrize("opcode,types", [
    ("QMMA.16832.F32.E4M3.E4M3", ("e4m3", "f32")),
    ("QMMA.16832.F16.E4M3.E4M3", ("e4m3", "f16")),
    ("QMMA.SF.16832.F32.E4M3.E4M3.E8", ("e4m3", "f32")),
    ("OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X", ("e2m1", "f32")),
])
def test_sm120_mma_operand_types(opcode, types):
    assert OT.mma_operand_types(opcode) == types


def test_struct_tools_reads_sm120_input_types():
    assert ST.mma_input_type("QMMA.16832.F32.E4M3.E4M3") == "e4m3"
    assert ST.mma_input_type("OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X") == "e2m1"


def test_precision_gate_sees_the_fp8_mainloop():
    assert OT.mma_precision(_instrs(CPASYNC)) == {
        "ops": {"QMMA:e4m3/f32": 64}, "inputs": ["e4m3"], "accums": ["f32"]}


def test_cpasync_mainloop_is_the_mma_consumer():
    """The cp.async mainloop issues LDGSTS and QMMA in one loop; it was labelled
    copy-producer while only *GMMA counted as an MMA issue."""
    labels = [r.label for r in OT.find_regions(_instrs(CPASYNC)) if r.kind == "loop"]
    assert labels == ["mma-consumer"]


def test_operand_census_skips_rather_than_fails_on_mma_sync():
    line = SC.render_census_line(SC.census_from_text(_page(CPASYNC)), CPASYNC)
    assert line == (f"SASS-OPERANDS-SKIP kernel={CPASYNC} "
                    "reason=mma-sync-operands-are-registers mma_sync=64")
    assert SC.parse_census_line(line) == {}
    # a page with no MMA of any kind still fails loudly
    assert SC.render_census_line(SC.census_from_text("/*0000*/ EXIT ;\n"), "k").startswith(
        "SASS-OPERANDS-FAIL")


@pytest.mark.parametrize("family,kind", [("QMMA", "MMA"), ("LDGSTS", "CPASYNC"),
                                         ("DEPBAR", "CPWAIT")])
def test_stall_segments_do_not_cost_sm120_mainloop_as_elementwise(family, kind):
    instrs = _instrs(CPASYNC)
    i = next(n for n, ins in enumerate(instrs) if ins.opcode.split(".", 1)[0] == family)
    assert stall_tools._kind_of(instrs[i], {}, i, SL)[0] == kind


def test_mma_sync_segments_anchor_like_wgmma_runs():
    seg = ("SEG idx=0 kind=CPWAIT depth=none pct=10.0 stalls=long_sb:90\n"
           "SEG idx=1 kind=MMA depth=none pct=40.0 stalls=math:60\n"
           "SEG idx=2 kind=EW depth=none pct=20.0 stalls=wait:50\n"
           "SEG idx=3 kind=MMA depth=none pct=30.0 stalls=math:70\n")
    shares = _anchor_shares(seg)
    assert shares["loop-top wait"][0] == 10.0
    assert shares["first HG run"][0] == 40.0
    assert shares["HG rest"][0] == 30.0
    assert shares["EW@none"][0] == 20.0


def test_struct_reads_a_named_barrier_qmma_construction():
    """``qmma_named_barrier_phases`` (sm120_mma_fixture.cu) profiled on an RTX PRO 5000:
    four warps fenced by ``bar.sync 1, 128``; one phase QMMAs and publishes bf16 pairs
    (F2FP + STS), the next reads a neighbour's slice back (LDS). The source-page CSV
    is trimmed to Address / Source / Instructions Executed."""
    csv_text = (FIX / "qmma_named_barrier_phases.source.csv").read_text()
    sass, counts = ST.source_csv_to_sass_and_counts(csv_text)
    assert len(counts) == 608
    lines = []
    assert ST.report(sass, exec_counts=counts, out=lines.append)
    text = "\n".join(lines)
    assert "BAR.SYNC 0x1,0x80 -> 4 warps" in text
    assert "QMMA.16832.F32.E4M3.E4M3 x1@4w" in text and "STS x1@4w" in text
    assert "LDS x4@4w" in text
    assert any(line.startswith("STRUCT-OK") for line in lines)


# --- compact mma.sync mainloops (cp.async multistage, CuTe DSL, sm_120a) ------------------------

PIPE2 = "cpasync_pipeline_fp8_2stage"
PIPE3 = "cpasync_pipeline_fp8_3stage"


def test_short_mma_sync_page_is_not_below_the_floor():
    """A cp.async -> ldmatrix -> QMMA tile at two CTAs per SM is ~350 instructions: under the
    sm90 floor of 500, yet the tensor kernel. A short page with no MMA family stays rejected."""
    instrs = _instrs(PIPE2)
    assert len(instrs) < sass_tools.MIN_INSTRUCTIONS
    assert not sass_tools.below_floor(instrs)
    bystander = [i for i in instrs if i.opcode.split(".", 1)[0] not in sass_tools.SM120_MMA_SYNC_FAMILIES]
    assert sass_tools.below_floor(bystander)
    assert sass_tools.below_floor(instrs[:sass_tools.MIN_MMA_SYNC_INSTRUCTIONS - 2]
                                  + [i for i in instrs if i.opcode.startswith("QMMA")][:1])   # a stub
    assert OT.summarize(_page(PIPE2)) is not None
    ok_sched, ok_live = SL.report(_page(PIPE2), out=lambda *_: None)
    assert ok_sched and ok_live


def test_commit_axis_reads_the_cp_async_wait_depth():
    """2 vs 3 stages differ only in cp.async.wait_group N -> DEPBAR.LE SB0, N."""
    c2, c3 = OT.summarize(_page(PIPE2))["commit"], OT.summarize(_page(PIPE3))["commit"]
    assert c2 == ["mma-consumer: DEPBAR {} LDGDEPBAR x1 CPWAIT {LE 0x0 x1}"]
    assert c3 == ["mma-consumer: DEPBAR {} LDGDEPBAR x1 CPWAIT {LE 0x1 x1}"]
    assert OT.summarize(_page(CPASYNC))["commit"] == ["mma-consumer: DEPBAR {} LDGDEPBAR x1 CPWAIT {LE 0x2 x1}"]


def test_pipeline_loop_is_one_mma_consumer_with_a_cache_op_anchor():
    d = OT.summarize(_page(PIPE3))
    assert d["loops"] == {"mma-consumer": 1}
    assert d["cache_ops"] == ["LDGSTS.E.BYPASS.LTC128B.128"]


# --- TMA + mbarrier mainloops (PipelineTmaAsync, CuTe DSL, sm_120a) --------------------------------

def test_mbarrier_addresses_resolve_through_the_lea_addend():
    """``ULEA UR4, UR4, UR5, 0x18`` = (SR_CgaCtaId << 24) + UR5: the offset is the ADDEND. Chasing the
    destination-named operand followed SR_CgaCtaId and left every mbarrier of this page UNRESOLVED
    (mbar_objs 0); following the addend resolves full + empty for each of the 3 stages."""
    for name in ("tma_mbarrier_fp8_ws", "tma_mbarrier_fp8_inline"):
        d = OT.summarize(_page(name))
        assert d["mbar_objs"] == 6 and d["mbar_tiers"] == {"CONST": 6}, name
    edges = [tuple(e) for e in OT.summarize(_page("tma_mbarrier_fp8_ws"))["edges"]]
    assert ("copy-producer", "mma-consumer") in edges and ("mma-consumer", "copy-producer") in edges


def test_tma_mainloop_cache_op_anchor_is_the_tma_load():
    for name in ("tma_mbarrier_fp8_ws", "tma_mbarrier_fp8_inline"):
        assert OT.summarize(_page(name))["cache_ops"] == ["UTMALDG.2D"]
