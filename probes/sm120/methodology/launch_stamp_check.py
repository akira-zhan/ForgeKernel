#!/usr/bin/env python3
"""Re-earn ``launch_identity_stamp`` on an sm_120a device (CuTe DSL).

    python3 -u launch_stamp_check.py            # parent: one process per stage
    python3 -u launch_stamp_check.py <stage>    # child

Two compiled instances of one parameterised stub kernel (variant id, stage constant, grid shape,
cluster direction, dynamic smem bytes) launched V1 -> V2 -> V1 again from a dispatch cache. Each
CTA's thread 0 writes one stamp row: trace-time constants + launch special registers.

Contracts:
- ``stamp_V1`` / ``stamp_V2`` / ``stamp_V1_again``: every launched CTA's row equals the expectation
  table field by field (max_diff = 0), rows of CTAs that do not exist stay at the -1 sentinel;
- ``dyn_smem``      : ``%dynamic_smem_size`` is the same on every CTA and pad <= reading <= pad + 1024;
- ``cluster_shape`` : ``block_in_cluster_dim()`` reads the cluster shape in CTAs and ``cluster_dim()``
  reads the number of clusters in the grid (the two are not interchangeable);
- ``gate_catches``  : V2's stamp checked against V1's expectation table is a mismatch;
- ``stamp_off``     : with the ``const_expr`` switch off the kernel writes nothing (buffer stays -1);
- ``smem_limit``    : 101376 B of dynamic smem (the sm120 opt-in maximum) launches and reads back exactly;
  1 KiB more still compiles and fails only at launch, with an error that does not name shared memory.
The expectation table is written here, from the launch parameters, not imported from the kernel.
"""
import os
import subprocess
import sys

HERE = os.path.abspath(__file__)
NF = 14
(F_VARIANT, F_STAGES, F_BID, F_GX, F_GY, F_GZ, F_BX, F_CX, F_CY, F_CSZ, F_SMEM, F_NCX, F_NCY, F_MAGIC) = range(NF)
MAXCTA = 64
V1 = dict(variant_id=1, stages=2, grid=(4, 2, 1), cluster=(2, 1, 1), pad_bytes=8192)
V2 = dict(variant_id=2, stages=3, grid=(3, 4, 1), cluster=(1, 2, 1), pad_bytes=96 * 1024)


def _v(tag, ok, md=""):
    print(f"[[V]]{tag}|{int(bool(ok))}|{md}", flush=True)


def _n(text):
    print(f"[[N]]{text}", flush=True)


def build():
    import cutlass
    import cutlass.cute as cute
    from cutlass.memory import SmemAllocator
    from cutlass._mlir.dialects import llvm
    from cutlass.cutlass_dsl import T, dsl_user_op

    I32 = cutlass.Int32

    @dsl_user_op
    def dyn_smem_size(*, loc=None, ip=None):
        return I32(llvm.inline_asm(T.i32(), [], "mov.u32 $0, %dynamic_smem_size;", "=r",
                                   has_side_effects=True, is_align_stack=False,
                                   asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip))

    class Stamped:
        def __init__(self, variant_id, stages, grid, cluster, pad_bytes, stamp=True):
            self.variant_id, self.stages = variant_id, stages
            self.grid, self.cluster, self.pad_bytes, self.stamp = grid, cluster, pad_bytes, stamp

        @cute.jit
        def __call__(self, gStamp: cute.Tensor):
            self.kernel(gStamp).launch(grid=list(self.grid), block=[128, 1, 1], cluster=list(self.cluster))

        @cute.kernel
        def kernel(self, gStamp: cute.Tensor):
            tidx, _, _ = cute.arch.thread_idx()
            bx, by, bz = cute.arch.block_idx()
            gx, gy, gz = cute.arch.grid_dim()               # %nctaid
            nx, _, _ = cute.arch.block_dim()                # %ntid
            cx, cy, _ = cute.arch.block_in_cluster_dim()    # %cluster_nctaid: cluster shape in CTAs
            ncx, ncy, _ = cute.arch.cluster_dim()           # %nclusterid: number of clusters
            csz = cute.arch.cluster_size()                  # %cluster_nctarank
            smem = SmemAllocator()
            pad = smem.allocate_tensor(element_type=cutlass.Int8,
                                       layout=cute.make_layout(self.pad_bytes), byte_alignment=1024)
            if cutlass.const_expr(self.stamp):
                if tidx == 0:
                    pad[0] = cutlass.Int8(1)
                    bid = bx + by * gx + bz * gx * gy
                    gStamp[bid, F_VARIANT] = I32(self.variant_id)
                    gStamp[bid, F_STAGES] = I32(self.stages)
                    gStamp[bid, F_BID] = bid
                    gStamp[bid, F_GX] = gx
                    gStamp[bid, F_GY] = gy
                    gStamp[bid, F_GZ] = gz
                    gStamp[bid, F_BX] = nx
                    gStamp[bid, F_CX] = cx
                    gStamp[bid, F_CY] = cy
                    gStamp[bid, F_CSZ] = csz
                    gStamp[bid, F_SMEM] = dyn_smem_size()
                    gStamp[bid, F_NCX] = ncx
                    gStamp[bid, F_NCY] = ncy
                    gStamp[bid, F_MAGIC] = I32(0x5EED)

    return Stamped


def expected(v):
    """Expectation rows written from the launch parameters (not imported from the kernel)."""
    import torch
    gx, gy, gz = v["grid"]
    cx, cy, cz = v["cluster"]
    n = gx * gy * gz
    rows = torch.full((n, NF), -1, dtype=torch.int32)
    for b in range(n):
        rows[b] = torch.tensor([v["variant_id"], v["stages"], b, gx, gy, gz, 128, cx, cy, cx * cy * cz,
                                -1, gx // cx, gy // cy, 0x5EED], dtype=torch.int32)
    return rows


def check(buf, v):
    """(max_diff over every field but smem, sentinel rows intact, smem readings)."""
    exp = expected(v)
    n = exp.shape[0]
    got = buf[:n].cpu()
    mask = torch_ones_but(F_SMEM)
    diff = (got[:, mask] - exp[:, mask]).abs().max().item()
    tail_ok = bool((buf[n:].cpu() == -1).all())
    return diff, tail_ok, got[:, F_SMEM].tolist()


def torch_ones_but(f):
    return [i for i in range(NF) if i != f]


def stage_stamp():
    import torch
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack
    Stamped = build()
    buf = torch.full((MAXCTA, NF), -1, dtype=torch.int32, device="cuda")
    t = from_dlpack(buf)
    cache = {}

    def run(v):
        key = (v["variant_id"], v["stages"], v["grid"], v["cluster"], v["pad_bytes"])
        fresh = key not in cache
        if fresh:
            cache[key] = cute.compile(Stamped(*key), t)
        buf.fill_(-1)
        cache[key](t)
        torch.cuda.synchronize()
        return fresh

    for tag, v in (("stamp_V1", V1), ("stamp_V2", V2), ("stamp_V1_again", V1)):
        fresh = run(v)
        diff, tail_ok, smem = check(buf, v)
        _n(f"{tag}: compiled_now={fresh} rows={len(smem)} max_diff={diff} sentinel_tail={tail_ok} "
           f"row0={buf[0].tolist()}")
        _v(tag, diff == 0 and tail_ok and (tag != "stamp_V1_again" or not fresh), f"max_diff={diff}")
        if tag == "stamp_V2":
            same = len(set(smem)) == 1
            pad = v["pad_bytes"]
            _n(f"dyn_smem V2: readings={sorted(set(smem))} pad={pad}")
            _v("dyn_smem", same and pad <= smem[0] <= pad + 1024, f"reading={smem[0]} pad={pad}")
            gx, gy, _ = v["grid"]
            cx, cy, _ = v["cluster"]
            r = buf[0].tolist()
            _v("cluster_shape", (r[F_CX], r[F_CY]) == (cx, cy) and (r[F_NCX], r[F_NCY]) == (gx // cx, gy // cy)
               and (r[F_CX], r[F_CY]) != (r[F_NCX], r[F_NCY]),
               f"block_in_cluster_dim=({r[F_CX]},{r[F_CY]}) cluster_dim=({r[F_NCX]},{r[F_NCY]})")
            # gate: V2's stamp against V1's expectation must be a mismatch
            n1 = expected(V1).shape[0]
            got = buf[:n1].cpu()
            mism = (got[:, torch_ones_but(F_SMEM)] - expected(V1)[:, torch_ones_but(F_SMEM)]).abs().max().item()
            _v("gate_catches", mism != 0, f"max_diff_vs_V1={mism}")
        if tag == "stamp_V1":
            _n(f"dyn_smem V1: readings={sorted(set(smem))} pad={v['pad_bytes']}")


def stage_off():
    import torch
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack
    Stamped = build()
    buf = torch.full((MAXCTA, NF), -1, dtype=torch.int32, device="cuda")
    t = from_dlpack(buf)
    f = cute.compile(Stamped(V1["variant_id"], V1["stages"], V1["grid"], V1["cluster"], V1["pad_bytes"],
                             stamp=False), t)
    f(t)
    torch.cuda.synchronize()
    _v("stamp_off", bool((buf == -1).all()), f"untouched={bool((buf == -1).all())}")


def stage_limit():
    """sm120 opt-in limit: 101376 B per block. At the limit the launch works; 1 KiB above it the
    kernel still compiles and the failure only appears at launch, without naming shared memory."""
    import torch
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack
    Stamped = build()
    buf = torch.full((MAXCTA, NF), -1, dtype=torch.int32, device="cuda")
    t = from_dlpack(buf)
    at = cute.compile(Stamped(9, 1, (2, 1, 1), (1, 1, 1), 101376), t)
    at(t)
    torch.cuda.synchronize()
    reading = buf[0, F_SMEM].item()
    _n(f"smem at the opt-in limit: reading={reading}")
    over = cute.compile(Stamped(9, 1, (2, 1, 1), (1, 1, 1), 101376 + 1024), t)  # compiles
    _n("smem 1 KiB over the limit: compiled")
    try:
        over(t)
        torch.cuda.synchronize()
        err = None
    except Exception as e:  # noqa: BLE001 - the failure is the reading
        err = f"{type(e).__name__}: {str(getattr(e, 'message', '') or e).strip().splitlines()[0][:160]}"
    _n(f"smem 1 KiB over the limit: launch -> {err}")
    _v("smem_limit", reading == 101376 and err is not None and "shared" not in err.lower(),
       f"at_limit={reading} over={err}")


STAGES = {"stamp": stage_stamp, "off": stage_off, "limit": stage_limit}
PLAN = [("stamp", ["stamp_V1", "stamp_V2", "dyn_smem", "cluster_shape", "gate_catches", "stamp_V1_again"]),
        ("off", ["stamp_off"]), ("limit", ["smem_limit"])]


def main():
    env = dict(os.environ)
    env.setdefault("CUTE_DSL_ARCH", "sm_120a")
    env["CUTE_DSL_SHOW_STACKTRACE"] = "1"
    verdicts, facts, why = {}, [], []
    for stage, tags in PLAN:
        try:
            p = subprocess.run([sys.executable, "-u", HERE, stage], env=env, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, timeout=300)
            rc, body = p.returncode, p.stdout.decode("utf-8", "replace")
        except subprocess.TimeoutExpired as te:
            rc, body = "timeout", (te.stdout or b"").decode("utf-8", "replace")
        seen = set()
        for ln in body.splitlines():
            if ln.startswith("[[V]]"):
                tag, ok, md = ln[5:].split("|", 2)
                verdicts[tag] = (ok == "1", md); seen.add(tag)
            elif ln.startswith("[[N]]"):
                facts.append(f"{stage}: {ln[5:]}")
        for t in tags:
            if t not in seen:
                verdicts[t] = (False, f"no verdict emitted (stage rc={rc})")
        if any(t not in seen for t in tags):
            why += [f"stage {stage} rc={rc}"] + ["  | " + ln[:200] for ln in body.splitlines()[-12:]]
    print("=== DIGEST ===")
    for f in facts:
        print(f"FACT {f}")
    for ln in why:
        print(ln)
    fails = [t for t, (ok, _) in verdicts.items() if not ok]
    for t, (ok, md) in verdicts.items():
        print(f"PROBE {'PASS' if ok else 'FAIL'} {t} ({md})")
    print(f"PROBE {'PASS' if not fails else 'FAIL'} launch-stamp contracts={len(verdicts)}"
          + (f" failed={','.join(fails)}" if fails else ""))
    return 0 if not fails else 1


if __name__ == "__main__":
    if len(sys.argv) > 1:
        STAGES[sys.argv[1]]()
    else:
        sys.exit(main())
