# ptx_escape (PTX instructions / modifiers the DSL does not wrap: the `llvm.inline_asm` escape hatch + the expressibility ladder)

> **Certification source: the sm90 campaign · re-checked on sm120 with CuTe DSL 4.8.0** (`probes/sm120/methodology/ptx_escape_check.py`,
> `PROBE PASS ptx-escape contracts=8`): forms 1-3, rung A, the footprint check (with its 4.8.0 switches) and gotchas 2, 4, 6;
> gotcha 5 (swizzled smem addresses) is **not** re-checked on sm120.
>
> One file per hardware interface. Operator-agnostic; mechanism only.
> This entry is a **general rule**: any instruction or modifier that "appears in the target SASS
> footprint but cannot be found in the CuTe DSL op's parameter list" is handled with the ladder
> here. Certified concrete use cases of the sm90 tree (`sm90/s2g_bulk_reduce`, `sm90/cp_async`,
> `sm90/cfrag_elementwise_fastpath`, `sm90/cfrag_fused_pack_liveness`) are not in this tree;
> the calling convention and the ladder are defined here.
> Names of the form `sm90/xxx` / `sm100/xxx` are entries of other trees and **are not in the read set of an sm120 run**;
> they are named so that you know they exist; their readings were taken on other hardware and DSL versions.

## Purpose

A CuTe DSL op object exposes only part of the PTX parameter surface. When an instruction (`cp.reduce.async.bulk`),
a modifier (`.L2::256B` on `cp.async`), or an intrinsic not exported by a given DSL version
(`shfl.sync.idx`) is not among the op's constructor parameters, `cutlass._mlir.dialects.llvm.inline_asm`
can emit the PTX directly inside a `@dsl_user_op`, in the same kernel as the DSL-generated code,
sharing registers and smem addresses with it.

## When needed (trigger) — the expressibility ladder

Symptom: one axis of the dossier diff (`cache_ops` / `commit` / `supply` / `nbar_verdicts`) is `!`,
and the difference is a **modifier or instruction variant** (same instruction family, different decoration), not a structural difference.

Such differences are **numerically invisible**: the result is the same with or without the modifier, and correctness always passes. So "the DSL may not be able to express it"
must not be written as a conclusion; **each rung must get a verdict**, each with evidence, and **you must not jump to C**:

| Rung | Question | Form of evidence | Verdict line |
|---|---|---|---|
| **A** DSL knob | Does the op's constructor have this parameter? Does the enum have this member? | the **verbatim print** of `inspect.signature(Op.__init__)` + `dir(module)` inside a job (names written from memory do not count) | `EXPR A=<spelling>` or `EXPR A=none (<signature text>)` |
| **B** inline PTX | Emit this PTX through this entry's hatch: does the footprint appear? Are the numerics still correct? | a **count** of the mnemonic in the compiled SASS ("Footprint check" below) + correctness | `EXPR B=<asm text>` or `EXPR B=none (<compile error / footprint absent>)` |
| **C** toolchain limit | may only be written after A and B were both measured as none | cite the A and B verdict lines | `EXPR C=toolchain-limit` |

A is often the answer: **check the signature before writing asm**. A knob once checked is recorded in the entry of the corresponding interface
(the cache qualifiers of `cp_async` are one example), and is cited directly next time.

## Minimal runnable example

Three forms cover all use cases: no operands, address operands, a return value. The calling convention is identical for all:
`@dsl_user_op` catches `loc`/`ip`; operands go through `.ir_value(loc=loc, ip=ip)`;
the asm uses `$0,$1,...` placeholders (`AsmDialect.AD_ATT`); the constraint string corresponds one-to-one with the placeholders
(`"l"` = 64-bit integer (gmem address), `"r"` = 32-bit integer (smem window address / scalar),
`"f"` = fp32, `"=r"` / `"=f"` = outputs); anything with side effects must have `has_side_effects=True`.

```python
import cutlass
from cutlass._mlir.dialects import llvm
from cutlass._mlir import ir
from cutlass.cutlass_dsl import dsl_user_op, T


# 1) No operands, side effects only: commit / wait style
@dsl_user_op
def cp_async_wait_group(n, *, loc=None, ip=None):
    llvm.inline_asm(None, [], "cp.async.wait_group %d;" % int(n), "",
                    has_side_effects=True, is_align_stack=False,
                    asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)


# 2) Address operands: a modifier the DSL op lacks (here the L2 prefetch width of cp.async)
@dsl_user_op
def cp_async_16_l2_256(dst_smem_addr, src_gmem_addr, *, loc=None, ip=None):
    """dst: Int32 (shared::cta window address); src: Int64 (global). 16 B aligned."""
    llvm.inline_asm(
        None,
        [dst_smem_addr.ir_value(loc=loc, ip=ip),
         src_gmem_addr.ir_value(loc=loc, ip=ip)],
        "cp.async.cg.shared.global.L2::256B [$0], [$1], 16;",
        "r,l",
        has_side_effects=True, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)


# 3) Return value: an intrinsic the DSL does not export
@dsl_user_op
def cvt_f16x2_f32(a, b, *, loc=None, ip=None) -> cutlass.Int32:
    return cutlass.Int32(llvm.inline_asm(
        T.i32(),
        [cutlass.Float32(a).ir_value(loc=loc, ip=ip),
         cutlass.Float32(b).ir_value(loc=loc, ip=ip)],
        "cvt.rn.f16x2.f32 $0, $2, $1;", "=r,f,f",
        has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip))
```

Where addresses come from: for gmem use `tensor.iterator.toint()` (Int64); for smem the same call yields the shared window
address (Int32). **For smem addresses see gotcha 5** — under a swizzled layout this integer is not the hardware address.

### Footprint check (the evidence for rung B, and the only criterion that "the modifier really got emitted")

```python
# Make the JIT keep the cubin at compile time, then count mnemonics; numerical correctness is irrelevant here.
# DSL 4.8.0 switches (all three matter, each was hit on sm120):
os.makedirs("/tmp/dump", exist_ok=True)     # the dump directory must EXIST: otherwise nothing is written, silently
env = {"CUTE_DSL_KEEP": "cubin",            # CUTE_DSL_KEEP_CUBIN=1 still works but is deprecated
       "CUTE_DSL_DUMP_DIR": "/tmp/dump",
       "CUTE_DSL_NO_CACHE": "1"}            # a JIT-cache hit compiles nothing and dumps nothing
# "sass" is also a CUTE_DSL_KEEP token, but needs nvdisasm >= 13.4 (with CUDA 13.0 the DSL refuses at import time);
# keep the cubin and disassemble it yourself:
# cubin = the newest /tmp/dump/*.cubin   (the compiled object's __cubin__ names it)
# sass  = `cuobjdump -sass <cubin>`
# e.g.: len(re.findall(r"\bLDGSTS\.E\.BYPASS\.LTC256B", sass))  -> expected == number of issue sites
```

Measured on sm120 for the three forms below: form 2 lands `LDGSTS.E.BYPASS.LTC256B.128`, form 1 lands `LDGDEPBAR` +
`DEPBAR`, form 3 lands `F2FP.F16.F32.PACK_AB`; the smem round trip of form 2 and the packed halves of form 3 are bit-exact.

Print one line of the form `FOOTPRINT <mnemonic>=<count>` and use it together with `PROBE PASS max_diff=...` as the verdict.
Each evaluation iteration's NCU job also runs the same census automatically (the `cache_ops`
and other axes of `ORCH-SUMMARY-JSON`), so whether a modifier landed is machine-readable in the trajectory.

## Gotchas

1. **Check rung A before writing asm.** A modifier is often an enum among the op's constructor parameters
   (e.g. `cache_mode` of the G2S copy op). Before writing asm, print
   `inspect.signature(Op.__init__)` and the enum members inside a job; names written from memory (`CG` / `L2`)
   mostly do not exist, while the correct name (`GLOBAL`) is right there in the printout. On 4.8.0:
   `CopyG2SOp.__init__(self, cache_mode: LoadCacheMode = LoadCacheMode.ALWAYS)`, members
   `ALWAYS / GLOBAL / LAST_USE / NONE / STREAMING` -- no L2 prefetch-width knob, so `.L2::256B` is rung B.
2. **Placeholders are `$n`, not `%n`.** Under the `AD_ATT` dialect `%0` is not substituted; on 4.8.0 the DSL passes it
   through and the failure surfaces in ptxas as `CompilerDiagnosticError: error: PTX assembly failed` -- no hint that the
   placeholder is the cause.
3. **Constraint string order = placeholder order**, and the types must be right: `l` for Int64, `r` for Int32, `f` for fp32;
   outputs use `=r` / `=f` and come first (corresponding to `$0`). A wrong type does not necessarily fail at compile time; at run time the address is wrong.
4. **Do not omit `has_side_effects=True`**: without it, an asm with no return value is removed as dead code,
   which shows up as "nothing happened, footprint 0". (Measured on sm120: `cp.async.commit_group` without the flag leaves
   0 `LDGDEPBAR` in the cubin, with it 1.)
5. **Addresses of swizzled smem.** *(Not re-checked on sm120.)* Calling `.iterator.toint()` on a sliced swizzled smem tensor
   yields the **unswizzled linear window address**; the DSL's copy ops apply the swizzle only during lowering.
   Hand it to asm directly and the footprint is right but the data is wrong (written to a different bank row). Two ways out:
   (a) apply the same swizzle by hand in the asm (128 B swizzle: byte address bits[4,7) ^= bits[7,10),
   i.e. `t = (addr >> 3) & 0x70; addr ^= t`); (b) let the DSL compute the address — take the address from an unswizzled linear view,
   or keep the DSL op and change only its parameters (back to rung A).
6. **The `.L2::<n>B` prefetch width and `.ca/.cg` are two independent bits.** `cg` (bypass L1) shows up in SASS as
   `BYPASS`, `L2::256B` as `LTC256B` (sm120: `LDGSTS.E.BYPASS.LTC256B.128`); the DSL's `cache_mode` only controls the former. Their value also
   differs (see the cache-qualifier section of `sm90/cp_async`); price them separately.
7. **Modifiers are "numerically invisible"**: no correctness probe can tell whether one is present. Certification evidence can only be
   the footprint count + interleaved A/B timing readings in the same job; the entry must state the **load-bearing conditions** (under which resource state
   it is worth anything), not a "general X%".
8. **File level: no `from __future__ import annotations`** (the type
   annotations of `@dsl_user_op` are evaluated at run time).
9. **DSL versions rot the scattered examples of this entry**: op parameter names, the `llvm` module path and where `T` is imported from all change between
   versions. Every use case that cites this general rule must first run its own probe on a new image before use.
   On 4.8.0 the imports used here (`cutlass._mlir.dialects.llvm`, `cutlass.cutlass_dsl.dsl_user_op` / `T`) are unchanged;
   what moved is the artifact switches of the footprint check (above).

## Sources

- PTX ISA: inline asm and the modifier syntax of each instruction
  <https://docs.nvidia.com/cuda/parallel-thread-execution/> (`.ca/.cg` and
  `.L2::prefetch_size` of `cp.async`; `cvt`; `shfl.sync`; `cp.reduce.async.bulk`);
  NVVM/LLVM inline asm constraint letters (`r`/`l`/`f`)
  <https://llvm.org/docs/NVPTXUsage.html>.
- Public infra API surface: `cutlass/cute/nvgpu/cpasync/helpers.py` (`fence_tma_desc_acquire` /
  `cp_fence_tma_desc_release`: where the `@dsl_user_op` + `llvm.inline_asm` + `AD_ATT` calling form
  comes from); `cutlass/cute/core.py::Pointer.toint` (gmem → Int64, smem → Int32);
  `cutlass.cute.nvgpu.cpasync.CopyG2SOp` (the template for rung A "check the signature first":
  `cache_mode: LoadCacheMode`).
- GPU certification on sm120 (CuTe DSL 4.8.0): `probes/sm120/methodology/ptx_escape_check.py` --
  `PROBE PASS ptx-escape contracts=8`: `PE-A-cache-mode-knob`, `PE-form2-numerics`, `PE-form2-footprint`
  (`LDGSTS.E.BYPASS.LTC256B.128`), `PE-form1-footprint`, `PE-form3-numerics`, `PE-G2-percent-placeholder`,
  `PE-G4-no-side-effects-dce`, `PE-dump-needs-existing-dir`.
- Transcribed from the sm90 tree: one probe per form plus the gotcha 5 pair (`asm_l2_A` hand-applied swizzle PASS /
  `asm_l2_B` linear address FAIL); not included in this release.
