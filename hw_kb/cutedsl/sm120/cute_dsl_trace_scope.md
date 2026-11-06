# cute_dsl_trace_scope (seven trace-scope gotchas: the decorator boundary, decorator syntax vs a `cute.jit(f)` call, closure capture, Constexpr transcription, API drift, `range_constexpr` in an undecorated helper, subscript stores through an object attribute)

> **Certification source: the sm90 / sm100 trees · re-checked on sm120 with CuTe DSL 4.8.0**: every gotcha below,
> including the new gotcha 7, is a contract of `probes/sm120/methodology/dsl_trace_scope_check.py`
> (`PROBE PASS dsl-trace-scope contracts=11`); the error texts quoted are the 4.8.0 texts.
>
> One entry per hardware interface. Operator-agnostic; mechanism only. This entry generalises `sm90/cp_async` gotcha 7b
> from `range_constexpr` to **all runtime control flow**, plus two probe-engineering facts.
> Names of the form `sm90/xxx` / `sm100/xxx` are entries of other trees and **are not in the read set of an sm120 run**;
> they are named so that you know they exist; their readings were taken on other hardware and DSL versions.

## Purpose

A CuTe DSL kernel is **traced Python**: which code the preprocessor rewrites and which values are visible at
trace time is decided by the **decorator boundary**, contrary to Python-semantics intuition. This entry records four
boundary facts about compile-time hard failures / false positives, all verified on the GPU as "one cut, the verdict flips".

## When needed (trigger)

When writing / changing any DSL kernel that has helper functions, runtime branches, or a cross-version migration; and
when writing a certification probe for such a kernel.

## Mechanism and gotchas

1. **Every helper that contains runtime control flow must carry its own `@cute.jit`.** The preprocessor that rewrites
   `if`/`for` into staged control flow **only walks the source of the decorated function itself**; it does not descend
   into the ordinary Python helpers it calls. A comparison against a runtime value in an undecorated helper
   (`if m1 >= s_q:`, `m1` derived from `bidx`) fails at trace time with (DSL 4.8.0):

       DSLUserCodeError  error[PHASE_DYNAMIC_TO_STATIC_BOOL]: Cannot use a Runtime value (Staged value) value
       where a Python value (Meta value) boolean is required (e.g. in plain Python control flow)

   The fix is one cut: add `@cute.jit` to the helper. A decorated helper can be called normally from a traced body --
   the control flow does **not** have to be inlined into the kernel body (this refines cp_async 7b's "the loop cannot
   be moved out": it cannot be moved out into an *undecorated* function; once decorated, it can).
   Trap shape: if such a helper is introduced only by **build-phase** commits (pre-green, no whole-machine eval), the
   defect can lie dormant for several rounds -- "the transcribed probe compiles" is not "the shipped file compiles",
   see gotcha 3.
   **Corollary (removing one reason not to add the decorator)**: adding `@cute.jit` to a helper does **not** cut its
   writes into register fragments **owned by the caller** -- the decorated helper writes into the fragment passed in
   as an argument, and after it returns the caller sees the written values. So "afraid of losing writes across the jit
   boundary" is not a reason to keep a helper undecorated; the two forms have the same semantics on this point, and
   only the former is rewritten correctly by the preprocessor.
   (GPU-certified on sm120: a decorated helper -- module-level or a nested `def` with `@cute.jit` -- writes a per-thread
   value into a fragment owned by the caller; the caller stores it, `max_diff == 0.0` over 128 threads.)

2. **SCOPE_CLOSURE_CAPTURE: a function called inside a staged `if`/`for` must not closure-capture runtime values.**
   When trace enters a function from staged control flow, runtime values in its closure (tensors, indices)
   are not available -- they must be passed explicitly as arguments. 4.8.0 text:
   `error[SCOPE_CLOSURE_CAPTURE]: Function `add_cap` captures variable `g`, which is not supported in staged for/while/if.`
   The same work through a module-level helper taking `(g, i, x)` explicitly runs bit-exact. When writing helpers, always passing explicitly is safest.

3. **Constexpr parameters must keep their annotation verbatim when transcribed.** A parameter declared as
   `cutlass.Constexpr` at the kernel entry, if **re-spelled** as an unannotated parameter in a probe/wrapper,
   is bound as dynamic by `cute.compile(wrapper, ...)` -- and a dynamic value cannot be fed into a
   `Constexpr` parameter: a hard compile-time failure. Measured on 4.8.0: `cute.compile(entry, t, 4)` runs; the same
   call through `wrapper(g, n): entry(g, n)` dies with
   `error[ARG_NON_CONSTANT]: range_constexpr() requires all arguments to be Python integers known at code-read time.` This is a "transcription-induced defect": the shipped entry itself
   has no problem. **A probe always `cute.compile`s the shipped entry as a whole** (together with its real
   constexpr/dynamic argument split); do not transcribe; the verdict of a transcribed version can be either a false
   negative or a false positive for the shipped file.

4. **Measure API drift; do not trust the error message's nearest suggestion.** Example: `cute.make_fragment(shape, dtype)`,
   which allocates a register tensor by shape/layout, has been removed in some DSL versions and reports

       AttributeError: module 'cutlass.cute' has no attribute 'make_fragment'.
       Did you mean: 'make_fragment_like'?

   The nearest suggestion is misleading -- `make_fragment_like` means "modelled on an existing tensor"; the new name
   for allocation by shape is **`cute.make_rmem_tensor`** (after the rename the same probe flips from FAIL to PASS).
   `cute.make_fragment` in this KB's older recipe examples reads as `cute.make_rmem_tensor` on such versions.
   **State on 4.8.0 (measured)**: `cute.make_fragment` absent; `cute.make_fragment_like`, `cute.make_rmem_tensor`,
   `cute.make_rmem_tensor_like` present. Other drift met on 4.8.0: `cutlass.utils.SmemAllocator` is deprecated in
   favour of `cutlass.memory.SmemAllocator` (a `DeprecationWarning` at the call); the per-artifact switches
   `CUTE_DSL_KEEP_CUBIN` / `_PTX` / `_SASS` are deprecated in favour of `CUTE_DSL_KEEP=cubin,ptx,...` (see
   [`ptx_escape`](ptx_escape.md)).
   **How to enumerate all drift in one pass (api-audit)**: walk every dotted DSL name that appears in the artifact and
   try to resolve it, reporting every unresolvable name in one round -- but **strip comments and strings** first before
   scanning (comments recording old names create false positives), and the audit is a text lint, so it may only use the
   `note:` diagnostic channel and **must not occupy a PROBE verdict line** (a check that cannot fail on the artifact
   must not be able to fail the verdict -- name resolution on the traced path is already covered by the compile contract).

5. **In an undecorated helper: `const_expr` works, `range_constexpr` does not.** An ordinary Python helper called from
   the kernel body (without `@cute.jit` / `@cute.kernel`) is not rewritten by the preprocessor. `cutlass.const_expr`
   is a real function; it folds trace-time Python values (ints / bools in the closure) as usual;
   `cutlass.range_constexpr`, however, is a **preprocessor marker**; when the preprocessor has not consumed it, it is
   itself a trace-time exception:

       DSLRuntimeError  error[INTERNAL]: range_constexpr should be preprocessed by preprocessor.

   So in the same helper, every `const_expr` gate before the `range_constexpr` works, and only the rung that reaches
   that loop dies -- it looks like "the construct this rung adds is broken", but it is the loop spelling. Fix:
   use a plain `range` under a static upper bound (it unrolls the same as `range_constexpr`), or add `@cute.jit` to the
   helper (gotcha 1). It costs one GPU round.
6. **Inside a staged region, do not subscript-store a tensor through a Python object's attribute; bind it to a plain local first.**
   `obj.face[(r, c)] = v` written inside a `for` / `if` that lowers to a region fails at trace time with:

       error[TYPE_DYNAMIC_EXPR_UNSUPPORTED]: A value carried through this `for` is a
       plain Python value (Meta value) (a `<your object class>`) that cannot be turned into a
       Runtime value (Staged value)

   The AST transform resolves the subscript store to its **root name** `obj`, sees it referenced inside the staged
   region, and tries to make this Python object, which has no IR value, a loop-carried value. It has nothing to do with
   the loop bound or with what the object holds. Fix: `face = obj.face` before entering any staged `for` / `if`, and
   only subscript locals (or parameters) inside the region.
   The same defect lies dormant just as well in a staged `if` such as `if kh == 0:`; it simply has not been reached yet.
   (Re-checked on sm120: `h.face[i] = ...` inside `for i in cutlass.range(nrun)` dies with the text above; `face = h.face`
   before the loop runs bit-exact.)
7. **The preprocessor needs the decorator *syntax*; `cute.jit(f)` as a call does not rewrite `f`.** Wrapping a plain helper
   at run time (`helper = cute.jit(helper_body)`) produces a callable that traces, but its `if` on a Runtime value fails
   with the same `PHASE_DYNAMIC_TO_STATIC_BOOL` as the undecorated helper. Writing `@cute.jit` above the `def` works,
   at module level and on a nested `def` alike. So "I did wrap it in `cute.jit`" is not gotcha 1's fix unless it is the
   decorator line. (Measured on sm120 / DSL 4.8.0: three forms of one helper body, two fail identically, one runs bit-exact.)

## Sources

- GPU certification on sm120 (CuTe DSL 4.8.0): `probes/sm120/methodology/dsl_trace_scope_check.py` --
  `PROBE PASS dsl-trace-scope contracts=11`: `TS1-undecorated-helper-rejects`, `TS1-decorated-helper-ok`,
  `TS1-call-form-not-rewritten`, `TS2-closure-capture-rejected`, `TS2-explicit-args-ok`, `TS3-constexpr-shipped-ok`,
  `TS3-transcribed-wrapper-differs`, `TS4-make-rmem-tensor`, `TS5-range-constexpr-needs-decorator`,
  `TS6-attr-subscript-store-rejected`, `TS6-local-binding-ok`.
- CuTe DSL API symbols: `cute.jit`, `cute.kernel`, `cute.compile`, `cutlass.Constexpr`, `cutlass.const_expr`,
  `cutlass.range_constexpr`, `cutlass.range`, `cute.make_rmem_tensor`, `cutlass.memory.SmemAllocator`.
- Official documentation (control flow, the preprocessor): https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/dsl_control_flow.html
- How to organise the probe side to get several facts back in one round: [`gpu_probe_ladder`](gpu_probe_ladder.md).
- Transcribed from the sm90 / sm100 trees (readings taken there): the origin of gotchas 1-6 (an fp8 scoring-type operator's
  compile probes, an epilogue probe, a QK probe; not included in this release).
