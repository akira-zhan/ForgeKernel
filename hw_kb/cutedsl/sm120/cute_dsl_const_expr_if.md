# cute_dsl_const_expr_if (compile-time branches: `cutlass.const_expr`)

> **Certification source: the sm90 tree · re-checked on sm120 with CuTe DSL 4.8.0** (`probes/sm120/methodology/dsl_const_expr_check.py`,
> `PROBE PASS dsl-const-expr contracts=5`; the minimal example below is the probe's kernel verbatim). Gotcha 4's cache
> hazard was **not** reproduced on 4.8.0 (see there).
>
> One entry per hardware interface. Operator-agnostic; mechanism only. This entry is the **branch-side** counterpart of
> [`cute_dsl_trace_scope`](cute_dsl_trace_scope.md): that one covers
> "which code the preprocessor rewrites"; this one covers "where the scope boundary of the rewritten region lies".

## Purpose

A CuTe DSL kernel is traced Python: the preprocessor walks the AST of `@cute.jit` / `@cute.kernel`
and decides, **statement by statement**, whether each control-flow statement is "evaluated on the spot in the Python interpreter" or "lowered to IR
and executed at run time". For `if` / `elif` / `else` / `while`, **the default is the latter** -- even when the condition is
a Python constant, the statement is lowered to a run-time region.

`cutlass.const_expr(<condition>)` flips that one statement to the former: the condition is evaluated by Python at trace time,
only the selected branch enters the trace, the `if` falls back to an ordinary Python branch, and bindings inside the branch are visible outside
under ordinary Python scoping. It is the **branch-side pair interface** of `cutlass.range_constexpr` (the loop side, compile-time unrolling)
-- since DSL 4.1.0 the default for `if`/`while` flipped from "compile time" to "run time", and `const_expr`
is the only annotation entry point (`dynamic_expr` was deprecated in the same version, see the official changelog).

## When needed (trigger)

Three symptoms, all pointing at the same thing:

1. **`NameError: name '<x>' is not defined`** -- on DSL 4.8.0 wrapped as
   `DSLUserCodeError: NameError in `<jit function>`: name '<x>' is not defined` (older versions: a bare `NameError` with a stack
   of the form `ast_helpers.ir_loop → if_region_N → if_executor.if_execute`).
   `<x>` is **first bound** in some `if` branch but referenced outside the branch (or in another branch). The official
   limitation reads "Values originating in control flow body are not available
   outside the control flow" -- a branch is a scope of its own, and names newly created inside a region **do not escape**.
   Note that this is orthogonal to the intuition "Python only executes the True branch, so at gear 0 these names are never evaluated":
   the error does not come from whether the branch is executed but from **the reference point being outside the region**.
2. **The condition is clearly a compile-time switch** (`_DEPTH >= 1`, `X is None`, a tile-size comparison,
   a `cutlass.Constexpr` parameter), yet a branch or the code of both branches is visible in SASS/PTX.
3. **`break` / `continue` / `return` / `raise` are rejected inside a control-flow body** (MLIR SSA requires
   every path to yield consistently), and the error message itself suggests `const_expr` instead -- when the condition is static, this
   is the official way out.

The criterion (the only one): **a condition built only from Python constants / module globals / `Constexpr` parameters / compile-time
shapes → must be wrapped in `const_expr`; a condition containing run-time values such as `tidx`/`bidx`/tensor reads →
must not be wrapped** (wrapping is rejected: feeding a dynamic value into `const_expr` is illegal).

## Minimal runnable example

```python
import cutlass
import cutlass.cute as cute

_GEAR = 1          # compile-time gear switch: a module global read at trace time

@cute.kernel
def k(mIn: cute.Tensor, mOut: cute.Tensor, mAux: cute.Tensor):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    i = bidx * 128 + tidx
    x = mIn[i]

    # (a) Loop side: compile-time unrolling; bindings in the body **can** escape (it is plain Python unrolling).
    for u in cutlass.range_constexpr(2):
        z = x * cutlass.Float32(float(u))
    y = z                                   # legal

    # (b) Branch side: the condition is a compile-time constant -> must be wrapped in const_expr.
    #     The two call sites use **the same** condition; only then do gear-specific values survive across sites.
    for t in cutlass.range(4):              # run-time IR loop
        if cutlass.const_expr(_GEAR >= 1):
            w = x * cutlass.Float32(4.0)    # gear-specific temporary, first bound here
            mAux[i] = cutlass.Float32(1.0)  # gear-specific side effect
        mOut[i] = mOut[i] + x
        if cutlass.const_expr(_GEAR >= 1):
            mOut[i] = mOut[i] + w           # harvest across call sites: visible only with the same condition

    # (c) if/else each binding once and used outside the branches: must be wrapped too, or `b` cannot leave the region.
    if cutlass.const_expr(_GEAR >= 1):
        b = x * cutlass.Float32(2.0)
    else:
        b = x - cutlass.Float32(3.0)
    mOut[i] = mOut[i] + b + y

    # (d) A truly run-time condition: do not wrap. Wrapping it is rejected.
    if tidx < 32:
        mOut[i] = mOut[i] * cutlass.Float32(0.5)
```

Remove the `cutlass.const_expr(...)` of (b) / (c), change nothing else, and `cute.compile` dies at
trace time with `NameError: name 'w' is not defined` / `name 'b' is not defined` --
this is exactly the single-variable control the certification probe runs (see Sources).

**A name used across branches has only three legal ways out**:
1. The condition is a compile-time quantity → wrap it in `const_expr`, and the whole stretch folds into plain Python (preferred);
2. The condition is a run-time quantity → bind the name unconditionally once **before** the region (give it a neutral initial value),
   and inside the region only **reassign** it, never bind it for the first time;
3. The name carries data → land it in a fragment / smem allocated before the region, and inside the region only write
   contents without creating new names.

## Gotchas

1. **Loops and branches are asymmetric; this is the point most likely to fool you.**
   Bindings inside a `range_constexpr` body escape (plain Python unrolling); bindings inside an `if` branch do not
   (region scope). The two constructs behave differently in the same kernel body; "my `range_constexpr` here
   has always been fine, so `if` should be the same" is the wrong model.
2. **The two call sites of a gear switch must use the same condition.** If the binding site is wrapped and the harvest site is not
   (or the other way round), the two sites live in different scopes: at best a `NameError`, at worst harvesting the value of another gear.
   In practice: gear-specific objects (a dedicated pipeline state, a dedicated temporary plane) **are created and used
   either inside the same `const_expr` section, or each under a literally identical `const_expr` condition**.
3. **`const_expr` only takes things that can be evaluated at trace time.** Wrapping a run-time value is illegal
   (`if cutlass.const_expr(dyn == 10)` is rejected); conversely, `for i in range(n)` is still lowered to an IR loop even when
   `n` is a Python int -- "a Python constant" is not "compile-time control flow";
   the annotation is.
4. **Module-global conditions are read at trace time.** When a global is used as a gear switch, the assignment must happen **before the first
   `cute.compile`**; changing the global in the same process and compiling the same entry again, the JIT cache may
   hand back the previous gear's cubin directly (observed with the DSL of the sm90 tree). **On 4.8.0 this was not
   reproduced**: compiling the same `@cute.jit` host object again after flipping the global in the same process, and two
   processes (gear 0 then gear 1) sharing one `CUTE_DSL_CACHE_DIR`, both produced the new gear's exact output. Keep the
   discipline anyway -- it costs nothing and the cache key is not a documented contract: one subprocess per gear, with
   `CUTE_DSL_CACHE_DIR` isolated per gear (or, better, make the gear a `cutlass.Constexpr` parameter, which is part of the
   compile signature).
5. **`const_expr` is not `cutlass.Constexpr`.** The latter is a parameter **annotation** (declaring the parameter a
   compile-time quantity, see [`cute_dsl_trace_scope`](cute_dsl_trace_scope.md) gotcha 3); the former is a
   statement-level **condition annotation**. Typical combination: a `Constexpr` parameter comes in, and `const_expr` tests it.
6. **Inside a run-time region a variable's type may not change**, nor may `break`/`continue`/`return`/`raise` be used.
   When you hit either, first ask whether the condition is static -- if it is, `const_expr` solves it in one step and the algorithm need not change.
7. **The folded-away branch lands nothing.** A `const_expr(False)` branch produces no code, and side effects inside the branch
   (writes to an auxiliary plane, counter increments) do not happen -- this has been certified on the GPU as "at gear 0 the auxiliary plane
   stays 0 element by element". Conversely, a bare `if` branch is a region function, and where its side effects land is decided by
   the lowered region; do not infer it from "Python will not take this branch".

8. **The same thing on the loop side: a Python container can only be subscripted with the induction variable of `range_constexpr`.** A common pattern in a segment
   is to build a Python staging list first (`s = [None] * E`) and write/read it in an unrolled loop according to a static run table
   (`RUNS[q]`, `s[e] = v[k]`). It does not matter if the outer loop is written as `cutlass.range(n)` (a dynamic trip) -- the list
   is built **inside its body** and used inside the body; but when the inner loop that walks the runs is written as a plain `range(...)` / `cutlass.range(...)`,
   `q` is a staged value, and the step `RUNS[q]` subscripts a Python tuple with an MLIR value, failing at trace time with
   `error[PHASE_DYNAMIC_INDEX]: Cannot use a Runtime value (Staged value) value as a list index or for loop range in plain
   Python code` (4.8.0 text). Write the inner loop as `cutlass.range_constexpr(...)` so that `q` stays a Python int;
   correspondingly, write the static guard inside the body (`q + S < len(RUNS)`) as `cutlass.const_expr(...)` per gotcha 1,
   or the values of the vector reads issued in the guarded branch cannot leave the region. A minimal repro of this shape (the staging mechanism inside a dynamic-trip
   loop, no TMEM / mbarrier) is contract (b) of a separate compile probe; contract (a) of the same probe
   is its sister gotcha: a nested `def` capturing a traced value reports `SCOPE_CLOSURE_CAPTURE`, which belongs to
   [`cute_dsl_trace_scope`](cute_dsl_trace_scope.md).

## Sources

- Official documentation (mechanism and limitations, the run-time/compile-time comparison table, the `break/continue/return`
  limitation, "Values originating in control flow body are not available outside the
  control flow"):
  <https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/dsl_control_flow.html>
- Official changelog (from 4.1.0 `if`/`while` lower to run time by default and need an explicit `const_expr`;
  `dynamic_expr` deprecated):
  <https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/changelog.html>
- Public infra API symbols: `cutlass.const_expr`, `cutlass.range_constexpr`,
  `cutlass.range`, `cutlass.Constexpr`, `cute.compile`.
- GPU certification on sm120 (CuTe DSL 4.8.0): `probes/sm120/methodology/dsl_const_expr_check.py` --
  `PROBE PASS dsl-const-expr contracts=5`: `C1-gear0` (bit-equal to the host formula, the folded branch's auxiliary-plane
  write never landed), `C2-gear1` (bit-equal, auxiliary plane all 1, `ref_sep > 0`), `C3-plain-b-nameerror` (`'w'`, inside a
  dynamic loop), `C6-plain-c-nameerror` (`'b'`, kernel top level), `G8-dynamic-index` (`PHASE_DYNAMIC_INDEX` for the
  `cutlass.range` spelling, the `range_constexpr` spelling inside a Runtime-trip body exact). Each gear runs in its own process
  with its own `CUTE_DSL_CACHE_DIR`; the gotcha 4 facts (same-process recompile, shared on-disk cache) are printed by the same probe.
- Transcribed from the sm90 tree: the original six-contract probe and the gotcha 8 compile probe (not included in this release).
