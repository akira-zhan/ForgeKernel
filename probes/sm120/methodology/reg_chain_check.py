#!/usr/bin/env python3
"""Register reduction chain on sm_120a: latency-bound vs issue-bound, and what ``.f32x2`` lowers to.

    python3 -u reg_chain_check.py --gpu <nvidia-smi index> [--out DIR]

Each thread holds N bf16x2 words. One iteration unpacks them, squares every bf16 value and reduces
to one fp32 scalar; the next iteration's words depend on that scalar (``w ^= bits(s) & Z``, Z = 0 at
run time), so iterations cannot overlap. The whole timed loop (``%clock64`` -> loop -> ``%clock64``)
is ONE inline-PTX block with ``.pragma "nounroll"``. Eight variants (k accumulators + a balanced
tree), each run at 1 / 2 / 4 / 8 warps per SM sub-partition (one block per SM).

Contracts:
- ``numerics``      : every variant's scalar equals the fp64 sum of squares (rel err < 1e-6);
- ``sass-no-packed``: no FFMA2 / FMUL2 / FADD2 anywhere (sm_120a has no packed fp32 issue);
- ``sass-split``    : on the two linear chains (no contraction possible) each PTX ``.f32x2`` op becomes exactly
  two scalar FP ops, the scalar chain one per value, and nothing runs on the uniform datapath;
- ``sass-fhfma``    : the ``fma.rn.f32.bf16`` variant issues 2N FHFMA.BF16 and >= 2N - 2 fewer integer
  (unpacking) ops than the scalar 8-accumulator variant;
- ``stage-run``     : the timed binary completes.
Facts: SASS FP / integer counts per variant, cycles per iteration per variant per warps/SMSP.
"""
import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
N = 16          # bf16x2 words per thread
REPS = 1024
WARPS = (1, 2, 4, 8)


def unpack(i):
    return [f"shl.b32 lo{i}, %{i}, 16;", f"and.b32 hi{i}, %{i}, 0xffff0000;"]


def packed(k):
    """k independent .f32x2 accumulators, round robin, then a log2(k) add tree. k = 0: full tree."""
    t, pk = [], 0
    for i in range(N):
        t += unpack(i) + [f"mov.b64 X{i}, {{lo{i}, hi{i}}};"]
    if k == 0:  # full tree: N muls + N-1 adds
        t += [f"mul.rn.f32x2 A{i}, X{i}, X{i};" for i in range(N)]; pk += N
        m = N
        while m > 1:
            t += [f"add.rn.f32x2 A{j}, A{2 * j}, A{2 * j + 1};" for j in range(m // 2)]; pk += m // 2; m //= 2
    else:
        t += [f"mul.rn.f32x2 A{j}, X{j}, X{j};" for j in range(k)]; pk += k
        for i in range(k, N):
            t.append(f"fma.rn.f32x2 A{i % k}, X{i}, X{i}, A{i % k};"); pk += 1
        m = k
        while m > 1:
            t += [f"add.rn.f32x2 A{j}, A{2 * j}, A{2 * j + 1};" for j in range(m // 2)]; pk += m // 2; m //= 2
    t += ["mov.b64 {sl, sh}, A0;", "add.rn.f32 s, sl, sh;"]
    return t, pk


def scalar(k):
    t, vals = [], []
    for i in range(N):
        t += unpack(i); vals += [f"lo{i}", f"hi{i}"]
    t += [f"mul.rn.f32 S{j}, {vals[j]}, {vals[j]};" for j in range(k)]
    for i in range(k, 2 * N):
        t.append(f"fma.rn.f32 S{i % k}, {vals[i]}, {vals[i]}, S{i % k};")
    m = k
    while m > 1:
        t += [f"add.rn.f32 S{j}, S{2 * j}, S{2 * j + 1};" for j in range(m // 2)]; m //= 2
    t.append("mov.f32 s, S0;")
    return t, 0


def mixed(k):
    """bf16 halves straight into fma.rn.f32.bf16 (no unpacking)."""
    t, hs = [], []
    for i in range(N):
        t.append(f"mov.b32 {{h{2 * i}, h{2 * i + 1}}}, %{i};"); hs += [f"h{2 * i}", f"h{2 * i + 1}"]
    t += [f"mov.f32 S{j}, 0f00000000;" for j in range(k)]
    for i in range(2 * N):
        t.append(f"fma.rn.f32.bf16 S{i % k}, {hs[i]}, {hs[i]}, S{i % k};")
    m = k
    while m > 1:
        t += [f"add.rn.f32 S{j}, S{2 * j}, S{2 * j + 1};" for j in range(m // 2)]; m //= 2
    t.append("mov.f32 s, S0;")
    return t, 0


VARIANTS = {  # name -> (generator, arg)
    "p_serial": (packed, 1), "p_acc2": (packed, 2), "p_acc4": (packed, 4), "p_acc8": (packed, 8),
    "p_tree": (packed, 0), "s_serial": (scalar, 1), "s_acc8": (scalar, 8), "m_acc8": (mixed, 8),
}


def kernel_src(name, body):
    zi, oi, ci = N, N + 1, N + 2
    decl = (f".reg .b32 lo<{N}>, hi<{N}>, tb, cnt; .reg .b64 X<{N}>, A<{N}>, c0, c1; "
            f".reg .f32 S<8>, s, sl, sh; .reg .b16 h<{2 * N}>; .reg .pred p;")
    fb = ["mov.b32 tb, s;", f"and.b32 tb, tb, %{zi};"] + [f"xor.b32 %{i}, %{i}, tb;" for i in range(N)]
    lines = (["{", decl, "mov.u64 c0, %clock64;", "mov.u32 cnt, 0;", f"L_{name}:", '.pragma \\"nounroll\\";']
             + body + fb + ["add.u32 cnt, cnt, 1;", f"setp.lt.u32 p, cnt, {REPS};", f"@p bra L_{name};",
                            "mov.u64 c1, %clock64;", f"sub.u64 %{ci}, c1, c0;", f"mov.f32 %{oi}, s;", "}"])
    asm = "\\n".join(lines)
    ins = ", ".join([f'"+r"(w[{i}])' for i in range(N)])
    return f'''
__global__ void {name}(const uint32_t* __restrict__ win, uint32_t z, float* out, unsigned long long* cyc) {{
  uint32_t w[{N}];
#pragma unroll
  // per-lane address: the values are equal across lanes, but the compiler cannot prove it, so the
  // chain stays on the vector datapath (identical inputs move it to UFFMA / ULOP3)
  for (int i = 0; i < {N}; ++i) w[i] = win[(threadIdx.x & 31) * {N} + i];
  float s; unsigned long long c;
  asm volatile("{asm}" : {ins}, "+r"(z), "=f"(s), "=l"(c));
  int g = blockIdx.x * blockDim.x + threadIdx.x;
  out[g] = s;
  if ((threadIdx.x & 31) == 0) cyc[g >> 5] = c;
}}
'''


def main_src():
    calls = "\n".join(
        f'  for (int wps : {{{", ".join(map(str, WARPS))}}}) {{ int th = 128 * wps; '
        f'{name}<<<sms, th>>>(dw, 0u, dout, dcyc); CK(cudaDeviceSynchronize()); '
        f'{name}<<<sms, th>>>(dw, 0u, dout, dcyc); CK(cudaDeviceSynchronize()); '
        f'report("{name}", wps, sms * th); }}' for name in VARIANTS)
    return f'''
#include <cstdio>
#include <vector>
#include <algorithm>
#define CK(x) do {{ cudaError_t e_ = (x); if (e_ != cudaSuccess) {{ printf("ERROR %s\\n", cudaGetErrorString(e_)); return 2; }} }} while (0)
static float* dout; static unsigned long long* dcyc;
static void report(const char* name, int wps, int threads) {{
  std::vector<unsigned long long> c(threads / 32); std::vector<float> o(1);
  cudaMemcpy(c.data(), dcyc, c.size() * 8, cudaMemcpyDeviceToHost);
  cudaMemcpy(o.data(), dout, 4, cudaMemcpyDeviceToHost);
  std::sort(c.begin(), c.end());
  printf("FACT variant=%s warps_per_smsp=%d cycles_per_iter=%.1f scalar=%.9g\\n", name, wps,
         (double)c[c.size() / 2] / {REPS}, o[0]);
}}
int main(int argc, char** argv) {{
  int sms; CK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
  uint32_t hw[{N}];
  for (int i = 0; i < {N}; ++i) hw[i] = strtoul(argv[1 + i], 0, 16);
  uint32_t hl[32 * {N}];
  for (int l = 0; l < 32; ++l) for (int i = 0; i < {N}; ++i) hl[l * {N} + i] = hw[i];
  uint32_t* dw; CK(cudaMalloc(&dw, sizeof hl)); CK(cudaMemcpy(dw, hl, sizeof hl, cudaMemcpyHostToDevice));
  CK(cudaMalloc(&dout, 110 * 1024 * 4)); CK(cudaMalloc(&dcyc, 110 * 32 * 8));
{calls}
  printf("DONE\\n");
  return 0;
}}
'''


def bf16_words(rng):
    vals = rng.uniform(-2, 2, size=2 * N).astype(np.float32)
    bits = (vals.view(np.uint32) >> 16).astype(np.uint32)  # truncate to bf16
    words = [int(bits[2 * i] | (bits[2 * i + 1] << 16)) for i in range(N)]
    exact = (bits.astype(np.uint32) << 16).view(np.float32).astype(np.float64)
    return words, float(np.sum(exact * exact))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", required=True)
    ap.add_argument("--out", default=str(HERE.parents[2] / ".sm120_scratch" / "reg_chain"))
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    verdicts, facts = [], []

    def verdict(tag, ok, md=""):
        verdicts.append((tag, ok)); print(f"PROBE {'PASS' if ok else 'FAIL'} {tag} ({md})", flush=True)

    src, pk_count = ["#include <cstdint>\n#include <cuda_runtime.h>\n"], {}
    for name, (gen, k) in VARIANTS.items():
        body, pk = gen(k)
        pk_count[name] = pk
        src.append(kernel_src(name, body))
    src.append(main_src())
    cu = out / "reg_chain.cu"; cu.write_text("".join(src))
    exe = out / "reg_chain"
    p = subprocess.run(["nvcc", "-gencode", "arch=compute_120a,code=sm_120a", "-O3", "-o", str(exe), str(cu)],
                       capture_output=True, text=True)
    if p.returncode:
        print(p.stderr[-3000:]); verdict("build", False); return 1

    # SASS census per kernel (whole kernel, like the sm100 table)
    sass = subprocess.run(["cuobjdump", "-sass", str(exe)], capture_output=True, text=True).stdout
    census, cur = {}, None
    ops = ("FFMA2", "FMUL2", "FADD2", "FFMA", "FMUL", "FADD", "FHFMA", "UFFMA", "SHF", "LOP3", "PRMT", "IMAD", "MOV")
    for line in sass.splitlines():
        m = re.search(r"Function : _Z\d+(\w+?)PKjjPfPy", line)
        if m:
            cur = m.group(1); census[cur] = dict.fromkeys(ops, 0); census[cur]["total"] = 0; continue
        m = re.search(r"/\*[0-9a-f]{4}\*/\s+(?:@!?U?P\w+\s+)?([A-Z][A-Z0-9_]*)(\.[A-Z0-9_.]*)?", line)
        if cur and m:
            census[cur]["total"] += 1
            if m.group(1) in census[cur]:
                census[cur][m.group(1)] += 1
    for name in VARIANTS:
        c = census.get(name, {})
        facts.append(f"sass {name}: " + " ".join(f"{k}={v}" for k, v in c.items() if v))
    verdict("sass-no-packed", all(c["FFMA2"] + c["FMUL2"] + c["FADD2"] == 0 for c in census.values()))
    fp = {n: census[n]["FFMA"] + census[n]["FMUL"] + census[n]["FADD"] for n in VARIANTS}
    uniform = {n: census[n]["UFFMA"] for n in VARIANTS}
    facts.append("scalar FP ops per variant " + str(fp) + "; uniform UFFMA " + str(uniform))
    # the two linear chains admit no contraction: each .f32x2 op must become exactly two scalar ops
    verdict("sass-split", fp["p_serial"] == 2 * pk_count["p_serial"] + 1 and fp["s_serial"] == 2 * N
            and not any(uniform.values()),
            f"p_serial ptx_f32x2={pk_count['p_serial']} sass_fp={fp['p_serial']}; s_serial sass_fp={fp['s_serial']}")
    ints = {n: census[n]["SHF"] + census[n]["LOP3"] + census[n]["PRMT"] + census[n]["IMAD"] for n in ("s_acc8", "m_acc8")}
    m = census["m_acc8"]
    verdict("sass-fhfma", m["FHFMA"] == 2 * N and ints["s_acc8"] - ints["m_acc8"] >= 2 * N - 2,
            f"FHFMA={m['FHFMA']} int ops s_acc8={ints['s_acc8']} m_acc8={ints['m_acc8']}")

    words, ref = bf16_words(np.random.default_rng(7))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu))
    p = subprocess.run([str(exe)] + [f"{w:08x}" for w in words], capture_output=True, text=True, env=env, timeout=300)
    verdict("stage-run", p.returncode == 0 and "DONE" in p.stdout, f"rc={p.returncode}")
    errs = []
    for line in p.stdout.splitlines():
        if line.startswith("FACT "):
            facts.append(line[5:])
            d = dict(re.findall(r"(\w+)=(\S+)", line))
            errs.append(abs(float(d["scalar"]) - ref) / ref)
    verdict("numerics", bool(errs) and max(errs) < 1e-6, f"max_rel_err={max(errs) if errs else 'n/a':.3g} ref={ref:.9g}")

    print("=== DIGEST ===")
    for f in facts:
        print(f"FACT {f}")
    fails = [t for t, ok in verdicts if not ok]
    print(f"PROBE {'PASS' if not fails else 'FAIL'} reg-chain contracts={len(verdicts)}"
          + (f" failed={','.join(fails)}" if fails else ""))
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
