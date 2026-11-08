# 在 RTX PRO 5000（sm120）上用 ForgeKernel 优化一个 kernel

本文是 `sm120-kb` 分支的上手说明：环境怎么起、任务怎么写、run 怎么跑、结果怎么验。每一条都是在这台机器上踩过的坑。

## 1. 已经验证过的结果（2026-11）

| 任务 | 参考实现 | forge 结果 | 说明 |
|---|---|---|---|
| nvfp4 GEMM（大形状） | cuBLASLt | 1.016x | 库已接近硬件上限 |
| attention 前向（prefill，causal GQA d128） | cuDNN（约 72% 峰值） | 1.034x | 同上 |
| **FP8 decode GEMM**（M=1~64） | CUTLASS sm120（sgl_kernel） | 第 1 代 1.49x，第 2 代 +8.8% | 在库没调好的地方明确超过库 |
| **nvfp4 decode GEMM**（M=1~64） | cuBLASLt | 1.47x（一轮，复用 stream-K 条目） | 同上；逐次调用含启动开销 1.22~1.43x |

结论：标准算子很难大幅超过库；收益在库覆盖差的地方——小 M、融合、sm120 上缺实现的路径。
已测好的候选空间（尚未做）：FP8 MoE grouped GEMM、FP8 blockwise GEMM（PyTorch 在 sm120 上均不支持），
RMSNorm → FP8 量化融合（两 kernel 合计只到融合理想带宽的 40%），bf16 MoE grouped decode（74% 带宽）。

## 2. 环境

- **容器 `kernel-agent`**（镜像 `kernel-agent:sandbox`，`--gpus all`）。它和普通容器的区别：
  `--security-opt apparmor=unconfined --security-opt seccomp=unconfined --security-opt systempaths=unconfined`，
  cap `SYS_ADMIN PERFMON SYS_NICE`，启动命令先跑 `/usr/local/bin/kernel-agent-init.sh`
  （卸掉 NVIDIA 运行时盖在 `/proc/driver/nvidia` 上的挂载——否则 Claude Code 的 bubblewrap 沙箱在 user
  namespace 里挂不上 proc）。容器里有 `claude`、登录凭据、代理设置、bubblewrap、socat、ncu 计数器权限。
- **GPU：只用 0~3 号卡，默认 3 号**（`local.env` 的 `FK_GPU=3`；forge 每次启动都按它重写各 run 的
  submit.toml，所以改卡要改 `local.env`，不要改 run 目录）。4~7 号卡不碰。
- **两个必须的环境变量**（启动 `forge run` 的 shell 里 export）：
  - `FK_AGENT_ALLOWED_TOOLS=Read,Write,Edit,MultiEdit,Glob,Grep,Bash,NotebookEdit,TodoWrite,Task,Agent`
    ——组织的托管策略禁用了 bypass 模式，不设这个，agent 写不了 cwd 以外的文件，run 会以 `agent_failed` 停下。
  - `FK_JOB_TIMEOUT_S=3600`——大形状、多变体的测价任务会超过默认的 1800 s。

## 3. 写一个任务（`tasks/<name>/`）

参考 `tasks/sm120_fp8_decode_gemm/`、`tasks/sm120_nvfp4_decode_gemm/`。

- `task.toml`：`arch = "sm120"`，`language = "cutedsl"`，`[target] ratio`（相对参考实现的倍数）。
  **访存瓶颈的算子、输入小于 96 MB L2 时，一定加 `l2 = "cold"`**：否则反复调用时数据一直在 L2 里，
  测出来的加速比会偏高（FP8 decode GEMM：热 1.39x，冷 1.23~1.29x）。
- `shapes.json`：维度只能是数字或轴名，不能写 `k // 2`；派生维度（打包字节、scale 列数、补齐行数）要
  作为轴写进去（见 nvfp4 decode 任务的 `kh / ks / mp`）。
- **把合法输入范围的边界放进形状**：判题只测声明的形状。FP8 decode 第一版只测 M=16/64，结果 M=1 会崩溃。
- `description.md`：签名、语义、合法输入。FP4 用 `torch.float4_e2m1fn_x2`，blocked scale 布局要写出偏移公式。
- `reference/run.py`：选**同一台卡上实测最快**的库实现（先用 `probes/sm120/device_facts/ref_ceilings.py`
  一类的脚本测一遍，并且在空卡上测——别人的任务占卡时数据会被压一半）。

## 4. 跑

```bash
docker exec -w /root/akiraz/welm/ForgeKernel kernel-agent bash -c '
  .venv/bin/forge new tasks/<name> --dry-run                      # 先查任务
  R=$(.venv/bin/forge new tasks/<name> | grep -o "\.artifacts/[^ :]*" | head -1)
  tmux new-session -d -s forge_<name> "export FK_AGENT_ALLOWED_TOOLS=...  FK_JOB_TIMEOUT_S=3600; \
      .venv/bin/forge run $R 2>&1 | tee <log>"'
```

- 续跑：同一个 run 目录再 `forge run` 一次，从断点接着走。
- 自进化（下一代以本代最佳 kernel 为参考）：用 `scripts/self_evolve.py` 里的 `next_generation()` 从已完成的
  run 生成 `tasks/<name>-g2/`，再按上面跑；目标倍数按剩余空间定（消融上限 +17% 时用 1.08，不是默认 1.05）。
- 监控日志时，过滤管道最后一级用 `sed -u`，不要用 `cut`（会缓冲，事件出不来）。

## 5. 验结果（不要只信 forge 的数字）

1. **正确性**：随机数据对 fp32 精确参考的误差（bf16 舍入量级）；**selector 输入**（B 每行只有一个 1）要求
   逐元素精确——随机数据看不出 operand 映射错误；**扫描合法范围内的各个尺寸**（如 M=1..128，每个尺寸单独
   一个进程，崩溃会污染 CUDA 上下文）；含跨 CTA 归约的 kernel（stream-K）**连续跑 50 次比对逐位一致**。
2. **性能**：自己用冷数据（轮换多份权重，总量 > 2x L2）、逐次调用计时再测一遍，和参考实现同卡对比。
   forge 的设备窗口不含启动开销，逐次调用含；两者都报。
3. **KB 审核**：forge 收尾时会把它认为值得沉淀的内容写进 `hw_kb/`，标成"待审核"。只收经过证实、
   且会改变 sm120 kernel 写法的内容；没被收录但有用的实测结论（如 decode GEMM 的分块经验）手动补进
   `device_facts`。提交前跑 `forge.loop.kb_lint`。

## 6. 已知的框架行为（不是 bug）

- 一个决策变慢了也可能被判 KEEP（"脚手架"），最佳 kernel 单独记录，后续轮次会撤销。
- 第二代的 M0 是先复现参考实现，持平是正常的，后续 milestone 才加新改进。
- 泄漏检查会从参考源码提取标识符；DSL 惯用名（`Storage`、带下划线的短名）已排除，如果又被误报，
  看 `src/forgekernel/loop/kb_lint.py` 的 `_DERIVE_STOPLIST`。
