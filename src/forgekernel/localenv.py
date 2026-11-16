"""``local.env``: the machine a run uses -- never part of a task.

A gitignored file at the repository root, one ``KEY=VALUE`` per line (``#`` comments,
optional ``export`` prefix, optional quotes). The process environment wins over the file
for any key both set, so a one-off ``FK_DELIVERY=ssh forge run ...`` needs no edit.

Keys the harness reads:

    FK_DELIVERY      local | ssh                    (default: local)
    FK_GPU           CUDA device index to run on    (default: 0)
    FK_JOB_TIMEOUT_S one GPU job's wall clock, in s (default: 1800)
    REMOTE_HOST      ssh: host alias; must log in non-interactively
    REMOTE_WORKDIR   ssh: absolute directory on the host, dedicated to forgekernel
    REMOTE_ENV_FILE  ssh, optional: sourced on the host before every command

Every key in the file (credentials such as ``ANTHROPIC_API_KEY`` or
``CLAUDE_CODE_OAUTH_TOKEN`` included) is exported to the agents and the GPU jobs a run
starts; values are never printed.
"""

from __future__ import annotations

import os
import re
import shlex
from dataclasses import dataclass, field
from pathlib import Path

from forgekernel.paths import repo_root

LOCAL_ENV = "local.env"
DELIVERY_LOCAL = "local"
DELIVERY_SSH = "ssh"
DELIVERIES = (DELIVERY_LOCAL, DELIVERY_SSH)
# One GPU job's wall clock. A long workload priced over many lever variants can need more
# (sm120 attention, b1 s16k: 12+ variant timings outran 1800 s): FK_JOB_TIMEOUT_S.
JOB_TIMEOUT_S = 1800
JOB_TIMEOUT_KEY = "FK_JOB_TIMEOUT_S"

_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class MachineError(ValueError):
    def __init__(self, problems: list[str]):
        self.problems = list(problems)
        lines = "\n".join(f"  - {p}" for p in self.problems)
        super().__init__(f"{LOCAL_ENV}: the machine is not usable\n{lines}")


@dataclass(frozen=True)
class Machine:
    delivery: str
    gpu: str
    remote_host: str = ""
    remote_workdir: str = ""
    remote_env_file: str = ""
    env: dict = field(default_factory=dict)   # every KEY=VALUE the file sets (merged)
    job_timeout: int = JOB_TIMEOUT_S          # FK_JOB_TIMEOUT_S, from the environment or the file

    def describe(self) -> str:
        if self.delivery == DELIVERY_SSH:
            return f"ssh {self.remote_host}:{self.remote_workdir} (GPU {self.gpu})"
        return f"this machine (GPU {self.gpu})"


def read_env_file(path: str | Path) -> dict:
    """KEY=VALUE pairs of ``path``; {} when it does not exist. A malformed line raises
    MachineError naming it (a credential silently dropped is worse than a refusal)."""
    p = Path(path)
    if not p.is_file():
        return {}
    out: dict = {}
    bad: list[str] = []
    for n, raw in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        key, sep, value = line.partition("=")
        key = key.strip()
        if not sep or not _KEY_RE.match(key):
            bad.append(f"line {n} is not KEY=VALUE")
            continue
        value = value.strip()
        if value[:1] in ("'", '"'):
            try:
                parts = shlex.split(value)
            except ValueError:
                bad.append(f"line {n}: unbalanced quotes")
                continue
            value = parts[0] if parts else ""
        out[key] = value
    if bad:
        raise MachineError([f"{p}: {b}" for b in bad])
    return out


def load_machine(root: str | Path | None = None, environ=None) -> Machine:
    """The machine ``local.env`` (under ``root``, the repository by default) describes,
    with the process environment taking precedence key by key."""
    environ = os.environ if environ is None else environ
    file_env = read_env_file(Path(root if root is not None else repo_root()) / LOCAL_ENV)
    merged = dict(file_env)
    for k in list(file_env) + ["FK_DELIVERY", "FK_GPU", "REMOTE_HOST", "REMOTE_WORKDIR",
                               "REMOTE_ENV_FILE", JOB_TIMEOUT_KEY]:
        if k in environ:
            merged[k] = environ[k]
    problems: list[str] = []
    delivery = (merged.get("FK_DELIVERY") or DELIVERY_LOCAL).strip()
    if delivery not in DELIVERIES:
        problems.append(f"FK_DELIVERY must be one of {list(DELIVERIES)}, got {delivery!r}")
    gpu = (merged.get("FK_GPU") or "0").strip()
    if not gpu.isdigit():
        problems.append(f"FK_GPU must be a CUDA device index, got {gpu!r}")
    # Read here, with the machine, so the file sets it like every other key: it used to be
    # read from the process environment when the task-card module was imported, and a value
    # in local.env was silently ignored (the job was killed at the default).
    raw_timeout = (merged.get(JOB_TIMEOUT_KEY) or "").strip()
    job_timeout = JOB_TIMEOUT_S
    if raw_timeout:
        if raw_timeout.isdigit() and int(raw_timeout) > 0:
            job_timeout = int(raw_timeout)
        else:
            problems.append(f"{JOB_TIMEOUT_KEY} must be a positive number of seconds, "
                            f"got {raw_timeout!r}")
    host = (merged.get("REMOTE_HOST") or "").strip()
    workdir = (merged.get("REMOTE_WORKDIR") or "").strip()
    env_file = (merged.get("REMOTE_ENV_FILE") or "").strip()
    if delivery == DELIVERY_SSH:
        if not host:
            problems.append("FK_DELIVERY=ssh needs REMOTE_HOST (an ssh alias that logs in "
                            "without a prompt)")
        if not workdir.startswith("/") or workdir.rstrip("/") in ("", "/root", "/home", "/tmp"):
            problems.append(f"FK_DELIVERY=ssh needs REMOTE_WORKDIR, an absolute directory on "
                            f"the host used only by forgekernel (got {workdir!r}); each run "
                            f"replaces trees under it")
    if problems:
        raise MachineError(problems)
    return Machine(delivery=delivery, gpu=gpu, remote_host=host, remote_workdir=workdir,
                   remote_env_file=env_file, env=merged, job_timeout=job_timeout)
