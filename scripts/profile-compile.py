#!/usr/bin/env python3
"""Isolate Wasmtime compilation/loading, without assets or a running game.

Build with `cargo build --release -p blot-host --example compile_time`, then run
`scripts/profile-compile.py game.wasm --out target/compile-profile` on Linux.
Each sample has a new process; only the named disk-cache sample shares a cache.
OS file caches are not flushed. getrusage's maximum RSS is the child's peak,
not a sum of peaks or an estimate of live allocation. Raw output is retained.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import time


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wasm", type=Path)
    parser.add_argument("--executable", type=Path, default=Path("target/release/examples/compile_time"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--inline-budget", type=int)
    args = parser.parse_args()
    if platform.system() != "Linux":
        parser.error("RSS units and process accounting require Linux")
    if args.runs < 1:
        parser.error("--runs must be positive")
    if args.inline_budget is not None and not 0 <= args.inline_budget <= 0xFFFFFFFF:
        parser.error("--inline-budget must fit U32")
    wasm, executable, out = args.wasm.resolve(), args.executable.resolve(), args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    report = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "wasm_sha256": digest(wasm),
        "executable_sha256": digest(executable),
        "wasm_bytes": wasm.stat().st_size,
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "affinity": sorted(os.sched_getaffinity(0)),
        "load_start": os.getloadavg(),
        "rayon_num_threads": os.environ.get("RAYON_NUM_THREADS"),
        "inline_budget": args.inline_budget,
        "samples": [],
    }
    for name in ("cpu.max", "memory.max"):
        path = Path("/sys/fs/cgroup") / name
        if path.exists():
            report[name] = path.read_text().strip()

    def sample(engine, inlining, run, phase="compile", cache=None):
        label = f"{engine}-inline{inlining}-{run}-{phase}"
        command = [str(executable), "--engine", engine]
        if args.inline_budget is not None:
            command += ["--inline-budget", str(args.inline_budget)]
        if cache is not None:
            command += ["--cache", str(cache)]
        command += [str(wasm)]
        env = dict(os.environ, BLOT_INLINING=str(inlining))
        started = time.monotonic()
        with (out / f"{label}.stdout").open("wb") as stdout, (out / f"{label}.stderr").open("wb") as stderr:
            child = subprocess.Popen(command, env=env, stdout=stdout, stderr=stderr)
            _, status, usage = os.wait4(child.pid, 0)
            child.returncode = os.waitstatus_to_exitcode(status)
        row = {
            "engine": engine, "inlining": bool(inlining), "run": run, "phase": phase,
            "command": command, "wall_seconds": time.monotonic() - started,
            "user_seconds": usage.ru_utime, "system_seconds": usage.ru_stime,
            "peak_rss_kib": usage.ru_maxrss, "exit_code": child.returncode,
        }
        measured = re.search(r": ([0-9]+) ms", (out / f"{label}.stdout").read_text())
        if measured:
            row["reported_module_ms"] = int(measured.group(1))
        report["samples"].append(row)
        report["load_end"] = os.getloadavg()
        (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(row), flush=True)
        if child.returncode != 0:
            raise RuntimeError(f"{label} failed; see {out}")

    modes = [("plain", 1), ("plain", 0), ("incremental", 1), ("incremental", 0)]
    for run in range(args.runs):
        # Reverse each pair's order to reduce systematic warming/drift bias.
        for engine, inlining in modes[::(-1 if run % 2 else 1)]:
            sample(engine, inlining, run)
    for engine, inlining in modes:
        for run in range(args.runs):
            cache = out / f"cache-{engine}-inline{inlining}-{run}"
            sample(engine, inlining, run, "cache-populate", cache)
            sample(engine, inlining, run, "cache-load", cache)


if __name__ == "__main__":
    main()
