#!/usr/bin/env python3
"""Pair Wasmtime budgets or two Wasm modules on deterministic game frames.

Build the memory_images example first. Frame timers exclude module compilation;
OS CPU/RSS cover the whole process, including compilation. Each sample starts a
fresh process and checks the complete packet trace and final snapshot hashes.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import time


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wasm", type=Path)
    parser.add_argument("--baseline", type=Path, help="alternate with this module and require identical traces and snapshots")
    parser.add_argument("--executable", type=Path, default=Path("target/release/examples/memory_images"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=7)
    parser.add_argument("--frames", type=int, default=600)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--budgets", default="default,128,256")
    args = parser.parse_args()
    if platform.system() != "Linux":
        parser.error("OS RSS and child-process accounting require Linux")
    if args.runs < 1 or args.frames < 2 or not 0 <= args.warmup < args.frames:
        parser.error("positive runs, frames >= 2 and 0 <= warmup < frames are required")
    try:
        budgets = [None if item == "default" else int(item) for item in args.budgets.split(",")]
        assert budgets and len(set(budgets)) == len(budgets)
        assert all(value is None or 0 <= value <= 0xFFFFFFFF for value in budgets)
    except (ValueError, AssertionError):
        parser.error("--budgets requires unique U32 integers and/or default")
    wasm, executable, out = args.wasm.resolve(), args.executable.resolve(), args.out.resolve()
    variants = [("baseline", args.baseline.resolve()), ("candidate", wasm)] if args.baseline else [("candidate", wasm)]
    out.mkdir(parents=True, exist_ok=False)
    report = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "wasm_sha256": digest(wasm), "executable_sha256": digest(executable),
        "modules": {name: {"path": str(path), "sha256": digest(path)} for name, path in variants},
        "platform": platform.platform(), "affinity": sorted(os.sched_getaffinity(0)),
        "load_start": os.getloadavg(), "frames": args.frames, "warmup": args.warmup,
        "budgets": budgets, "runs": args.runs, "samples": [],
    }
    for name in ("cpu.max", "memory.max"):
        path = Path("/sys/fs/cgroup") / name
        if path.exists():
            report[name] = path.read_text().strip()
    expected = None
    for run in range(args.runs):
        # Rotate the first budget; reverse alternate cycles to limit order bias.
        order = budgets[run % len(budgets):] + budgets[:run % len(budgets)]
        if run // len(budgets) % 2:
            order.reverse()
        variant_order = variants if run % 2 == 0 else list(reversed(variants))
        schedule = [(budget, name, path) for budget in order for name, path in variant_order]
        for budget, variant, module in schedule:
            prefix = f"{run}-{variant}" if args.baseline else str(run)
            label = f"{prefix}-{budget if budget is not None else 'default'}"
            command = [str(executable), str(module), str(args.frames)]
            if budget is not None:
                command.append(str(budget))
            started = time.monotonic()
            with (out / (label + ".stdout")).open("wb") as stdout, (out / (label + ".stderr")).open("wb") as stderr:
                child = subprocess.Popen(command, env=dict(os.environ, BLOT_INLINING="1"), stdout=stdout, stderr=stderr)
                _, status, usage = os.wait4(child.pid, 0)
                child.returncode = os.waitstatus_to_exitcode(status)
            if child.returncode:
                raise RuntimeError(f"{label} exited {child.returncode}; see {out}")
            rows = [line[7:] for line in (out / (label + ".stdout")).read_text().splitlines() if line.startswith("bench: ")]
            if len(rows) != 1:
                raise RuntimeError(f"{label} emitted {len(rows)} benchmark records")
            data = json.loads(rows[0])
            timings = data["frame_ns"]
            assert len(timings) == args.frames and all(isinstance(value, int) and value >= 0 for value in timings)
            identity = data["trace"], data["snapshot"]
            if expected is None:
                expected = identity
            if identity != expected:
                raise RuntimeError(f"{label}: packet trace or final snapshot differs")
            measured = sorted(timings[args.warmup:])
            row = {"run": run, "variant": variant, "inline_budget": budget, "wall_seconds": time.monotonic() - started,
                   "cpu_seconds": usage.ru_utime + usage.ru_stime, "peak_rss_kib": usage.ru_maxrss,
                   **data, "p50_ms": statistics.median(measured) / 1e6,
                   "p95_ms": measured[int((len(measured) - 1) * .95)] / 1e6,
                   "p99_ms": measured[int((len(measured) - 1) * .99)] / 1e6}
            report["samples"].append(row)
            (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps({key: value for key, value in row.items() if key != "frame_ns"}), flush=True)
    assert digest(wasm) == report["wasm_sha256"] and digest(executable) == report["executable_sha256"]
    for name, path in variants:
        assert digest(path) == report["modules"][name]["sha256"]


if __name__ == "__main__":
    main()
