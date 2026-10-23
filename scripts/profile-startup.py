#!/usr/bin/env python3
"""Measure source export and cold/warm headless startup in isolated caches.

RSS is the sampled sum of resident process sizes, including shared pages in each
process. It is not a unique physical-memory or live-allocation measurement.
Per-process CPU is sampled Linux ticks; wait4 separately records the completed
command's accounting. Headless packets exclude rendering and GPU asset upload.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import selectors
import signal
import subprocess
import time


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tree_pins(root, ignored_roots=()):
    pins = {}
    for directory, children, files in os.walk(root):
        if Path(directory) == root:
            children[:] = sorted(child for child in children if child not in ignored_roots)
        for name in sorted(files):
            path = Path(directory) / name
            pins[str(path.relative_to(root))] = digest(path)
    return pins


def witness(root):
    files = [root / "schema.txt", *sorted((root / "output").glob("*.f32"))]
    if len(files) < 2:
        raise RuntimeError(f"No packet witness in {root}")
    h = hashlib.sha256()
    for path in files:
        h.update(str(path.relative_to(root)).encode())
        h.update(b"\0")
        h.update(path.read_bytes())
    return {"sha256": h.hexdigest(), "files": len(files), "bytes": sum(p.stat().st_size for p in files)}


def measure(command, env, out, interval, timeout):
    out.mkdir()
    started = time.monotonic()
    child = subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
    selector = selectors.DefaultSelector()
    streams = {}
    for name, pipe in (("stdout", child.stdout), ("stderr", child.stderr)):
        os.set_blocking(pipe.fileno(), False)
        selector.register(pipe, selectors.EVENT_READ, name)
        streams[name] = (out / name).open("wb")
    samples, events, processes = [], [], {}
    hz, page = os.sysconf("SC_CLK_TCK"), os.sysconf("SC_PAGE_SIZE")
    completed = None
    finished = False
    last_sample = 0.0
    try:
        while selector.get_map() or completed is None:
            now = time.monotonic()
            if now - started > timeout:
                raise TimeoutError(f"Command exceeded {timeout}s: {command[0]}")
            if now - last_sample >= interval:
                last_sample = now
                stats = {}
                for entry in Path("/proc").iterdir():
                    if not entry.name.isdecimal():
                        continue
                    try:
                        text = (entry / "stat").read_text()
                        name, tail = text.rsplit(") ", 1)
                        fields = tail.split()
                        stats[int(entry.name)] = (int(fields[1]), name.split("(", 1)[1], int(fields[11]) + int(fields[12]), int(fields[19]), int(fields[21]) * page)
                    except (OSError, ValueError, IndexError):
                        pass
                owned = {child.pid}
                while True:
                    expanded = owned | {pid for pid, data in stats.items() if data[0] in owned}
                    if expanded == owned:
                        break
                    owned = expanded
                resident = 0
                observed = []
                for pid in sorted(owned & stats.keys()):
                    _, name, ticks, birth, rss = stats[pid]
                    resident += rss
                    key = f"{pid}:{birth}"
                    process = processes.setdefault(key, {"pid": pid, "name": name, "cpu_seconds_sampled": 0, "peak_rss_bytes_sampled": 0})
                    process["cpu_seconds_sampled"] = max(process["cpu_seconds_sampled"], ticks / hz)
                    process["peak_rss_bytes_sampled"] = max(process["peak_rss_bytes_sampled"], rss)
                    observed.append({"pid": pid, "cpu_ticks": ticks, "rss_bytes": rss})
                samples.append({"seconds": now - started, "rss_sum_bytes": resident, "processes": observed})
            for key, _ in selector.select(interval):
                data = os.read(key.fileobj.fileno(), 65536)
                if not data:
                    selector.unregister(key.fileobj)
                    continue
                streams[key.data].write(data)
                events.append({"seconds": time.monotonic() - started, "stream": key.data, "text": data.decode(errors="replace")})
            if completed is None:
                pid, status, usage = os.wait4(child.pid, os.WNOHANG)
                if pid:
                    child.returncode = os.waitstatus_to_exitcode(status)
                    completed = usage
        result = {"command": command, "wall_seconds": time.monotonic() - started, "exit_code": child.returncode,
                  "wait4_user_seconds": completed.ru_utime, "wait4_system_seconds": completed.ru_stime,
                  "wait4_maxrss_kib": completed.ru_maxrss, "peak_sampled_rss_sum_bytes": max((s["rss_sum_bytes"] for s in samples), default=0),
                  "processes": list(processes.values()), "events": events}
        (out / "samples.json").write_text(json.dumps(samples) + "\n")
        (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        if child.returncode:
            raise RuntimeError(f"Command failed ({child.returncode}); see {out}")
        finished = True
        return result
    finally:
        selector.close()
        for stream in streams.values():
            stream.close()
        for pipe in (child.stdout, child.stderr):
            pipe.close()
        if not finished:
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        if completed is None:
            child.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-project", type=Path, required=True)
    parser.add_argument("--candidate-project", type=Path, required=True)
    parser.add_argument("--pen", type=Path, default=Path("target/release/pen"))
    parser.add_argument("--baseline-pen", type=Path, help="compare this host against --pen using the respective projects")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--interval-ms", type=int, default=50)
    parser.add_argument("--timeout-seconds", type=int, default=120)
    args = parser.parse_args()
    if platform.system() != "Linux" or min(args.runs, args.frames, args.interval_ms, args.timeout_seconds) < 1:
        parser.error("Linux and positive measurement parameters are required")
    pen, out = args.pen.resolve(), args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    projects = {"baseline": args.baseline_project.resolve(), "candidate": args.candidate_project.resolve()}
    hosts = {"baseline": args.baseline_pen.resolve() if args.baseline_pen else pen, "candidate": pen}
    host_pins = {str(path): digest(path) for path in hosts.values()}
    # Project::written identifies .pen and target as generated output. Git
    # metadata is not an input; all actual sources and assets remain pinned.
    ignored_project_roots = (".git", ".pen", "target")
    pins = {name: tree_pins(root, ignored_project_roots) for name, root in projects.items()}
    compiler_pins, library_pins = {}, {}
    for root in projects.values():
        config = json.loads((root / "pen.json").read_text())
        compiler = Path(config["plugins"]["blot"]["run"]["blotc"]).resolve()
        compiler_pins[str(compiler)] = digest(compiler)
        library = Path(config["compiler"]["settings"]["std"]).resolve()
        library_pins[str(library)] = tree_pins(library)
    plugin_root = os.environ.get("PEN_PLUGINS")
    plugin_pins = {str(p): digest(p) for p in sorted(Path(plugin_root).glob("*.wasm"))} if plugin_root else {}
    report = {"started_at": datetime.now(timezone.utc).isoformat(), "pen_sha256": digest(pen), "script_sha256": digest(Path(__file__)),
              "platform": platform.platform(), "load_start": os.getloadavg(), "source_pins": pins, "interval_ms": args.interval_ms,
              "compiler_pins": compiler_pins, "library_pins": library_pins, "plugin_pins": plugin_pins, "host_pins": host_pins,
              "affinity": sorted(os.sched_getaffinity(0)), "cpu_count": os.cpu_count(),
              "environment": {key: os.environ.get(key) for key in ("BLOT_INLINING", "RAYON_NUM_THREADS")},
              "scope": __doc__, "ignored_project_roots": ignored_project_roots, "runs": args.runs, "frames": args.frames, "samples": []}
    for name in ("cpu.max", "memory.max"):
        path = Path('/sys/fs/cgroup') / name
        if path.exists():
            report[name] = path.read_text().strip()
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    expected_packets = None
    for run in range(args.runs):
        for variant in (["baseline", "candidate"] if run % 2 == 0 else ["candidate", "baseline"]):
            variant_pen = hosts[variant]
            root = out / f"{run}-{variant}"
            root.mkdir()
            export_env = dict(os.environ, XDG_CACHE_HOME=str(root / "export-cache"))
            runtime_env = dict(os.environ, XDG_CACHE_HOME=str(root / "runtime-cache"))
            cold, warm = root / "bundle-cold", root / "bundle-warm"
            phases = [
                ("export-cold", [str(variant_pen), "export", "--project", str(projects[variant]), "--out", str(cold)], export_env),
                ("export-warm", [str(variant_pen), "export", "--project", str(projects[variant]), "--out", str(warm)], export_env),
                ("packets-cold", [str(variant_pen), "packets", str(root / "packets-cold-output"), "--game", str(cold), "--frames", str(args.frames)], runtime_env),
                ("packets-warm", [str(variant_pen), "packets", str(root / "packets-warm-output"), "--game", str(cold), "--frames", str(args.frames)], runtime_env),
            ]
            for phase, command, environment in phases:
                result = measure(command, environment, root / phase, args.interval_ms / 1000, args.timeout_seconds)
                row = {"run": run, "variant": variant, "phase": phase, **result}
                cache = Path(environment["XDG_CACHE_HOME"])
                row["cache_root"] = str(cache)
                entries = [p.stat().st_size for p in cache.rglob("*") if p.is_file()]
                row["cache_files"] = len(entries)
                row["cache_bytes"] = sum(entries)
                row["cache_largest_bytes"] = max(entries, default=0)
                if phase == "export-warm":
                    if digest(cold / "game.wasm") != digest(warm / "game.wasm"):
                        raise RuntimeError("Warm export changed the Wasm")
                if phase.startswith("packets-"):
                    row["witness"] = witness(root / f"{phase}-output")
                    if expected_packets is None:
                        expected_packets = row["witness"]
                    if row["witness"] != expected_packets:
                        raise RuntimeError("Packet or schema bytes changed across the paired startup runs")
                report["samples"].append(row)
                (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
                print(json.dumps({key: row[key] for key in ("run", "variant", "phase", "wall_seconds", "wait4_user_seconds", "wait4_system_seconds", "wait4_maxrss_kib", "peak_sampled_rss_sum_bytes")}), flush=True)
    for name, root in projects.items():
        if tree_pins(root, ignored_project_roots) != pins[name]:
            raise RuntimeError(f"Project changed during profiling: {name}")
    if digest(pen) != report["pen_sha256"]:
        raise RuntimeError("Pen executable changed during profiling")
    for path, expected in {**compiler_pins, **plugin_pins, **host_pins}.items():
        if digest(Path(path)) != expected:
            raise RuntimeError(f"Executable changed during profiling: {path}")
    for path, expected in library_pins.items():
        if tree_pins(Path(path)) != expected:
            raise RuntimeError(f"Library changed during profiling: {path}")


if __name__ == "__main__":
    main()
