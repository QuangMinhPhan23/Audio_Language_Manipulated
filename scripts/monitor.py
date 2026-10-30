"""Event-only Katana pipeline monitor, meant to run under the Monitor tool.

Prints a line only when something changes: job state, a finished job (with its exit status and
validation result), output progress, a stall, or ssh trouble. Exits when no jobs remain.
Stalls are judged by outputs, not CPU time: API-bound jobs sit near 0 CPU while working.

Usage (from WSL): python3 /mnt/d/WMM/scripts/monitor.py [poll_seconds=120] [stall_polls=5] [heartbeat_min=0]
heartbeat_min > 0 also prints a one-line STATUS summary at that interval.
"""
import subprocess
import sys
import time

POLL = int(sys.argv[1]) if len(sys.argv) > 1 else 120
STALL_POLLS = int(sys.argv[2]) if len(sys.argv) > 2 else 5
HEARTBEAT_S = int(sys.argv[3]) * 60 if len(sys.argv) > 3 else 0
PROGRESS_STEP = 25
EARLY_VALIDATE_AT = 5
REMOTE = "/srv/scratch/z5531827/musicwhy"
CHECKS_FOR_JOB = {
    "mw_metadata": "manifest",
    "mw_manifest": "manifest",
    "mw_world_ctx": "world",
    "mw_stems": "stems",
    "mw_annotate": "annotations",
    "mw_benchmark": "benchmark",
    "mw_eval": "eval eval_open",
    "mw_eval_qwen2": "qwen",
    "mw_embeddings": "pairs",
    "mw_p0_names": "phase0",
    "mw_p0_genver": "phase0",
    "mw_p0_calib": "phase0_calib",
}
JOB_FOR_COUNT = {"annotations": "mw_annotate", "stems": "mw_stems"}


def say(msg):
    print(f"[{time.strftime('%H:%M')}] {msg}", flush=True)


def ssh(cmd, timeout=150):
    try:
        r = subprocess.run(
            ["ssh", "-o", "ConnectTimeout=20", "-o", "BatchMode=yes", "katana", cmd],
            capture_output=True, text=True, timeout=timeout,
        )
        return r.returncode, r.stdout
    except subprocess.TimeoutExpired:
        return 255, ""


def poll():
    rc, out = ssh(f"bash {REMOTE}/scripts/mw_status.sh")
    if rc != 0 or "COUNT" not in out:
        return None
    jobs, counts, files = {}, {}, {}
    for line in out.splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "JOB" and len(parts) >= 4:
            jobs[parts[1]] = {"name": parts[2], "state": parts[3], "host": parts[4] if len(parts) > 4 else ""}
        elif parts[0] == "COUNT":
            counts = {k: int(v) for k, v in (p.split("=") for p in parts[1:])}
        elif parts[0] == "FILE" and len(parts) == 3:
            files[parts[1]] = parts[2]
    return jobs, counts, files


def validate(checks):
    rc, out = ssh(f"cd {REMOTE} && .venv/bin/python -I scripts/validate.py {checks}")
    problems = "; ".join(l for l in out.splitlines() if l.startswith(("FAIL", "WARN")))
    if rc == 0:
        say(f"VALIDATE {checks}: ok" + (f" (warnings: {problems})" if problems else ""))
    else:
        say(f"VALIDATE {checks}: FAILED - {problems or out.strip()[-300:]}")


def exit_status(job_id):
    _, out = ssh(f"grep -h 'Exit Status' {REMOTE}/*.o{job_id} 2>/dev/null | tail -1; "
                 f"grep -hE 'FAILED|Traceback|Error|bouncing|Requeuing|Resubmitted' {REMOTE}/*.o{job_id} 2>/dev/null | tail -2")
    return " | ".join(l.strip() for l in out.splitlines() if l.strip()) or "no output file yet"


def bounced(status):
    """Bounced or deleted (qdel gives exit 271): nothing meaningful to validate."""
    return any(word in status for word in ("bouncing", "Requeuing", "Resubmitted", "Termination request"))


def main():
    prev_jobs, prev_counts, prev_files = None, {}, {}
    ssh_failures, stall_polls, stalled = 0, 0, False
    early_validated = set()
    last_heartbeat = time.time()
    while True:
        state = poll()
        if state is None:
            ssh_failures += 1
            if ssh_failures == 3:
                say("SSH_FAIL: 3 polls in a row could not reach Katana")
            time.sleep(POLL)
            continue
        if ssh_failures >= 3:
            say("SSH_OK: reachable again")
        ssh_failures = 0
        jobs, counts, files = state

        if prev_jobs is None:
            summary = ", ".join(f"{i} {j['name']} {j['state']}" for i, j in jobs.items()) or "no jobs"
            say(f"START {summary} | " + " ".join(f"{k}={v}" for k, v in counts.items()))
            if not jobs:
                return
            prev_jobs, prev_counts, prev_files = jobs, counts, files
            time.sleep(POLL)
            continue

        for job_id, job in jobs.items():
            old = prev_jobs.get(job_id)
            if old is None or old["state"] != job["state"]:
                before = f"{old['state']} -> " if old else ""
                say(f"JOB {job_id} {job['name']} {before}{job['state']} {job['host']}".rstrip())
        for job_id, job in prev_jobs.items():
            if job_id not in jobs:
                status = exit_status(job_id)
                say(f"DONE {job_id} {job['name']}: {status}")
                ran = status != "no output file yet"
                if job["name"] in CHECKS_FOR_JOB and ran and not bounced(status):
                    validate(CHECKS_FOR_JOB[job["name"]])

        for key, value in counts.items():
            old = prev_counts.get(key, 0)
            if value // PROGRESS_STEP > old // PROGRESS_STEP:
                say(f"PROGRESS {key}={value}")
            producer = JOB_FOR_COUNT.get(key)
            running = any(j["name"] == producer and j["state"] == "R" for j in jobs.values())
            if running and value >= EARLY_VALIDATE_AT and key not in early_validated:
                early_validated.add(key)
                validate(key)
        for path in files:
            if path not in prev_files:
                say(f"NEW FILE {path}")

        progressed = counts != prev_counts or files != prev_files
        running = {i: j for i, j in jobs.items() if j["state"] == "R"}
        if progressed:
            if stalled:
                say("RESUMED: outputs advancing again")
            stall_polls, stalled = 0, False
        elif running:
            stall_polls += 1
            if stall_polls == STALL_POLLS:
                stalled = True
                where = ", ".join(f"{i} {j['name']}@{j['host']}" for i, j in running.items())
                say(f"STALL: no new outputs for {STALL_POLLS * POLL // 60} min while running: {where}")

        if HEARTBEAT_S and time.time() - last_heartbeat >= HEARTBEAT_S:
            last_heartbeat = time.time()
            summary = ", ".join(f"{j['name']} {j['state']}" for j in jobs.values()) or "no jobs"
            say(f"STATUS {summary} | " + " ".join(f"{k}={v}" for k, v in counts.items()))

        prev_jobs, prev_counts, prev_files = jobs, counts, files
        if not jobs:
            say("IDLE: all jobs finished")
            return
        time.sleep(POLL)


if __name__ == "__main__":
    main()
