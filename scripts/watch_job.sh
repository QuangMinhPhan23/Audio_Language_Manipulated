#!/usr/bin/env bash
# Watch a PBS job with stall detection.
# Usage: watch_job.sh <jobid> <artifact_path> [poll_s] [max_stall_polls]
set -euo pipefail

JOBID=${1:?Usage: watch_job.sh <jobid> <artifact_path> [poll_s] [max_stall_polls]}
ARTIFACT=${2:?Need artifact path to monitor}
POLL=${3:-60}
MAX_STALL=${4:-5}

stall_count=0
prev_cput="none"

echo "Watching job ${JOBID}, polling every ${POLL}s, stall limit ${MAX_STALL}"

while true; do
    state=$(qstat -f "${JOBID}" 2>/dev/null | grep "job_state" | awk '{print $3}' || true)
    cput=$(qstat -f "${JOBID}" 2>/dev/null | grep "resources_used.cput" | awk '{print $3}' || true)
    node=$(qstat -f "${JOBID}" 2>/dev/null | grep "exec_host" | awk '{print $3}' | cut -d/ -f1 || true)

    if [ -z "${state}" ] || [ "${state}" = "F" ]; then
        echo "[$(date +%H:%M:%S)] Job ${JOBID} finished."
        if [ -e "${ARTIFACT}" ]; then
            echo "Artifact exists: $(ls -lh ${ARTIFACT} | tail -1)"
        else
            echo "WARNING: Artifact ${ARTIFACT} not found!"
        fi
        break
    fi

    # Stall detection
    if [ "${state}" = "R" ] && [ "${cput}" = "${prev_cput}" ] && [ "${prev_cput}" != "none" ]; then
        stall_count=$((stall_count + 1))
        echo "[$(date +%H:%M:%S)] state=${state} cput=${cput} node=${node} STALL ${stall_count}/${MAX_STALL}"
        if [ "${stall_count}" -ge "${MAX_STALL}" ]; then
            echo "STALL LIMIT REACHED on node ${node}. Consider qdel ${JOBID} and resubmit."
            break
        fi
    else
        stall_count=0
        echo "[$(date +%H:%M:%S)] state=${state} cput=${cput} node=${node}"
    fi

    prev_cput="${cput}"
    sleep ${POLL}
done
