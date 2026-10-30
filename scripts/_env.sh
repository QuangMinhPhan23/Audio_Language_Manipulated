#!/usr/bin/env bash
# MusicWhy environment variables — sourced by every PBS script.
# Keep this file on scratch next to the repo.

# Load Python 3.11 (system default is 3.6)
source /etc/profile.d/modules.sh 2>/dev/null || true
module load python/3.11.3 2>/dev/null || true

export REPO=/srv/scratch/z5531827/musicwhy
export PYTHON=${REPO}/.venv/bin/python
export DATA_ROOT=${REPO}/data
export HF_HOME=/srv/scratch/z5531827/.cache/huggingface
export PIP_CACHE_DIR=/srv/scratch/z5531827/.cache/pip
export TRANSFORMERS_CACHE=${HF_HOME}/hub
export NUMBA_CACHE_DIR=/srv/scratch/z5531827/.cache/numba
export TORCH_HOME=/srv/scratch/z5531827/.cache/torch

# Phase 0 (PHASE0_stimuli_pipeline.md): ACE-Step has its own venv (it pins torch 2.10 / transformers <4.58);
# the verify venv overlays the main one (jobs/phase0/install_verify.pbs)
export ACESTEP_ROOT=/srv/scratch/z5531827/ACE-Step-1.5
export ACESTEP_CHECKPOINTS_DIR=/srv/scratch/z5531827/.cache/acestep_checkpoints
export ACESTEP_PYTHON=${ACESTEP_ROOT}/.venv/bin/python
export VERIFY_PYTHON=${REPO}/.venv_verify/bin/python
export PHASE0_MODELS=/srv/scratch/z5531827/.cache/phase0_models

# Activate venv
source ${REPO}/.venv/bin/activate

# Unbuffered so PBS spool shows live progress; no .pyc writes so jobs don't contend on shared scratch
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1

# API keys live outside the repo (chmod 600) so deploys and git never touch them
if [ -f "${HOME}/.musicwhy_secrets" ]; then
    source "${HOME}/.musicwhy_secrets"
else
    echo "WARNING: ${HOME}/.musicwhy_secrets missing; ANTHROPIC_API_KEY not set" >&2
fi

# Stage the code bundle (scripts/make_bundle.sh) to node-local /tmp. Cold compute nodes can block for
# minutes on each first file open on scratch, so jobs import from this local copy instead.
# Run Python with -S and cd "${CODE_ROOT}" afterwards so nothing is imported from the scratch venv.
mw_stage_bundle() {
    MW_LOCAL=/tmp/mw_bundle_${PBS_JOBID%%.*}
    mkdir -p "${MW_LOCAL}"
    local t0
    t0=$(date +%s)
    tar -xf "${REPO}/bundle/sp_bundle.tar" -C "${MW_LOCAL}" || { echo "sp bundle staging failed"; return 1; }
    tar -xf "${REPO}/bundle/repo_bundle.tar" -C "${MW_LOCAL}" || { echo "repo bundle staging failed"; return 1; }
    echo "Staged code bundle to ${MW_LOCAL} in $(( $(date +%s) - t0 ))s"
    export MW_LOCAL CODE_ROOT="${MW_LOCAL}/repo" PYTHONPATH="${MW_LOCAL}/sp:${MW_LOCAL}/repo"
    trap 'rm -rf "${MW_LOCAL}"' EXIT
}

# Ensure data directories exist
mkdir -p ${DATA_ROOT}/{raw,features,stems,annotations,world_context,benchmark,results}
