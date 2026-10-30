#!/usr/bin/env bash
# One-time setup on Katana. Run from login node.
set -euo pipefail

SCRATCH=/srv/scratch/z5531827
REPO=${SCRATCH}/musicwhy

echo "=== Setting up MusicWhy on Katana ==="

# Cache dirs on scratch (not home!)
mkdir -p ${SCRATCH}/.cache/{huggingface,pip,numba}
export HF_HOME=${SCRATCH}/.cache/huggingface
export PIP_CACHE_DIR=${SCRATCH}/.cache/pip

# Create venv on scratch
if [ ! -d "${REPO}/.venv" ]; then
    echo "Creating venv..."
    python3 -m venv ${REPO}/.venv
fi
source ${REPO}/.venv/bin/activate

# Install core deps (no GPU deps yet — those go in a GPU job)
echo "Installing core dependencies..."
pip install --upgrade pip wheel setuptools
pip install \
    librosa==1.0.0 \
    soundfile \
    numpy \
    scipy \
    pandas \
    h5py \
    pyyaml \
    click \
    rich \
    tqdm \
    joblib \
    faiss-cpu \
    datasets \
    wbgapi \
    fredapi \
    requests \
    pydantic

# Install the package itself
cd ${REPO}
pip install -e .

echo "=== Core setup done ==="
echo "Venv: ${REPO}/.venv"
echo "Python: $(which python) ($(python --version))"
echo "Disk usage: $(du -sh ${REPO}/.venv | cut -f1)"
