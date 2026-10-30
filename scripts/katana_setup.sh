#!/bin/bash
set -euo pipefail

source /etc/profile.d/modules.sh 2>/dev/null || true
module load python/3.11.3

cd /srv/scratch/z5531827/musicwhy
export HF_HOME=/srv/scratch/z5531827/.cache/huggingface
export PIP_CACHE_DIR=/srv/scratch/z5531827/.cache/pip
mkdir -p /srv/scratch/z5531827/.cache/huggingface
mkdir -p /srv/scratch/z5531827/.cache/pip
mkdir -p /srv/scratch/z5531827/.cache/numba

echo "Creating venv with Python 3.11..."
python3 -m venv .venv
source .venv/bin/activate
echo "Venv Python: $(python --version)"

pip install --upgrade pip wheel setuptools -q
echo "Installing core deps..."
pip install librosa soundfile numpy scipy pandas h5py pyyaml click rich tqdm joblib faiss-cpu datasets wbgapi fredapi requests pydantic -q 2>&1 | tail -5
echo "Installing musicwhy..."
pip install -e . -q 2>&1 | tail -3

echo "=== Verify ==="
python -c "import musicwhy; print('musicwhy v' + musicwhy.__version__)"
python -c "from musicwhy.config import load_config; cfg = load_config(None); print('Config OK: seed=' + str(cfg.project.seed))"
python -c "from musicwhy.utils import get_device; print('Device: ' + get_device())"
echo "Venv size: $(du -sh .venv | cut -f1)"
df -h /srv/scratch | tail -1
