#!/usr/bin/env bash
# Install GPU deps inside a GPU job (needs CUDA toolkit visible).
# Run as: qsub -l walltime=02:00:00 -l select=1:ncpus=8:mem=48gb:ngpus=1 -I
# Then: bash scripts/install_gpu_deps.sh
set -euo pipefail

source /srv/scratch/z5531827/musicwhy/scripts/_env.sh

echo "=== Installing GPU dependencies ==="
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"

pip install \
    torch \
    torchaudio \
    demucs \
    transformers \
    accelerate

# Verify
${PYTHON} -c "
import torch
print(f'PyTorch {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GB')
"

echo "=== GPU deps installed ==="
