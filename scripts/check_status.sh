#!/bin/bash
# Quick status check for MusicWhy pipeline — called by monitoring loop.
# Outputs structured lines for easy parsing.

source /srv/scratch/z5531827/musicwhy/scripts/_env.sh 2>/dev/null

echo "=== JOBS ==="
qstat -u z5531827 2>/dev/null || echo "NO_JOBS"

echo "=== MUSDB18 ==="
ZIP=/srv/scratch/z5531827/musicwhy/data/raw/musdb18hq/musdb18hq.zip
DONE=/srv/scratch/z5531827/musicwhy/data/raw/musdb18hq/.download_complete
if [ -f "$DONE" ]; then
    echo "STATUS=complete"
    cat "$DONE"
elif [ -f "$ZIP" ]; then
    SIZE=$(stat --format=%s "$ZIP" 2>/dev/null || echo 0)
    echo "STATUS=downloading SIZE=$SIZE TARGET=22656664047"
else
    echo "STATUS=not_started"
fi

echo "=== WORLD_CONTEXT ==="
if [ -f "${DATA_ROOT}/world_context/world_context.db" ]; then
    SIZE=$(stat --format=%s "${DATA_ROOT}/world_context/world_context.db" 2>/dev/null)
    echo "STATUS=exists SIZE=$SIZE"
else
    echo "STATUS=missing"
fi

echo "=== MANIFEST ==="
if [ -f "${DATA_ROOT}/manifest.csv" ]; then
    LINES=$(wc -l < "${DATA_ROOT}/manifest.csv")
    echo "STATUS=exists TRACKS=$LINES"
else
    echo "STATUS=missing"
fi

echo "=== FEATURES ==="
H5_COUNT=$(ls ${DATA_ROOT}/features/*.h5 2>/dev/null | wc -l)
FEAT_DONE=${DATA_ROOT}/features/.extract_done
echo "H5_COUNT=$H5_COUNT DONE=$([ -f $FEAT_DONE ] && echo yes || echo no)"

echo "=== STEMS ==="
STEM_COUNT=$(ls -d ${DATA_ROOT}/stems/*/ 2>/dev/null | wc -l)
SEP_DONE=${DATA_ROOT}/stems/.separate_done
echo "STEM_COUNT=$STEM_COUNT DONE=$([ -f $SEP_DONE ] && echo yes || echo no)"

echo "=== GPU_DEPS ==="
${PYTHON} -c "import torch; print('TORCH='+torch.__version__, 'CUDA='+str(torch.cuda.is_available()))" 2>/dev/null || echo "TORCH=missing"

echo "=== DISK ==="
du -sh /srv/scratch/z5531827/musicwhy/ 2>/dev/null
du -sh /srv/scratch/z5531827/ 2>/dev/null

echo "=== END ==="
