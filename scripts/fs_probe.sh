#!/bin/bash
# Time each filesystem step Python's venv startup touches; run on a compute node to locate GPFS stalls.
# Usage: PBS_JOBID=<job on node> ssh <node> bash /srv/scratch/z5531827/musicwhy/scripts/fs_probe.sh
R=/srv/scratch/z5531827/musicwhy
SP=$R/.venv/lib/python3.11/site-packages
probe() {
    local start end rc
    start=$(date +%s%N)
    timeout -k 2 15 "$@" > /dev/null 2>&1
    rc=$?
    end=$(date +%s%N)
    printf '%-70s rc=%-3s %6d ms\n' "$*" "$rc" $(( (end - start) / 1000000 ))
}
echo "node $(hostname)"
probe stat "$R"
probe cat "$R/.venv/pyvenv.cfg"
probe ls "$SP"
probe cat "$SP/distutils-precedence.pth"
probe cat "$SP/__editable__.musicwhy-0.1.0.pth"
probe ls "$R"
probe ls "$R/musicwhy"
probe ls "$R/data"
probe stat /srv/scratch/z5531827
echo "editable pth: $(cat "$SP/__editable__.musicwhy-0.1.0.pth" 2>/dev/null)"
PY=$R/.venv/bin/python
echo "--- interpreter startup (rc=124 = hung):"
echo "venv home: $(grep '^home' "$R/.venv/pyvenv.cfg")"
probe "$PY" -S -c pass
probe "$PY" -I -c pass
probe "$PY" -c pass
probe "$PY" -I -c 'import json, csv'
probe "$PY" -I -c 'import numpy'
echo "--- file reads that site.py startup performs (slow or rc=124 = blocked):"
for f in "$SP"/__editable___musicwhy_0_1_0_finder.py "$SP"/__pycache__/* \
         "$SP"/_distutils_hack/*.py "$SP"/_distutils_hack/__pycache__/*; do
    [ -e "$f" ] && probe cat "$f"
done 2>/dev/null | awk '$NF+0 > 50 || /rc=[1-9]/ {print; bad=1} END {if (!bad) print "all fast (<50 ms)"}'
