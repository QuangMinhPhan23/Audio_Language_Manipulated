#!/bin/bash
# Narrow down a hanging `import numpy` on a compute node: shared-library reads vs OpenBLAS thread start.
R=/srv/scratch/z5531827/musicwhy
PY=$R/.venv/bin/python
SP=$R/.venv/lib/python3.11/site-packages
t() {
    local start end rc
    start=$(date +%s%N)
    env "$@" > /tmp/np_probe.out 2>&1
    rc=$?
    end=$(date +%s%N)
    printf '%-70s rc=%-3s %6d ms\n' "$(echo "$*" | cut -c1-70)" "$rc" $(( (end - start) / 1000000 ))
}
echo "node $(hostname), nproc=$(nproc), cpus on node=$(grep -c ^processor /proc/cpuinfo)"
ls -la "$SP"/numpy.libs/ 2>/dev/null | awk 'NR>1 {print "  lib:", $5, $9}'
for lib in "$SP"/numpy.libs/*.so*; do t timeout -k 2 20 cat "$lib"; done
t OPENBLAS_NUM_THREADS=1 timeout -k 2 20 "$PY" -I -c 'import numpy'
t timeout -k 2 20 "$PY" -I -c 'import numpy'
t OPENBLAS_NUM_THREADS=1 timeout -k 2 20 "$PY" -I -c 'import numpy'
timeout -k 2 20 "$PY" -I -X importtime -c 'import numpy' 2>&1 | tail -3
