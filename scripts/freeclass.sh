#!/usr/bin/env bash
# Show nodes with genuinely free GPUs plus their memory class.
pbsnodes -av 2>/dev/null | awk '
/^[a-z]/ { node=$1 }
/resources_available.ngpus/ { total=int($3) }
/resources_assigned.ngpus/  { used=int($3) }
/resources_available.mem_per_gpu/ { memclass=$3 }
/resources_available.ncpus/ { tcpu=int($3) }
/resources_assigned.ncpus/  { ucpu=int($3) }
/state = free/ {
    free = total - used
    fcpu = tcpu - ucpu
    if (free > 0 && fcpu >= 4)
        printf "%s  gpu_free=%d  cpu_free=%d  %s\n", node, free, fcpu, memclass
}
' | sort
