#!/usr/bin/env bash
# Cluster-wide GPU count: total / assigned / free.
pbsnodes -av 2>/dev/null | awk '
/resources_available.ngpus/ { total += int($3) }
/resources_assigned.ngpus/  { used  += int($3) }
END { printf "total=%d  assigned=%d  free=%d\n", total, used, total-used }
'
