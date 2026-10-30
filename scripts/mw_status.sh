#!/bin/bash
# Pipeline status for scripts/monitor.py, one fact per line.
# No timestamps on JOB/COUNT lines, so identical state prints identically between polls.
D=/srv/scratch/z5531827/musicwhy/data

for id in $(qstat -u z5531827 2>/dev/null | awk '/^[0-9]+\./ {split($1, a, "."); print a[1]}'); do
    qstat -f "$id" 2>/dev/null | awk -v id="$id" '
        /Job_Name =/ {name = $3}
        /job_state =/ {state = $3}
        /exec_host =/ {split($3, h, "/"); host = h[1]}
        END {print "JOB", id, name, state, host}'
done

count() { ls $1 2>/dev/null | wc -l; }
echo "COUNT annotations=$(count "$D/annotations/*.json") stems=$(count "$D/features/stems/*.json")" \
     "features=$(count "$D/features/*.h5") musdb=$(count "$D/raw/musdb18hq/*/*/mixture.wav")" \
     "moisesdb=$(count "$D/raw/moisesdb/tracks/*/meta.json")"

for f in benchmark/musicwhy_benchmark.json results/eval_claude_text.json results/eval_claude_text_open.json \
         results/eval_qwen2audio.json embeddings/mert_embeddings.npz embeddings/similar_pairs.json; do
    [ -f "$D/$f" ] && echo "FILE $f $(stat -c %Y "$D/$f")"
done
exit 0
