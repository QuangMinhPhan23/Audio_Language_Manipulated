#!/usr/bin/env bash
# Push local code to Katana: only changed files are written (checksum rsync, atomic rename),
# so jobs importing the package mid-deploy never see a half-written file.
# Usage (from WSL): bash /mnt/d/WMM/scripts/deploy.sh
set -euo pipefail

LOCAL=/mnt/d/WMM
REMOTE=/srv/scratch/z5531827/musicwhy
SSH="ssh -o ConnectTimeout=20"

timeout 60 ${SSH} katana 'test -f $HOME/.musicwhy_secrets' \
    || { echo "ABORT: ~/.musicwhy_secrets missing on Katana"; exit 1; }

cd "${LOCAL}"
CHANGED=$(timeout 180 rsync -rlc --out-format='%n' -e "${SSH}" \
    --exclude='__pycache__' --exclude='*.pyc' \
    musicwhy jobs scripts configs tests docs pyproject.toml "katana:${REMOTE}/" | grep -v '/$' || true)

timeout 120 ${SSH} katana "
    cd ${REMOTE}
    find jobs scripts -type f \\( -name '*.pbs' -o -name '*.sh' -o -name '*.py' \\) -exec sed -i 's/\\r\$//' {} +
    for f in jobs/*/*.pbs scripts/*.sh; do bash -n \"\$f\" || echo \"SYNTAX ERROR: \$f\"; done
    bash scripts/make_bundle.sh
"
echo "deployed $(printf '%s\n' "${CHANGED}" | grep -c . || true) changed files:"
printf '%s\n' "${CHANGED}" | sed 's/^/  /'
