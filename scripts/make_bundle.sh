#!/bin/bash
# Pack what the CPU jobs (Phases 3-5 text) import into tarballs on scratch. Run on the login node
# (deploy.sh does). Jobs copy these files to node-local /tmp (mw_stage_bundle in _env.sh) instead of
# opening hundreds of module files on GPFS, where a cold first open can block for minutes.
#   sp_bundle.tar    third-party packages (numpy, h5py, anthropic, ...); built only when missing or
#                    with --full, since reading ~9000 venv files can itself take minutes
#   repo_bundle.tar  the musicwhy package and scripts/*.py; rebuilt on every deploy (small)
set -euo pipefail
R=/srv/scratch/z5531827/musicwhy
SP=.venv/lib/python3.11/site-packages
SP_TAR=${R}/bundle/sp_bundle.tar
REPO_TAR=${R}/bundle/repo_bundle.tar
PKGS="aiohappyeyeballs aiohttp aiosignal annotated_types anthropic anyio attr attrs docstring_parser
      frozenlist h11 h5py h5py.libs httpcore2 httpx2 idna jiter multidict numpy numpy.libs opentelemetry
      propcache pydantic pydantic_core sniffio truststore typing_extensions.py typing_inspection yarl
      soundfile.py _soundfile.py _soundfile_data cffi _cffi_backend.cpython-311-x86_64-linux-gnu.so pycparser
      requests urllib3 certifi charset_normalizer tqdm click pandas dateutil six.py
      librosa scipy scipy.libs numba llvmlite sklearn scikit_learn.libs joblib soxr audioread lazy_loader
      decorator msgpack pooch platformdirs threadpoolctl.py packaging cloudpickle narwhals"

mkdir -p "${R}/bundle"
if [ "${1:-}" = "--full" ] || [ ! -f "${SP_TAR}" ]; then
    cd "${R}/${SP}"
    present=$(for p in ${PKGS}; do [ -e "$p" ] && echo "$p"; done)
    tar -cf "${SP_TAR}.tmp" --transform 's,^,sp/,' ${present} *.dist-info
    mv "${SP_TAR}.tmp" "${SP_TAR}"
    echo "sp bundle: $(du -h "${SP_TAR}" | cut -f1)"
fi

cd "${R}"
tar -cf "${REPO_TAR}.tmp" --exclude=__pycache__ --transform 's,^,repo/,' musicwhy scripts/*.py
mv "${REPO_TAR}.tmp" "${REPO_TAR}"
echo "repo bundle: $(du -h "${REPO_TAR}" | cut -f1)"
