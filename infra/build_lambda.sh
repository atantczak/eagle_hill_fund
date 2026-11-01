#!/usr/bin/env bash
# Build the Lambda deployment zip WITHOUT Docker: pure-Python package + manylinux
# aarch64 wheels downloaded by pip for the Lambda runtime (python3.11, arm64).
#
#   infra/build_lambda.sh            -> infra/build/lambda.zip
#
# Only the modules the job imports are shipped (reporting, signals, complexity).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="${ROOT:?}/infra/build"
STAGE="${BUILD:?}/lambda"
ZIP="${BUILD:?}/lambda.zip"
PY="${PYTHON:-${ROOT:?}/.venv311/bin/python}"
PY_VERSION="3.11"
PLATFORM="manylinux2014_aarch64"

rm -rf "${STAGE:?}" "${ZIP:?}"
mkdir -p "${STAGE:?}"

echo "==> installing wheels for cp${PY_VERSION/./} / $PLATFORM"
"$PY" -m pip install \
  --quiet --disable-pip-version-check \
  --platform "$PLATFORM" --implementation cp --python-version "$PY_VERSION" \
  --only-binary=:all: --upgrade \
  --target "${STAGE:?}" \
  "pandas>=2.2,<3.1" "numpy>=1.26,<3" "python-dateutil>=2.8,<3" "requests>=2.31" "anthropic>=0.60"

echo "==> copying ehf package subset"
for d in ehf ehf/server ehf/server/research; do
  mkdir -p "${STAGE:?}/$d" && cp "$ROOT/$d/__init__.py" "${STAGE:?}/$d/__init__.py"
done
cp -R "$ROOT/ehf/server/research/reporting"  "${STAGE:?}/ehf/server/research/reporting"
cp -R "$ROOT/ehf/server/research/signals"    "${STAGE:?}/ehf/server/research/signals"
cp -R "$ROOT/ehf/server/research/complexity" "${STAGE:?}/ehf/server/research/complexity"

echo "==> pruning"
find "${STAGE:?}" -type d \( -name "__pycache__" -o -name "tests" -o -name "test" -o -name "*.dist-info" \) -prune -exec rm -rf {} + 2>/dev/null || true
find "${STAGE:?}" -type f \( -name "*.pyc" -o -name "*.ipynb" \) -delete
# boto3/botocore ship with the runtime; don't bundle them if a dependency pulled them in.
for extra in boto3 botocore bin; do
  [ -e "${STAGE:?}/$extra" ] && rm -rf "${STAGE:?}/${extra:?}"
done

echo "==> zipping"
( cd "${STAGE:?}" && zip -qr -X "${ZIP:?}" . )
du -sh "${STAGE:?}" "${ZIP:?}"
echo "built ${ZIP}"
