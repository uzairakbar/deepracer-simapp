#!/usr/bin/env bash
###############################################################################
# Build uzairakbar/deepracer:v1 (CPU-only, amd64) from source.
#
# 1. Builds the modern CPU deepracer-simapp base image, which compiles the
#    bundle/ source (including the pure-simulation markov edits on this branch).
# 2. Layers the ZMQ sim-only overlay (docker/Dockerfile.zmqsim) on top.
#
# Usage: ./build-zmqsim.sh [-s]    (-s skips the base build if already present)
###############################################################################
set -euo pipefail
cd "$(dirname "$0")"

PREFIX="awsdeepracercommunity"
VERSION="$(jq -r '.simapp' VERSION)"
# Output image is overridable so CI can tag per-architecture (e.g.
# uzairakbar/deepracer-test:v0-amd64). Set PUSH=1 to push it after building.
# Note: Docker repository names must be lowercase.
OUT_IMAGE="${OUT_IMAGE:-uzairakbar/deepracer:v1}"
PUSH="${PUSH:-0}"

# Default to the host architecture (native build, no emulation). Override with
# ARCH=amd64 to produce the P4-compatible image (slow under emulation on arm64).
HOST_ARCH="$(uname -m)"
case "${ARCH:-${HOST_ARCH}}" in
    arm64|aarch64) TARGET_ARCH="arm64" ;;
    *)             TARGET_ARCH="amd64" ;;
esac
PLATFORM="linux/${TARGET_ARCH}"
BASE_IMAGE="${PREFIX}/deepracer-simapp:${VERSION}-cpu-${TARGET_ARCH}"

SKIP_BASE="false"
[ "${1:-}" = "-s" ] && SKIP_BASE="true"

if [ "${SKIP_BASE}" = "true" ] && [ -n "$(docker images -q "${BASE_IMAGE}" 2>/dev/null)" ]; then
    echo "==> Reusing existing base image ${BASE_IMAGE}."
else
    echo "==> Building CPU base image ${BASE_IMAGE} (compiles edited bundle source) for ${PLATFORM}..."
    ./build.sh -a cpu --platform "${PLATFORM}"
fi

echo "==> Building ZMQ sim overlay ${OUT_IMAGE}..."
# Use the legacy builder (DOCKER_BUILDKIT=0). buildx resolves the `FROM` base from
# its own content store, which can serve a stale copy of a just-rebuilt local base
# image; the legacy builder reads `FROM` straight from the daemon image store.
DOCKER_BUILDKIT=0 docker build . --platform "${PLATFORM}" \
    -f docker/Dockerfile.zmqsim \
    --build-arg BASE_IMAGE="${BASE_IMAGE}" \
    --build-arg TARGETARCH="${TARGET_ARCH}" \
    -t "${OUT_IMAGE}"

if [ "${PUSH}" = "1" ]; then
    echo "==> Pushing ${OUT_IMAGE}..."
    docker push "${OUT_IMAGE}"
fi

echo "==> Done: ${OUT_IMAGE}"
