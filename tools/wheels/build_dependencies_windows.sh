#!/usr/bin/env bash
# Windows counterpart to build_dependencies.sh. Runs under MSYS2 UCRT64 so
# FMIL is built with the same gcc the extension modules link against,
# avoiding the MSVC/MinGW ABI mismatch that breaks delvewheel bundling.
set -eux

FMIL_VERSION="3.0.4"
INSTALL_PREFIX="/c/deps"
NPROC=$(nproc 2>/dev/null || echo 4)

mkdir -p "${INSTALL_PREFIX}/bin" "${INSTALL_PREFIX}/lib" "${INSTALL_PREFIX}/include"

curl -fSsL \
    "https://github.com/modelon-community/fmi-library/archive/${FMIL_VERSION}.tar.gz" \
    | tar xz -C /tmp

cmake -S "/tmp/fmi-library-${FMIL_VERSION}" \
      -B "/tmp/fmi-library-${FMIL_VERSION}/build" \
      -G Ninja \
      -DCMAKE_INSTALL_PREFIX="${INSTALL_PREFIX}" \
      -DFMILIB_BUILD_TESTS=OFF \
      -DCMAKE_BUILD_TYPE=Release

cmake --build "/tmp/fmi-library-${FMIL_VERSION}/build" --parallel "${NPROC}"
cmake --install "/tmp/fmi-library-${FMIL_VERSION}/build"
rm -rf "/tmp/fmi-library-${FMIL_VERSION}"
