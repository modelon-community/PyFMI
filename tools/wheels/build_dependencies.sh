#!/usr/bin/env bash
# Builds FMI Library into /usr. The single source of truth for the FMIL
# version used by the dev image, the manylinux image, and cibuildwheel.
# Requires curl, tar, cmake, and a C compiler on PATH.
set -eux

# cibuildwheel's before-all environment can ship a minimal PATH; restore the
# standard system paths so /usr/bin tools resolve.
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin${PATH:+:${PATH}}"

FMIL_VERSION="3.0.4"
NPROC=$(nproc)

curl -fSsL \
    "https://github.com/modelon-community/fmi-library/archive/${FMIL_VERSION}.tar.gz" \
    | tar xz -C /tmp

cmake -S "/tmp/fmi-library-${FMIL_VERSION}" \
      -B "/tmp/fmi-library-${FMIL_VERSION}/build" \
      -DCMAKE_INSTALL_PREFIX=/usr \
      -DFMILIB_BUILD_TESTS=OFF

make -C "/tmp/fmi-library-${FMIL_VERSION}/build" -j"${NPROC}"
make -C "/tmp/fmi-library-${FMIL_VERSION}/build" install
rm -rf "/tmp/fmi-library-${FMIL_VERSION}"
