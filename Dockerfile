FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# Python toolchain
RUN apt-get update && apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y \
      python3.11 python3.11-dev python3.11-venv python3-pip

# Build prerequisites for FMIL + runtime libs pulled in by assimulo-testing
# (it dynamically links libopenblas; SUNDIALS + SuperLU are statically
# embedded in the wheel).
RUN apt-get install -y \
      cmake make curl git vim bash-completion \
      libopenblas-dev gfortran

# Install fmilib 3.0.4 into /usr
RUN cd /tmp && \
    curl -fSsL https://github.com/modelon-community/fmi-library/archive/3.0.4.tar.gz | tar xz && \
    cd fmi-library-3.0.4 && \
    cmake -DCMAKE_INSTALL_PREFIX=/usr -DFMILIB_BUILD_TESTS=OFF -B build . && \
    cd build && \
    make -j"$(nproc)" && \
    make install && \
    rm -rf /tmp/fmi-library-3.0.4

# Venv with pip + pytest preinstalled. PyFMI itself is built/installed by
# `make build`, which runs pip install . with meson-python; build isolation
# resolves Cython/numpy/assimulo-testing automatically from pyproject.toml.
ARG PYTHON_VENV=/src/.venv
ENV PATH=${PYTHON_VENV}/bin:$PATH
WORKDIR /src
