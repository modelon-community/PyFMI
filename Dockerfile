FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y \
      python3.11 python3.11-dev python3.11-venv python3-pip \
      build-essential cmake curl git vim bash-completion \
      libgfortran5 && \
    rm -rf /var/lib/apt/lists/*

# Build FMIL from the same script the wheel pipeline uses, so the dev image
# and the published wheels link against an identical FMIL build.
COPY tools/wheels/build_dependencies.sh /tmp/build_dependencies.sh
RUN bash /tmp/build_dependencies.sh && rm /tmp/build_dependencies.sh

ARG PYTHON_VENV=/src/.venv
ENV PATH=${PYTHON_VENV}/bin:$PATH
WORKDIR /src
