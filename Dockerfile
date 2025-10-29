FROM ubuntu:latest

# Install Python
RUN apt update && apt install software-properties-common -y && add-apt-repository ppa:deadsnakes/ppa
RUN apt update && apt install python3.11 python3-pip python3.11-dev python3.11-venv -y

# Install system
RUN python3.11 -m pip install Cython numpy scipy matplotlib setuptools==69.1.0
RUN apt-get -y install cmake liblapack-dev libsuitesparse-dev libhypre-dev curl git make vim bash-completion
RUN cp -v /usr/lib/x86_64-linux-gnu/libblas.so /usr/lib/x86_64-linux-gnu/libblas_OPENMP.so

# Install superlu
RUN cd /tmp && \
    curl -fSsL https://github.com/xiaoyeli/superlu_mt/archive/refs/tags/v4.0.1.tar.gz | tar xz && \
    cd superlu_mt-4.0.1 && \
    cmake -Denable_examples=OFF -Denable_tests=OFF -DPLAT="_OPENMP" -DCMAKE_INSTALL_PREFIX=/usr -DCMAKE_INSTALL_LIBDIR=lib -DSUPERLUMT_INSTALL_INCLUDEDIR=include . && \
    make -j4 && \
    make install

# Install sundials
RUN git clone --depth 1 -b v2.7.0 https://github.com/LLNL/sundials /tmp/sundials && \
    cd /tmp/sundials && \
    echo "target_link_libraries(sundials_idas_shared lapack blas superlu_mt_OPENMP)" >> src/idas/CMakeLists.txt && \
    echo "target_link_libraries(sundials_kinsol_shared lapack blas superlu_mt_OPENMP)" >> src/kinsol/CMakeLists.txt && \
    cmake -LAH -DSUPERLUMT_BLAS_LIBRARIES=blas -DSUPERLUMT_INCLUDE_DIR=/usr/include -DSUPERLUMT_LIBRARY=/usr/lib/libsuperlu_mt_OPENMP.a -DSUPERLUMT_THREAD_TYPE=OpenMP -DCMAKE_INSTALL_PREFIX=/usr -DSUPERLUMT_ENABLE=ON -DLAPACK_ENABLE=ON -DEXAMPLES_ENABLE=OFF -B build .  && \
    cd build  && \
    make -j4  && \
    make install

#Install assimulo
RUN git clone --depth 1 -b Assimulo-3.7.2 https://github.com/modelon-community/Assimulo /tmp/Assimulo && \
    cd /tmp/Assimulo && \
    rm setup.cfg && \
    python3.11 setup.py install --user --sundials-home=/usr --blas-home=/usr/lib/x86_64-linux-gnu/ --lapack-home=/usr/lib/x86_64-linux-gnu/ --superlu-home=/usr

# Install fmilib
RUN cd /tmp && \
    curl -fSsL https://github.com/modelon-community/fmi-library/archive/3.0.4.tar.gz | tar xz && \
    cd fmi-library-3.0.4 && \
    cmake -DCMAKE_INSTALL_PREFIX=/usr -DFMILIB_BUILD_TESTS=OFF -B build . && \
    cd build && \
    make -j4 && \
    make install

# Setup a venv to put pip and python on path for convinience running commands
ARG PYTHON_VENV=/src/.venv
ENV PATH=${PYTHON_VENV}/bin:$PATH
WORKDIR /src
