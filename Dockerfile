# syntax = docker/dockerfile:experimental

ARG CUDA_BASE
FROM nvidia/cuda:$CUDA_BASE

ARG PYTHON_VERSION=3.9
ARG DEBIAN_FRONTEND=noninteractive

# Install python
RUN apt-get update && \
    apt-get install --no-install-recommends -y software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install --no-install-recommends -y \
        python${PYTHON_VERSION} \
        python${PYTHON_VERSION}-dev \
        python${PYTHON_VERSION}-venv \
        python${PYTHON_VERSION}-distutils && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python${PYTHON_VERSION} -m venv /venvs/base
ENV PATH="/venvs/base/bin:${PATH}"

# Install requirements
ARG CUDA_TORCH
COPY requirements-*.txt /venvs/base/
RUN pip install --upgrade pip
RUN --mount="type=cache,target=/root/.cache/pip" \
    pip install -r /venvs/base/requirements-torch_${CUDA_TORCH}.txt \
                -r /venvs/base/requirements-common.txt

# Configure user
ARG USER_ID
ARG GROUP_ID
RUN groupadd --gid $GROUP_ID docker && \
    adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID --shell /bin/bash docker && \
    adduser docker sudo && \
    echo "%sudo ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Copy code
COPY --chown=$USER_ID:$GROUP_ID . /home/docker/metric-learning
