# BSD 2-Clause License
#
# Copyright (c) 2021-2024, Hewlett Packard Enterprise
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# Changes done by A Shao and Gregor von Laszewski

# FROM nvcr.io/nvidia/tensorflow:24.05-tf2-py3

FROM ubuntu:22.04


# FROM nvcr.io/nvidia/tensorflow:23.10-tf2-py3
# FROM tensorflow/tensorflow:latest-gpu

# Install additional dependencies

LABEL maintainer="MLCommons derived from Cray Labs"
# LABEL org.opencontainers.image.source https://github.com/CrayLabs/SmartSim

ARG DEBIAN_FRONTEND="noninteractive"
ENV TZ=US/Seattle

# Make basic dependencies
RUN apt-get update \
    && apt-get install --no-install-recommends -y build-essential \
    git gcc make git-lfs wget libopenmpi-dev openmpi-bin unzip \
    python3-pip python3 python3-dev cmake wget

# Install Cudatoolkit 12.5 only if we du ubuntu image
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb && \
    apt-get update -y && \
    apt-get install -y cuda-toolkit-12-5

# Install cuDNN 8.9.7  only if we du ubuntu image
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/libcudnn8_8.9.7.29-1+cuda12.2_amd64.deb && \
    dpkg -i libcudnn8_8.9.7.29-1+cuda12.2_amd64.deb

# Install SmartSim and SmartRedis
RUN pip install git+https://github.com/CrayLabs/SmartRedis.git && \
    pip install git+https://github.com/CrayLabs/SmartSim.git@cuda-12-support

ENV CUDA_HOME="/usr/local/cuda/"
ENV PATH="${PATH}:${CUDA_HOME}/bin"

# Install machine-learning python packages consistent with RedisAI
# Note: pytorch gets installed in the smart build step
# This step will be deprecated in a future update
RUN pip install tensorflow==2.15.0

# Build ML Backends
RUN smart clean
RUN smart build --device=cuda121
# RUN smart build --device=gpu 


 # Correct permissions for RedisAI
RUN find /usr/local/lib/python3.10/dist-packages/smartsim -name "*.so*" -exec chmod a+rx {} \;



RUN pwd
RUN ls

RUN apt-get install -y \
    lsof 

# Install TensorFlow Serving
# RUN pip install tensorflow-serving-api[and-cuda]
# Install other required packages
# RUN pip install grpcio 

RUN pip install numpy requests tqdm 

# Install Cloudmesh
RUN pip install cloudmesh-common cloudmesh-gpu cloudmesh-apptainer

RUN smart validate

# Set the working directory
#WORKDIR /app

# Copy your application code to the container
#COPY . /app

# Set the entrypoint command
#CMD ["python", "your_script.py"]


# %post
#     echo "deb [arch=amd64] http://storage.googleapis.com/tensorflow-serving-apt stable tensorflow-model-server tensorflow-model-server-universal" | tee /etc/apt/sources.list.d/tensorflow-serving.list && \
# curl https://storage.googleapis.com/tensorflow-serving-apt/tensorflow-serving.release.pub.gpg | apt-key add -
#     apt-get update 
#     apt install -y tensorflow-model-server
#     apt upgrade -y tensorflow-model-server

#     apt install -y lsof
#     apt install -y python3-pip
#     # pip install pip -U


    # #pip install tensorflow[and-cuda]
    # pip install tensorflow-serving-api[and-cuda]


    # pip install grpcio
    # pip install numpy
    # pip install requests
    # pip install tqdm

    
	# pip install cloudmesh-common
	# pip install cloudmesh-gpu
    # pip install cloudmesh-apptainer
    
    # #smartredis==0.3.1
