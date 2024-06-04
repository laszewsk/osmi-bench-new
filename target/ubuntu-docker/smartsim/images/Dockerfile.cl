FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

#FROM nvidia/cuda:12.5.0-base-ubuntu22.04

#FROM nvidia/cuda:12.5.0-runtime-ubuntu22.04
#FROM nvidia/cuda:12.5.0-devel-ubuntu22.04

ENV TZ="America/New_York"
ENV CUDNN_LIBRARY="/usr/lib/x86_64-linux-gnu/libcudnn.so.9"
# ENV CUDNN_INCLUDE_DIR
# ENV CUDNN_INCLUDE_PATH
ENV CUDNN_LIBRARY_PATH="/usr/local/cuda-12.4"
ENV CUDA_TOOLKIT_ROOT_DIR="/usr/local/cuda-12.4"


# RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
RUN dpkg -i cuda-keyring_1.1-1_all.deb
RUN apt-get update
RUN apt-get install cuda-toolkit -y



ARG DEBIAN_FRONTEND="noninteractive"

RUN apt update 
RUN apt install -y \
    build-essential \
    gcc \
    make \
    wget \
    unzip \
    zlib1g

RUN apt install -y git-all git-lfs --fix-missing

RUN apt install libopenmpi-dev openmpi-bin -y

# RUN apt install libcudnn9-cuda-12 -y
# RUN apt install cudnn9-cuda-12 -y

RUN apt-get install python3-pip python3 python3-dev cmake -y

RUN pip install cloudmesh-common cloudmesh-gpu cloudmesh-apptainer

RUN python3 -m pip install tensorflow[and-cuda]

RUN pip install onnxruntime-gpu tf2onnx  

RUN pip install torch




# ENV LD_LIBRARY_PATH="/usr/local/cuda-12.5/targets/lib:${LD_LIBRARY_PATH}"

# Install SmartSim
RUN pip install smartsim
RUN smart clean
RUN smart build --device=gpu 
RUN smart validate
