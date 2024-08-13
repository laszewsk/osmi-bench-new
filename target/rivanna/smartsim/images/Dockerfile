# FROM nvcr.io/nvidia/pytorch:24.05-py3
FROM smartsim:cuda-12

# Install additional dependencies
RUN apt-get update 
RUN apt-get install -y \
    lsof \
    python3-pip \
    git-lfs


# Install TensorFlow Serving
# RUN pip install tensorflow-serving-api[and-cuda]

# Install other required packages
RUN pip install grpcio numpy requests tqdm 

# Install Cloudmesh
RUN pip install cloudmesh-common cloudmesh-gpu cloudmesh-apptainer
