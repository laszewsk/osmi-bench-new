PYTORCH_VERSION=24.05-py3
TF_VERSION=24.05-tf2-py3
PYTORCH_IMAGE=cloudmesh-pytorch-${PYTORCH_VERSION}
TF_IMAGE=cloudmesh-tf-${TF_VERSION}
UID=`id -u`
GID=`id -g`

#CACHE=--no-cache
CACHE=

DOCKER_FLAGS=--gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864
DOCKER_USER=-v /etc/passwd:/etc/passwd:ro -v /etc/group:/etc/group:ro  --user ${UID}:${GID}
DOCKER_BUILD_FLAGS=--progress=plain ${CACHE}


# ubuntu image
# driver=555.82.01
# toolkit 12.5.0
# PYTHON 3.10.8
# TF=
# TORCH=
# SMARTSIM=

# BAREMETAL
# DOCKER
# singularity


# Cloudmesh= DONT care 

osmi=config.yaml


image: docker-tf docker-torch

docker-tf:
	docker build ${DOCKER_BUILD_FLAGS} -t cloudmesh/tf -f Dockerfile.tf .

docker-torch:
	docker build ${DOCKER_BUILD_FLAGS} -t cloudmesh/torch -f Dockerfile.torch .

docker-cl:
	# wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
	docker build ${DOCKER_BUILD_FLAGS} -t cloudmesh/cl -f Dockerfile.cl .


shell-cl:
	docker run ${DOCKER_FLAGS} ${DOCKER_USER} -v ${PWD}:${PWD} -w ${PWD} -it cloudmesh/cl bash


shell-cl-su:
	docker run ${DOCKER_FLAGS}  -v ${PWD}:${PWD} -w ${PWD} -it cloudmesh/cl bash


shell-tf:
	docker run ${DOCKER_FLAGS} ${DOCKER_USER} -v ${PWD}:${PWD} -w ${PWD} -it cloudmesh/tf bash

shell-torch:
	docker run ${DOCKER_FLAGS} ${DOCKER_USER} -v ${PWD}:${PWD} -w ${PWD} -it cloudmesh/torch bash

#--ipc=host --ulimit memlock=-1 --ulimit stack=67108864  bash

#${PWD}:${PWD} -w ${PWD} bash



images: pytorch-image tf-image

pytorch-image: ${PYTORCH_IMAGE}.def
	-apptainer build --force ${PYTORCH_IMAGE}.sif ${PYTORCH_IMAGE}.def

pytorch-shell:
	-apptainer shell --nv ${PYTORCH_IMAGE}.sif

tf-image: ${TF-IMAGE}.def
	-apptainer build --force ${TF_IMAGE}.sif ${TF_IMAGE}.def

tf-shell:
	-apptainer shell --nv ${TF_IMAGE}.sif



# watch: 
# 	watch squeue --format=\"%.18i %.9P %.50j %.8u %.8T %.10M %.9l %.6D %R\" --me

# run:
# 	cd ${CODE_DIR}; mkdir -p outputs
# 	cd ${CODE_DIR}; singularitxy exec ${BIND} --nv cloudmask.sif bash -c "python cloudmask_v2.py --config=config-new.yaml"




# image-haproxy: haproxy_latest.sif
# 	mkdir -p image-singularity
# 	cd image-singularity; time singularity pull docker://haproxy


# run-localscratch:
# 	cd ${CODE_DIR}; mkdir -p outputs
# 	cd ${CODE_DIR}; singularity exec ${BIND_LOCALSCRATCH} --nv cloudmask.sif bash -c "python cloudmask_v2.py --config=config-new.yaml"

# #singularity exec --nv ${NAME}.sif papermill ${NAME}.ipynb ${NAME}_output.ipynb

# shell-localscratch:
# 	singularity ${BIND_LOCALSCRATCH} shell --nv ${IMAGE}

# shell-rivanna:
# 	singularity shell --nv ${IMAGE}



# cancel: stop

# stop:
# 	for i in "$$(squeue --user $$USER | awk 'NR>1{print $$1}')"; do scancel $$i ; done

# view:
# 	watch tail -n 50 *.err

# all: delete clean image submit view
