PYTORCH_VERSION=24.05-py3
TF_VERSION=24.05-tf2-py3
PYTORCH_IMAGE=cloudmesh-pytorch-${PYTORCH_VERSION}
TF_IMAGE=cloudmesh-tf-${TF_VERSION}
UID=`id -u`
GID=`id -g`
CODE=${HOME}/Desktop/osmi-bench-new/target/ubuntu-docker/smartsim

#CACHE=--no-cache
CACHE=

DOCKER_FLAGS=--gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864
DOCKER_USER=-v /etc/passwd:/etc/passwd:ro -v /etc/group:/etc/group:ro  --user ${UID}:${GID}
DOCKER_BUILD_FLAGS=--progress=plain ${CACHE}

help:
	docker run ${DOCKER_FLAGS} ${DOCKER_USER} -v ${CODE}:${CODE} -v ${HOME}:${HOME} -w ${CODE} -it cloudmesh/tf bash -c "python3 train.py --help"

train:
	docker run ${DOCKER_FLAGS} ${DOCKER_USER} -v ${CODE}:${CODE} -v ${HOME}:${HOME} -w ${CODE} -it cloudmesh/tf bash -c "python3 train.py"

