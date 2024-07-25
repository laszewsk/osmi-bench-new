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

image:
	cd images; make all

help:
	docker run ${DOCKER_FLAGS} ${DOCKER_USER} -v ${CODE}:${CODE} -v ${HOME}:${HOME} -w ${CODE} -it cloudmesh/smartsim bash -c "python3 train.py --help"

train:
	docker run ${DOCKER_FLAGS} ${DOCKER_USER} -v ${HOME}:${HOME} -v ${CODE}:${CODE} -w ${CODE} -it cloudmesh/smartsim bash -c "python3 train.py"

bench:
	docker run ${DOCKER_FLAGS} ${DOCKER_USER} -v ${HOME}:${HOME} -v ${CODE}:${CODE} -w ${CODE} -it cloudmesh/smartsim bash -c "/bin/sh bench.sh"


train-setup: clean
	mkdir -p project-train
	cms ee generate \
	        --source=bench-train.in.sh \
	        --config=config.in.yaml \
	        --name=project-train \
	        --noos \
	        --os=USER,HOME \
		   	--nocm \
	        --output_dir=./project-train \
            --source_dir=. \
		    --verbose \
			--copycode="train.py,archs,Makefile" 
	cms ee generate submit --name=project-train --job_type=bash > project-jobs-train.sh
	cms ee list project-train 

train-run: 
	sh project-jobs-train.sh

train-clean:
	@-rm -rf project-train project-train.json jobs-project-train.sh
	@-rm -rf '__pycache__'
	@-rm -rf *~

clean:
	@-rm -rf project-train project-train.json jobs-project-train.sh
	@-rm -rf project project.json jobs-project.sh
	@-rm -rf '__pycache__'
	@-rm -rf *~