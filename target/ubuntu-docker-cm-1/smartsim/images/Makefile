PYTORCH_VERSION=24.05-py3
TF_VERSION=24.05-tf2-py3
PYTORCH_IMAGE=cloudmesh-pytorch-${PYTORCH_VERSION}
TF_IMAGE=cloudmesh-tf-${TF_VERSION}
UID=`id -u`
GID=`id -g`
CODE=/home/green/Desktop/osmi-bench-new/target/ubuntu-docker/smartsim


#CACHE=--no-cache
CACHE=

DOCKER_FLAGS=--gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864
DOCKER_USER=-v /etc/passwd:/etc/passwd:ro -v /etc/group:/etc/group:ro  --user ${UID}:${GID}
DOCKER_BUILD_FLAGS=--progress=plain ${CACHE}

osmi=config.yaml

banner = @echo "------------------------"; \
		 echo "$1"; \
		 echo "------------------------"

all:
	$(call banner, "start")
	rm -rf ~/Desktop/osmi-bench-new/ashao/SmartSim 
	cd ~/Desktop/osmi-bench-new/ashao/ ; 	git clone --branch docker-gpu https://github.com/ashao/SmartSim.git
	@echo "CREATE CONTAINER smartsim:cuda-12"
	$(call banner, "container build smartsim:cuda-12")
	cd ~/Desktop/osmi-bench-new/ashao/SmartSim/docker/prod-cuda12; \
		docker build ${DOCKER_BUILD_FLAG} -t smartsim:cuda-12 .
	echo "VALIDATE CONTAINER smartsim:cuda-12"
	$(call banner, "smartsim validate")
	docker run -it ${DOCKER_FLAGS} ${DOCKER_USER} -w /tmp smartsim:cuda-12 smart validate
	$(call banner, "cloudmesh install")
	docker build ${DOCKER_BUILD_FLAGS} -t cloudmesh/smartsim -f Dockerfile .
	$(call banner, "CONTAINER cloudmesh/smartsim created.")



#shell:
# 	docker run ${DOCKER_FLAGS} ${DOCKER_USER} -v ${PWD}:${PWD} -w ${PWD} -it cloudmesh/smartsim bash

benchmark-train:
	docker run ${DOCKER_FLAGS} ${DOCKER_USER} -v ${HOME}:${HOME} -v ${CODE}:${CODE} -w ${CODE} -it cloudmesh/smartsim bash -c "python3 train.py"

benchmark:
	docker run ${DOCKER_FLAGS} ${DOCKER_USER} -v ${HOME}:${HOME} -v ${CODE}:${CODE} -w ${CODE} -it cloudmesh/smartsim bash -c "/bin/sh bench.sh"


clean:
	docker system prune -a --volumes
	docker rm -vf $(docker ps -aq)
