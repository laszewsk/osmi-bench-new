NAME=osmi

SHELL=/bin/bash

OS_TYPE=macos
OS_DIR=..
PROJECT_DIR=../..

.PHONY: images image-docker project

benchmark:
	-make project
	-make run  

all: requirements data

# #####################################################################
# REQUIREMENTS
# #####################################################################

requirements:
	time pip install -r ../requirements.txt -U

# #####################################################################
# PROJECT MANAGEMENT
# #####################################################################

project: project-slurm generate

#setup:
#	python setup_env_and_yaml.py
#	source ~/ENV3/bin/activate && pip install -r /scratch/${USER}/mlcommons/benchmarks/${NAME}/experiments/rivanna/requirements.txt

generate: project-jobs-slurm.sh

run: submit

submit:
	-sh project-jobs-slurm.sh

project-slurm:
	cp ../../../../tfserving/train.py ../build/tfserving/train.py
	mkdir -p project
	cms ee generate \
	        --source=model.in.slurm \
	        --config=config.in.yaml \
	        --name=project \
	        --noos \
	        --os=USER,HOME \
		   	--nocm \
	        --output_dir=./project \
            --source_dir=. \
		    --verbose \
			--copycode="../build/tfserving/archs,../build/tfserving/train.py"
	cms ee generate submit --name=project --job_type=slurm > project-jobs-slurm.sh
	cms ee list


kill: stop

stop:
	for i in "$$(squeue --user $$USER | awk 'NR>1{print $$1}')"; do scancel $$i ; done

inspect:
	$(eval D=$(shell ls project/$(ls -1) | head -n 1))
	echo ${D}
	$(shell emacs project/${D}/config.yaml project/${D}/job.slurm)

watch: status

status:
	watch squeue --format=\"%.18i %.9P %.50j %.8u %.8T %.10M %.9l %.6D %R\" --me

clean:
	@-rm -rf project project.json jobs-project.sh
	@-rm -rf '__pycache__'
	@-rm -rf *~

# PROJECT_USER=/project/bii_dsc_community/tma5gv
PROJECT_USER=/scratch/tma5gv

get-results:
	-rsync -av --progress rivanna:${PROJECT_USER}/${NAME}/target/rivanna/project .

# #####################################################################
# APPTAINER IMAGE BUILD
# #####################################################################

image: 
	cd images/tensorflow; time make image

shell:
	cd images/tensorflow; make shell

clean-images:
	@-rm -rf images/*.sif

images: image




