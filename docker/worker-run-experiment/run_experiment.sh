#!/bin/bash

## run experiment
## script requires 1 argument, path to experiment directory
## experiment directory must contain run_experiment.sh and .env

# run_experiment.sh is the shell script which runs the code 
# it can be as simple as "python my_file.py"

# .env defines environment variables like the 
# path to the dataset to use, address of the mlflow tracking server,
# and several others. See "env_template" for a sample list

# validate input deck
INPUT=$1
if [ -z $INPUT ]; then
	echo "supply path to experiment"
	exit
fi

for f in 'start.sh' '.env';
do
  if [ ! -f ${INPUT}/${f}.sh ]; then
    echo "experiment must have ${f}!"
    exit
  fi
done

# docker mounts require absolute path, convert argument 1 to absolute
EXP=$( realpath $INPUT) # path to experiment (directory includes run_experiment.sh and other files)

# load environment variables from experiment
source ${EXP}/.env # defines dataset name, tracking uri, paths on host/container, etc

# if processed_data directory does not exist, create it
# (if it does exist, it is left alone, not overwritten)
mkdir -p ${HOST_PROCESSED}

## NOTE: you can't have ANY characters (spaces, comments) after line break (\)
## otherwise the command will not run, you will get a vague error message,
## and it  will be VERY difficult to debug. 
## Workaround is to put comments in backticks BEFORE the linebreak.

# note that experiment and dataset mounts are readonly so script cannot accidentally 
# alter experiment parameters or dataset
# intermediate data needs to be writeable
echo "starting container ${DOCKER_IMAGE}"
CID=$( docker run -itd  --rm --name experiment-run \
	 --gpus all `#enable gpu` \
	 --env-file ${INPUT}/.env `#pass env variables to container` \
	 --mount type=bind,source=${EXP},target=${CONTAINER_EXP},readonly`# mount experiment directory` \
	 --mount type=bind,source=${HOST_DATASET},target=${CONTAINER_DATASET},readonly `# mount dataset` \
	 --mount type=bind,source=${HOST_PROCESSED},target=${CONTAINER_PROCESSED} `#mount processed data` \
	 ${DOCKER_IMAGE} ) #  (can't run experiment file yet 
                           # because mounted directories not yet available)
echo container started: $CID # verify container started correctly
docker exec $CID bash -c 'echo "verifying gpu access"  && nvidia-smi' # should show gpu name
sleep 2
docker exec $CID bash -c 'echo $CONTAINER_DATASET'
docker exec $CID bash -c 'echo running /root/exp/start.sh'
docker exec $CID bash -c 'cd /root/exp/ && bash "start.sh"'
echo "stopping container"
CID=$(docker stop $CID)
# TODO poweroff if needed # remember mlflow server also needs to be shut down
# think about how to best do this (considering multiple workers may run in 
# parallel. use lockfiles on cloud storage?)
# sudo shutdown
exit
