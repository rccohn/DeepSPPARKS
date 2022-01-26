#!/bin/bash

ROOT=.job_queue
QUEUE=${ROOT}/queue
FINISHED=${ROOT}/finished

SHUTDOWN=0

VERSION="" # will default to main

while (("$#")); do
	case $1 in 
		--debug)
			VERSION="--version dev"
			shift
			;;
	esac
done

for PARAM_FILE in $(ls -1 ${QUEUE}); do
	SRC=${QUEUE}/${PARAM_FILE}
	TGT=${FINISHED}/${PARAM_FILE}
	printf "\n#####processing ${SRC}#####\n"
	mv ${SRC} ${TGT}
	bash run_experiment.sh ${VERSION} --param-file ${TGT}
	echo "done"
done


# TODO figure out shutdown of remote host (tracking server)

if [ ${SHUTDOWN} -eq 1 ]; then
	sudo shutdown -h now
fi

