#!/bin/bash

ROOT=.job_queue
QUEUE=${ROOT}/queue
FINISHED=${ROOT}/finished

SHUTDOWN=0

for PARAM_FILE in $(ls -1 ${QUEUE}); do
	SRC=${QUEUE}/${PARAM_FILE}
	TGT=${FINISHED}/${PARAM_FILE}
	echo "processing ${SRC}"
	mv ${SRC} ${TGT}
	bash run_experiment.sh --param-file ${TGT}
	echo "done"
done


# TODO figure out shutdown of remote host (tracking server)

if [ ${SHUTDOWN} -eq 1 ]; then
	sudo shutdown -h now
fi

