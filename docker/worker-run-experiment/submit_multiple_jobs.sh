#!/bin/bash

# queue: files to process
# finished: params moved here after job runs
ROOT=.job_queue
QUEUE=${ROOT}/queue
FINISHED=${ROOT}/finished
LOG=${ROOT}/log.txt

SHUTDOWN=0

VERSION="main" # will default to main

# clear log for next set of runs
# note that command ... 2>&1 | tee -a ${LOG}
# appends stdout and sterr to log while still displaying them on terminal
# This can be used to both monitor jobs in progress, and 
# the previous job that shut down a vm
if [ -f ${LOG} ]; then
	rm ${LOG}
	touch ${LOG}
fi

while (("$#")); do
	case $1 in 
		--version)
			VERSION="${2}"
			shift
			shift
			;;
		--shutdown)
			SHUTDOWN=1
			shift
			;;
		--reset-queue)
			echo "resetting queue"
			mv ${FINISHED}/* ${QUEUE}
			exit 0
			;;
	esac
done

for PARAM_FILE in $(ls -1 ${QUEUE}); do
	SRC=${QUEUE}/${PARAM_FILE}
	TGT=${FINISHED}/${PARAM_FILE}
	printf "\n#####processing ${SRC}#####\n" 2>&1 | tee -a ${LOG}
	mv ${SRC} ${TGT}
	bash run_experiment.sh --version ${VERSION} --param-file ${TGT} \
		2>&1 | tee -a ${LOG}
	echo "done"
done


# TODO figure out shutdown of remote host (tracking server)

if [ ${SHUTDOWN} -eq 1 ]; then
	echo "Shutting down" 2>&1 | tee -a ${LOG}
	sleep 300 # give time to cancel shutdown
	sudo shutdown -h now 2>&1 | tee -a ${LOG}
fi

