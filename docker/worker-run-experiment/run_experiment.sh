#!/bin/bash
EXP="/home/ryan/Documents/School/Research/Projects/AGG-new/DeepSPPARK/experiments/2022-01-03-vgg16-neu-baseline"

ENV_FILE=.env
POSITIONAL_ARGS=()
GPU_ARG="-A gpus=all" # run with gpus unless --cpu is specified
PARAM_FILE=.run_params.yaml

while (("$#" )); do
    case "$1" in
        -e|--env-file)
            ENV_FILE=$2
            shift # past argument
            shift # past value
        ;;
		--param-file)
			PARAM_FILE=$2
			shift # past argument
			shift # past value
		;;
        --cpu)
            GPU_ARG=""
            shift
        ;;
        *)
        POSITIONAL_ARGS+=("$1") # save positional arg
        shift # past argument
        ;;
    esac
done

PROJECT_DIR=${POSITIONAL_ARGS[0]}

if [ -z ${PROJECT_DIR}  ]; then
    echo "missing argumernt 1: path to project to run"
    exit 0
fi
if [ ! -d ${PROJECT_DIR} ]; then
    echo "Project directory does not exist: ${PROJECT_DIR}"
fi

if [ ! -f ${ENV_FILE} ]; then
    echo "Missing or invalid env file: ${ENV_FILE}"
	exit 0
fi

if [ ! -f ${PARAM_FILE} ]; then
	echo "Missing or invalid param file: ${PARAM_FILE}"
	exit 0
fi

export PARAM_FILE=$(realpath ${PARAM_FILE})
echo param file ${PARAM_FILE}
# export variables from .env file
parse_line(){
    LINE=${1}
    # if line is not empty OR does not start with spaces then comment
    if [ ! -z "$(echo ${LINE})" ]; then
        # if line does not start with comment, export line (VAR=VALUE)
        if [ ! "${LINE:0:1}" = "#" ]; then
            export "${LINE}" # export variable
        fi
    fi
}

while read LINE; do parse_line "${LINE}"; done < ${ENV_FILE}

# parse mlflow experiment
export MLFLOW_EXPERIMENT_NAME=$(python read_params.py ${PARAM_FILE})
echo "experiment: ${MLFLOW_EXPERIMENT_NAME}"

# mount datasets as read only

# note: by default, mlflow only allows unique arguments to be passed to docker 
# (ie you can't specify two --mount arguments, even though docker allows this)
# a workaround is to pass multiple docker arguments within a single -A argument for mlflow
# for example, 

echo listing vars
echo $MLFLOW_TRACKING_URI
echo ${PARAM_FILE}
echo ${VGG16_IMAGENET_PATH}
echo $CONTAINER_SSH
echo ${RAW_DATA_ROOT}
echo ${PROCESSED_ROOT}
echo done

mlflow run  \
    -e main `# entrypoint (default main)`\
	-A network="host -it" `# docker: host networking`\
	${GPU_ARG}\
	${PROJECT_DIR}  # project uri
# -A gpus all `#enable gpu' \


