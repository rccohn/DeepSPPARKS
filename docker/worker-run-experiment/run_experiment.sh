#!/bin/bash

ENV_FILE=.env # defines env variables such as MLFLOW_TRACKING_URI, location of dataset on host, etc
GPU_ARG="-A gpus=all" # run with gpus unless --cpu is specified
PARAM_FILE=.run_params.yaml # path to input parameter deck for mlflow project
ENTRYPOINT="main" # mlflow project entrypoint

POSITIONAL_ARGS=()
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
	--entrypoint)
		ENTRYPOINT=$2
		shift
		shift
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


# parse environment variables
while read LINE; do parse_line "${LINE}"; done < ${ENV_FILE}

# python and mlflow executibles
PYTHON_EXE=${PYTHON_ENV}/python3
MLFLOW_EXE=${PYTHON_ENV}/mlflow
echo "python exe: ${PYTHON_EXE}"

# parse mlflow experiment name, project uri,
# and version from param file
VARS=( $(${PYTHON_EXE} read_params.py ${PARAM_FILE}) )
for v in ${VARS[@]};
do
	export ${v}
done

# mount datasets as read only

# note: by default, mlflow only allows unique arguments to be passed to docker 
# (ie you can't specify two --mount arguments, even though docker allows this)
# a workaround is to pass multiple docker arguments within a single -A argument for mlflow
# for example, 

echo "Experiment paths:"
echo "    param file: ${PARAM_FILE}"
echo "    pretrained vgg16/imagenet: ${VGG16_IMAGENET_PATH}"
echo "    container ssh: ${CONTAINER_SSH}"
echo "    raw data: ${RAW_DATA_ROOT}"
echo "    processed data: ${PROCESSED_ROOT}"
echo "MLflow params"
echo "    tracking uri: $MLFLOW_TRACKING_URI"
echo "    experiment name: ${MLFLOW_EXPERIMENT_NAME}"
echo "    project uri: ${PROJECT_URI}"
echo "    project version (blank if file): ${VERSION}"
echo "    project entrypoint: ${ENTRYPOINT}"
echo "    GPU? (blank for cpu): ${GPU_ARG}"


${MLFLOW_EXE} run  \
    -e ${ENTRYPOINT} `# entrypoint (default main)`\
    ${VERSION_ARG} ${VERSION} `# mlflow project version`\
	-A network="host" `# docker: host networking`\
	${GPU_ARG}\
	${PROJECT_URI}  # project uri
# -A gpus all `#enable gpu' \


