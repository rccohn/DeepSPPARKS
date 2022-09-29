source .env
docker run --rm  \
	--net host \
	-e MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI} \
	-e MLFLOW_TRACKING_USERNAME=${MLFLOW_TRACKING_USERNAME} \
	-e MLFLOW_TRACKING_PASSWORD=${MLFLOW_TRACKING_PASSWORD} \
	-e CONTAINER_NOTEBOOK_DIR=/mnt \
	-v $(pwd)/jupyter:/home/jupyter/notebooks \
	-v $(pwd)/data:/home/jupyter/data \
	rccohn/deepspparks-worker:dex \
