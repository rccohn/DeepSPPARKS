source .env # defines PGPASS
artifact_path=$(pwd)/mlruns/ # path to store artifacts, should be reachable from sftp
backend_uri=postgresql+psycopg2://postgres:${PGPASS}@localhost:5432/mlflow # uri for backend store
artifact_uri=sftp://localhost${artifact_path}
IMAGE=rccohn/mlflow-tracking-server-1.21.0:latest

# runs tracking server
# with countainer_ssh mounted as ~/.ssh for the container
# note that keys/configs/known_host must be added to container_ssh before using, 
# see the README for more info!
docker run --rm -d --name mlflow_local_sever \
	--network "host" \
	--mount type=bind,source=$(pwd)/container_ssh,target=/root/.ssh,readonly \
	 ${IMAGE} \
	 mlflow server --host=localhost --port=5001 \
	--backend-store-uri ${backend_uri} \
	--default-artifact-root ${artifact_uri}

