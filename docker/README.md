#Docker images for running AGG experiments on GCP

Contains the following
  - agg-worker-docker[base,deploy]: container for running experiments. Base image has conda/cuda environment, and deploy adds code on top of it.
  - agg-utils-docker[base, deploy]: container for running code in non-gpu environments (ie pre-processing datasets.) Base image has conda environment set up, and deploy adds agg code on top of it.
  - mlflow-tracking-server: container for running mlflow tracking server with artifact storage on either gcp or aws
  - worker-run-experiment: script for running a containerized experiment on gcp. Contains relevant docker commands for mounting dataset, copying experiment files and environment variables, and running script containing an mlflow experiment 
  - access-mlflow-ui: docker-compose file for running gcp cloud sql proxy and local mlflow tracking server, so results can be viewed locally.

