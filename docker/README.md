# Docker images for running AGG experiments on GCP
## Docker images
These directories contain Dockerfiles and supporting files for building the Dockerized environmetns used in this study. Each directory contains a file called `container` which contains commands for building the image.
```bash
# to build an image:
./container -b
```
Directories with Dockerfiles are:
  - `worker-node` contains the GPU-enabled environment used to run experiments
  - `worker-node-cpu` contains the same environment but only runs on CPU, and is used to run the example project in the *getting-started* section.
  - `mlflow-tracking-server` contains code for the tracking server used in this project. For new users, it is recommended to use the tracking server in [mlflow-server-docker](https://github.com/rccohn/mlflow-server-docker), which provides a more updated configuration.

## Compose files
These directories contain compose files used to start the tracking server and a jupyter environment for interacting with results. Note that [mlflow-server-docker](https://github.com/rccohn/mlflow-server-docker) provides a cleaner, more up-to-date approach for running a tracking server.

## Running experiments
`worker-run-experiment/` contains `run_experiment.sh`, the shell script used to configure the experiment and invoke `mlflow run` to actually run the mlflow projects. For new users, `getting-started` provides a simplified guide for running a basic experiment.
