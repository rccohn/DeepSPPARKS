# Experiments
This section contains various approaches for training and evaluating deep learning models to predict the grain growth trajectories in the datasets. Experiments are configured as [mlflow projects](https://www.mlflow.org/docs/latest/projects.html) and have common structures.

## Project structure
Each project contains the following files:
  - `MLProject`: MLflow project file that defines the environment that the project runs in and its entrypoints.
  - `sample_params.yaml`, which contains all of the inputs that can be passed to the experiment. 
  - `start.sh`: Used to run the entrypoints. The project could be configured to run without needing these, but at one point it helped with debugging.
  - `Various python files`: Code that is actually executed when running the experiment.  

## Project environment
Projects run with the `rccohn/deepspparks-worker:latest` Docker image, which can be built in the `docker/worker-node` section of this repository. 

## Project inputs and outputs
The inputs and outputs are defined by volumes in the `docker_env` section of the `MLproject` files. Typically, projects require the following:
    - `PARAM_FILE`: input parameters for experiments
    - `RAW_DATA_ROOT`: raw dataset is mounted to container as read-only directory
    - `PROCESSED_DATA_ROOT`: To reduce time when re-running experiments with different sets of parameters, processed data is saved to the host and reused when possible.

## Running experiments
`docker/worker-run-experiment` contains the script for running each experiment. Typically, the process is to copy `sample_params.yaml` from the experiment directory, update the parameter values, and then invoke `run_experiment.sh` to execute the code.

## Saving results
When experiments are run, all relevant results are saved to the mlflow tracking server. 


## Getting started
See the `getting-started` section of this repository for more info and steps to run the simple graph convolution experiment on a small sample dataset.