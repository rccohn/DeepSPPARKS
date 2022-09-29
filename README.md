# DeepSPPARKS
Cloud-native, high throughput experimunts utilizing Graph neural networks for predicting abnormal grain growth (AGG) in SPPARKS Monte Carlo simulations of grain growth.
<p align="center">
<img src=.github/graph-img.png width=25% height=auto max-width=500px min-width=250px alt="Graph overlaid on SPPARKS microstructure">
</p>

# 

# Repository Structure
## Getting started
This section provides a working example for running the simple graph convolution experiment with a sample dataset and accessing the results in a jupyter environment.

## experiments/
This section contains the code for individual experiments for characterizing grain growth trajectories. Experiments are configured as mlflow projects. Each experiment contains a sample YAML file with descriptions of the parameters that can be passed to the experiment. To run mlflow projects, `docker/worker-run-experiment/run_experiment.sh` is used. 

**NOTE:**
Some of the paths have changed, so some experiments may return file not found or permission denied errors. Following the code should reveal what paths need to be updated, and I'll fix this when I can get around to it, but in the meantime, see the `getting-started/` section for a fully working example.

## deepspparks/
This section defines the Graph data structure containing the SPPARKS simulation data, and provides some additional data analysis and
visualization functions that are commonly used in several experiments.

## docker/
This section contains Dockerfiles used to build the environments that the experiments and tracking server run in, docker compose files to run the server and a jupyter environment with different systems, and the script used to actually run experiments.

## datasets/
This section contains code for processing raw data into datasets.

# Data availability
The original SPPARKS datasets are too large to include in a git repository, but the code used to generate simulations has been released in the [https://github.com/holmgroup/spparks-meso](https://github.com/holmgroup/spparks-meso) repository.

