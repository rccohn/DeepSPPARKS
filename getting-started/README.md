# Getting started
This section demonstrates the core functionality of Deepspparks, including:
  - Running a simple graph convolution experiment with a sample dataset.
  - Loading and interacting with graphs
  - Accessing results from sample experiment programatically
Note that DeepSPPARKS was built around [Docker](https://www.docker.com/) to manage dependencies and make it easier to run experiments in a cloud computing environment. You will need Docker to be able to run the mlflow projects in this repository. Docker-compose is also needed to run certain parts of this repository. 

# Run MLflow tracking server
DeepSPPARKS uses [mlflow](https://mlflow.org/) to track experiments. If you don't already have a tracking server configured, [mlflow-server-docker](https://github.com/rccohn/mlflow-server-docker) provides a preconfigured one that will work.  To set up the tracking server
- Clone the repositry
- Optionally, change the values in the `.env` file to set new values for `MLFLOW_SERVER_PORT`, `MLFLOW_TRACKING_USERNAME`, and `MLFLOW_TRACKING_PASSWORD`
- Start the server with: `$ docker compose up -d `
- Open a browser and navigate to [http://localhost:5000](http://localhost:5000) and login with the username and password specified in `.env`. You should see the interactive MLflow UI

# Running sample experiment
  ## Build project environment
  This step creates the environment that the sample project executes in.
  ```bash
  # in the getting-started directory
  cd ../docker/worker-node-cpu && ./container -b
  ```
## Set up environment to run project
A separate python environment is needed to invoke the `mlflow run` command. Because this is a simple environment, Docker would be overkill, and we just use a normal virtual environment instead.
```bash
# in the getting-started directory
python -m venv env  && \
  source env/bin/activate && \
  python -m pip install -r requirements.txt
```

## Configure `.env` file
Update `MLFLOW_TRACKING_USERNAME` and `MLFLOW_TRACKING_PASSWORD` in `.env` to match the login credentials specified for the tracking server.

## Run the project
```bash
source env/bin/activate && python run_sgc.py
```
The project will run. Open a browser and navigate to [http://localhost:5000](http://localhost:5000) and you should see several runs populate the table.
To change the parameters like learning rate, you can change the values in `sample_params.yaml` and re-run the experiment. 

# Accessing results with jupyter
## Build the jupyter environment
```bash
# in the getting-started directory
cd ../docker/worker-node-cpu && ./container -b --dex
```
## Start the jupyter server
Jupyter is run in a dockerized environment. `run_jupyter.sh` provides the correct command to start the server and also passes MLflow tracking credentials to the container so the server is actually reachable.
```bash
source env/bin/activate && bash run_jupyter.sh
```
`getting_started.ipynb` demonstrates a few common functions with deepspparks and mlflow for working with data and results.

# Generating a new dataset
The simulation code is available in the [spparks-meso](https://www.github.com/holmgroup/spparks-meso) repository. After following the instructions to set up the meso utility, run it with docker to generate simulations. The following command will generate a dataset of 5 initial states and 3 repeated grain growth simulations.

```bash
$ docker run --rm \
         -v $(pwd)/results:/home/meso-home/runs:rw \
         meso:main --n-init 5 --n-grow 3 \
		 --animate --size 256 --log-iter 25000
```
