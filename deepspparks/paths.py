from pathlib import Path
import numpy

numpy.pi

# read-only docker mount for inputs to experiment
# usually this is the params.yaml file
INPUT_PATH = Path("/", "root", "inputs")

# params.yaml from which all parameters are read
PARAM_PATH = INPUT_PATH / "params.yaml"

# path inside docker container (ie not mounted from host)
# save artifacts here before logging to mlflow server
# note that as this path is not mounted, all files stored here are removed
# when container is deleted .
ARTIFACT_PATH = Path("/", "root", "artifacts")

# parent directory that contains all data (raw and processed)
DATA_ROOT = Path("/", "root", "data")

# read-only path to mounted directory containing raw datasets
RAW_DATA_ROOT = DATA_ROOT / "datasets"

# read-write path to mounted directory containing processed data used by experiments
# processed data is written to host so that it can be reused on subsequent experiments
PROCESSED_DATA_ROOT = DATA_ROOT / "processed"
