"""
Commonly paths that are reused when running experiments
in Dockerized environments. Paths are saved here as constants
to make it easier to avoid typos.
"""
from pathlib import Path
import os

assert os.environ.get("HOME") is not None, "$HOME is not set!"

_HOME = Path(os.environ["HOME"])  # convert $HOME to Path object
_VOL = _HOME / "volumes"  # datasets/inputs mounted here

# read-only docker mount for inputs to experiment. Usually
# this is the params.yaml file, but can include others as well.
INPUT_PATH = _VOL / "inputs"

# params.yaml from which all parameters are read.
PARAM_PATH = INPUT_PATH / "params.yaml"

# read-only path to mounted directory containing raw datasets
RAW_DATA_ROOT = _VOL / "datasets"

# read-write path to mounted directory containing processed data used by experiments
# processed data is written to host so that it can be reused on subsequent experiments
PROCESSED_DATA_ROOT = _VOL / "processed"

# path inside docker container (ie not mounted from host)
# save artifacts here before logging to mlflow server
# note that as this path is not mounted, all files stored here are removed
# when container is deleted .
ARTIFACT_PATH = _HOME / "temp" / "artifacts"
