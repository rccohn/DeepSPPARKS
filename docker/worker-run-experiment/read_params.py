from pathlib import Path
from sys import argv, stdout
import yaml

params_path = Path(argv[1])
with open(params_path, 'r') as f:
    params = yaml.safe_load(f)

if 'mlflow' in params.keys():
    experiment = params['mlflow'].get('experiment_name', None)
    if experiment is not None:
        stdout.write("{}\n".format(experiment))
