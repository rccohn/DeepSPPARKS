from pathlib import Path
from sys import argv, stdout
import yaml

params_path = Path(argv[1])
with open(params_path, 'r') as f:
    params = yaml.safe_load(f)

option = argv[2]

if option == "experiment":
    stdout.write('{}\n'.format(params['mlflow']['experiment_name']))

elif option == "project_uri":
    stdout.write('{}\n'.format(params['mlflow']['project']))

elif option == "project_file":
    # replace git repo with "../../", the root directory of the project
    # relative to DeepSPPARKS/docker/worker-run-experiment
    # assume there are no "#" characters in file name...
    root = Path('..', '..')
    exp = params['mlflow']['project'].split('#')[1].strip(' ')
    stdout.write('{}\n'.format(root / exp))
