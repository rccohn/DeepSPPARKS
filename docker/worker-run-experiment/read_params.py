from pathlib import Path
from sys import argv, stdout
import yaml

params_path = Path(argv[1])
with open(params_path, 'r') as f:
    params = yaml.safe_load(f)

if argv[2] == 0:
    stdout.write('{}\n'.format(params['mlflow']['experiment_name']))

elif argv[2] == 1:
    stdout.write('{}\n'.format(params['mlflow']['project'])
