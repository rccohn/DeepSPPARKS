from pathlib import Path
from sys import argv, stdout
import yaml

params_path = Path(argv[1])
with open(params_path, 'r') as f:
    params = yaml.safe_load(f)

option = int(argv[2])

if option == 0:
    stdout.write('{}\n'.format(params['mlflow']['experiment_name']))

elif option == 1:
    stdout.write('{}\n'.format(params['mlflow']['project']))
