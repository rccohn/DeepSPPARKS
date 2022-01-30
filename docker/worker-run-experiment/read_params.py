# Read env variables from params file that are specific to individual
# projects (ie it does not make sense to store in .env), but must
# be parsed before calling mlflow run, like project uri/version, etc.

# accepts 1 argument from argv, the path to the parameter file to parse.
# to call: python read_params.py <path_to_params.yaml>

# prints variables to terminal on separate lines with the form NAME=VALUE, 
# and can be converted to bash variable with:
# $ vars=( $(python read_params.py <path_to_params.yaml>) )
# $ for v in ${vars[@]}; do export ${v}; done;

from pathlib import Path
from sys import argv, stdout
import yaml


params_path = Path(argv[1])
with open(params_path, 'r') as f:
    params = yaml.safe_load(f)

# experiment name to store runs under in mlflow tracking database
stdout.write('MLFLOW_EXPERIMENT_NAME={}\n'
        .format(params['mlflow']['experiment_name']))

project_uri=params['mlflow']['project_uri']
version=params['mlflow']['project_version']
if version == ".file":
    # run locally from file, not from github
    # version cannot be passed to mlflow run
    # as versions are not supported for file uris
    # unset version (do not use -v $VERSION)
    stdout.write('VERSION_ARG=\n')
    stdout.write('VERSION=\n')
    # relative path to DeepSPPARKS from DeepSPPARKS/docker/worker-run-experiment
    root = Path('..','..')
    # get path to experiment from git uri (git@github...#experiments/...)
    # strip spaces to ignore whitespace before comments in param file
    exp = params['mlflow']['project_uri'].split('#')[1].strip(' ')
    stdout.write('PROJECT_URI={}\n'.format(str(root / exp)))

else:
    # uri is git repo --> use versioning
    # mlflow run -v ${VERSION}
    stdout.write('VERSION_ARG=-v\n')
    stdout.write('VERSION={}\n'.format(version))
    stdout.write('PROJECT_URI={}\n'.format(project_uri))
