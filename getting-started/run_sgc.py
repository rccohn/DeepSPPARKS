from dotenv import load_dotenv
import mlflow
import os
from pathlib import Path
from yaml import safe_load


def main():
    param_file="sample_params.yaml"
    env_file=".env"
    project_uri="../experiments/2021-12-23-SGC"
    
    load_dotenv()
    os.environ['PARAM_FILE']=str(Path(param_file).absolute())
    os.environ['RAW_DATA_ROOT']=str(Path('data','json').absolute())
    os.environ['PROCESSED_DATA_ROOT']=str(Path('data','processed').absolute())
    with open(param_file, 'r') as f:
        experiment = safe_load(f)['mlflow']['experiment_name']
    os.environ['MLFLOW_EXPERIMENT_NAME'] = experiment
    mlflow.projects.run(project_uri, docker_args={'net': 'host'})

if __name__ == "__main__":
    main()
