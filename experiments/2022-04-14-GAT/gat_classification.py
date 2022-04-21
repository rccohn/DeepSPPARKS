from deepspparks.utils import load_params
import mlflow
from data import Dataset
from deepspparks.paths import PARAM_PATH  # ARTIFACT_PATH


def main():
    # classification based on mean, std growth ratio above a threshold
    print("Loading params: {}".format(PARAM_PATH))
    assert PARAM_PATH.is_file(), "param file not found!"
    params = load_params(PARAM_PATH)

    with mlflow.start_run(run_name="GAT-classification"):
        dataset = Dataset(params)
        dataset.process(force=params["dataset"]["force_process"])

        print("done!")


if __name__ == "__main__":
    main()
