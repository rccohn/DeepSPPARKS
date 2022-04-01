# 2 purposes:
# split list of files/folders to train/val/test
# convert input to graph with defined number of timesteps (ie lmd)
# see params file for specifications, but should be pretty straightforward
# copy multiprocessing and mlflow setup from patch_compression
from pathlib import Path
from deepspparks.graphs import Graph
from deepspparks.utils import load_params
import mlflow
import numpy as np
from itertools import repeat
from shutil import rmtree
import os
from multiprocessing import get_context, cpu_count


def process_graph(args):
    """
    Reads input file (either json or SPPARKS/meso output directory),
    and processes the graph to limit the amount of timesteps included
    in results, which can drastically reduce storage overhead.

    Parameters
    ----------
    args: tuple
        contains (input file, output parent directory,
        indices of time steps to keep (or None), and number of
        intermediate timesteps to keep (or None))

    """
    # unpack arguments (stored in single tuple for multiprocessing.map)
    in_file = args[0]
    out_dir = args[1]
    time_step_idx = args[2]
    nsteps = args[3]

    # apply correct method to load graph (either json or SPPARKS)
    if in_file.is_dir():
        g = Graph.from_spparks_out(in_file)
    else:
        g = Graph.from_json(in_file)

    # limit timesteps included in results
    # from file, graph only contains single metadata entry
    time_steps_all = g.metadata["subgraph_metadata"][0]["timesteps"]
    nts = len(time_steps_all[0])  # number of timesteps in each repeat
    if time_step_idx is None:
        if nsteps == -1:
            nsteps = nts
        time_step_idx = np.linspace(0, nts - 1, nsteps, dtype=int)

    g.metadata["subgraph_metadata"][0]["timesteps"] = [
        x[time_step_idx] for x in time_steps_all
    ]

    # only report grain sizes for desired timesteps
    # reducing number of intermediate steps included drastically
    # reduces file size
    for n in g.nodes:
        gs_all = g.nodes[n]["grain_size"]
        for i, gs in enumerate(gs_all):
            g.nodes[n]["grain_size"][i] = gs[time_step_idx]

    # save processed graph to disk
    out_name = in_file.name + ".json" if in_file.is_dir() else in_file.name
    g.to_json(out_dir / out_name)


def main():
    # load params
    params = load_params("/root/inputs/params.yaml")

    with mlflow.start_run(nested=False):
        # save param file
        mlflow.log_artifact("/root/inputs/params.yaml")

        parent_dataset = params["parent_dataset"]
        output_dataset_name = "{}-v{}".format(
            params["dataset_name"], params["dataset_version"]
        )
        mlflow.set_tags(
            {
                "mlflow.runName": "json graphs",
                "source": parent_dataset,
                "dataset": output_dataset_name,
            }
        )

        # get absolute number of datapoints in each split,
        # and record timesteps as 'metrics' so that it can be logged as an actual array

        parent_dataset_path = Path("/root", "data", "datasets", parent_dataset)
        output_dataset_path = Path("/root", "data", "processed", output_dataset_name)

        # if dataset already exists, clear file to avoid ambiguous combination
        # of new and old dataset
        if output_dataset_path.exists():
            rmtree(output_dataset_path)

        # read input files (either json or spparks outputs)
        files_all = [x for x in parent_dataset_path.glob("*.json")]
        if not files_all:
            files_all = [x for x in parent_dataset_path.glob("*") if x.is_dir()]
        assert files_all, "no json or spparks inputs found in {}".format(
            parent_dataset_path
        )

        n = len(files_all)

        # read validation and test sizes from data
        # if they are given as floats between 0-1, then
        # sizes determine fraction of samples in each split
        # Otherwise, they give the absolute number of samples
        # in each split.
        val_size = params["val_size"]
        if val_size > 0 and val_size < 1:
            val_size = int(val_size * n)
        test_size = params["test_size"]
        if test_size > 0 and test_size < 1:
            test_size = int(test_size * n)

        # indices for splitting single list of all files into train/val/test subsets
        train_lim = n - (val_size + test_size)
        val_lim = n - test_size

        # sort and shuffle with fixed seed to ensure reproducibility
        seed = params.get("random_seed", -1)
        if seed == -1:
            seed = np.random.randint(2**31)

        rng = np.random.default_rng(seed)

        files_all.sort()
        rng.shuffle(files_all)

        # split data into training, validation, testing
        train_files = files_all[:train_lim]
        val_files = files_all[train_lim:val_lim]
        test_files = files_all[val_lim:]

        # timesteps to keep between initial and final states of system,
        # can be manually entered with parameter "timesteps"
        timesteps = params.get("timesteps", None)

        # alternatively, it can be generated from parameter "n_timesteps"
        # if "timesteps" is not specified or null.
        # n_timesteps gives the number of timesteps between t=0 and t=tmax to keep.
        # Note that the initial and final steps are always included.

        # n == -1 --> keep all states (may result in very large files)
        # n == 0 --> only keep initial and final states
        # n = 1 --> keep state at t=0, t=tmax//2, t=tmax (ie 1 intermediate state)
        # etc

        # compute params that can be passed to np.linspace
        if timesteps is None:
            nsteps = params["n_timesteps"]
            if nsteps != -1:
                nsteps = nsteps + 2  # + 1 for t=0, +1 for t=tmax
        else:
            nsteps = None

        # record parameters and metrics
        mlflow.log_params(
            {
                "seed": params["random_seed"],
                "num_train": len(train_files),
                "num_val": len(val_files),
                "num_test": len(test_files),
            }
        )

        if timesteps is not None:
            for i, t in enumerate(timesteps):
                mlflow.log_metric("timesteps", t, step=i)
        else:
            mlflow.log_metric("n_timesteps", nsteps)

        # process graphs in parallel
        for subset, files in zip(
            ("train", "val", "test"), (train_files, val_files, test_files)
        ):
            print("processing {}: {} files".format(subset, len(files)))

            # create directories to store train/val/test data
            out_dir = Path(output_dataset_path, subset)
            os.makedirs(out_dir, exist_ok=True)

            # pack arguments into single tuple compatible with multiprocessing.map
            args = [
                (f, od, ts, ns)
                for f, od, ts, ns in zip(
                    files, repeat(out_dir), repeat(timesteps), repeat(nsteps)
                )
            ]

            # process graphs in parallel
            with get_context("fork").Pool(processes=cpu_count()) as pool:
                pool.map(process_graph, args)


if __name__ == "__main__":
    main()
