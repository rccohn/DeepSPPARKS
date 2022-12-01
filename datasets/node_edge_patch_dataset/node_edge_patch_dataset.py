from deepspparks.graphs import Graph
from deepspparks.utils import load_params
import deepspparks.paths as dp
import mlflow
from multiprocessing import get_context, cpu_count
import os
from pathlib import Path

from itertools import repeat
from shutil import rmtree
from subprocess import check_output
import patch_compression as pc


# wrapper functions for multiprocessing
def save_node_patch(args):
    g = args[0]
    node_idx = args[1]
    i = args[2]
    node_patch_root = args[3]
    node_patch_n = args[4]
    bounds = args[5]
    patch = g.node_patch(node_idx, bounds=bounds)
    patch_c = pc.compress_node_patch(patch)
    # assert pc.recover_node_patch(patch_c, 96) == patch
    with open(node_patch_root / "{:09d}".format(node_patch_n + i), "w") as f:
        f.write(patch_c)


def save_edge_patch(args):
    g = args[0]
    edge_idx = args[1]
    i = args[2]
    edge_patch_root = args[3]
    edge_patch_n = args[4]
    edge_patch_size = args[5]
    edge_width = args[6]
    patch = g.edge_patch(edge_idx, edge_patch_size, edge_width)
    patch_c = pc.compress_edge_patch(patch)
    # patch_recover = pc.recover_edge_patch(patch_c, 48)
    # for k, v in patch_recover.items():
    #     if k != 'edge_coords':
    #         assert v == patch[k], '{}-{}-{}'.format(k,v,patch[k])
    #     else:
    #         ec2 = patch[k]
    #         for a1, a2 in zip(v, ec2):
    #             for a3, a4 in zip(a1, a2):
    #                 assert len(a3) == len(a4), '{}-{}'.format(len(a3), len(a4))
    #                 assert all(a3==a4), '  {}\n  {}\n  {}'.format(k, a3, a4)

    with open(edge_patch_root / "{:09d}".format(edge_patch_n + i), "w") as f:
        f.write(patch_c)


def main():
    print("parsing params")
    params = load_params(dp.PARAM_PATH)

    with mlflow.start_run(nested=False):
        # save the param file
        mlflow.log_artifact(dp.PARAM_PATH)
        # dataset from which images are generated
        parent_dataset = params["mlflow"]["parent_dataset"]
        # indices r1:r2, c1:c2 from rolled images used to extract node patch images
        node_patch_window = "{}_{}_{}_{}".format(*params["node_patches"]["window"])
        # patch size (rows, columns) for edge patch images
        edge_patch_size = params["edge_patches"]["size"]

        mlflow.set_tags(
            {
                "mlflow.runName": "node and edge patches",
                "parent_dataset": parent_dataset,
            }
        )
        edge_width = params["edge_patches"]["width"]
        mlflow.log_params(
            {
                "node_patch_window": node_patch_window,
                "edge_patch_size": edge_patch_size,
                "edge_width": edge_width,
                "subgraph_radius": params["subgraph_radius"],
            }
        )

        output_dataset_name = "{}_patches_{}_{}_v{}".format(
            parent_dataset,
            node_patch_window,
            edge_patch_size,
            params["mlflow"]["dataset_version"],
        )
        mlflow.set_tag("dataset", output_dataset_name)
        parent_dataset_path = Path(dp.RAW_DATA_ROOT, parent_dataset)
        output_dataset_path = Path(dp.PROCESSED_DATA_ROOT, output_dataset_name)

        # TODO remove this or make it conditional? it was used a lot for testing
        #     to avoid previously written data
        if output_dataset_path.exists():
            rmtree(output_dataset_path)
        assert not output_dataset_path.exists()
        os.makedirs(output_dataset_path, exist_ok=False)
        # store patch info once so it is not replicated millions of times
        patch_sizes = [
            "node_window_idx: {}".format(node_patch_window),
            "edge: {}".format(edge_patch_size),
        ]
        with open((output_dataset_path / "patch_sizes.txt"), "w") as f:
            f.write(
                "\n".join(
                    patch_sizes,
                )
                + "\n"
            )

        subsets = ("train", "val", "test")
        n_files_total = 0
        total_num_nodes = 0
        total_num_edges = 0
        for sub in subsets:
            sub_root = parent_dataset_path / sub
            assert sub_root.is_dir(), "dir {} not found!".format(sub_root)

            # track total number of patches in each subset
            node_patch_n = 0
            edge_patch_n = 0

            node_patch_root = output_dataset_path / sub / "node"
            edge_patch_root = output_dataset_path / sub / "edge"
            for f in (node_patch_root, edge_patch_root):
                os.makedirs(f, exist_ok=True)

            # sort to ensure consistent ordering when run multiple times
            files = sorted(sub_root.glob("*.json"))
            n = len(files)
            n_files_total += n
            assert n, "dir {} has no json graphs!".format(sub_root)
            print("{}: processing {} files".format(sub_root, n))

            for f in files:
                g = Graph.from_json(f)
                g = g.get_subgraph(g.cidx[0], r=params["subgraph_radius"])
                # extract node patches
                nli = g.nli
                args = [
                    (g_, node_idx_, i, node_patch_root_, node_patch_n_, bounds_)
                    for i, (
                        g_,
                        node_idx_,
                        node_patch_root_,
                        node_patch_n_,
                        bounds_,
                    ) in enumerate(
                        zip(
                            repeat(g),
                            nli,
                            repeat(node_patch_root),
                            repeat(node_patch_n),
                            repeat(params["node_patches"]["window"]),
                        )
                    )
                ]

                with get_context("fork").Pool(processes=cpu_count()) as pool:
                    pool.map(save_node_patch, args)

                node_patch_n += len(nli)

                # extract edge patches
                eli = g.eli
                args = [
                    (
                        g_,
                        edge_idx_,
                        i,
                        edge_patch_root_,
                        edge_patch_n_,
                        edge_patch_size_,
                        edge_width_,
                    )
                    for i, (
                        g_,
                        edge_idx_,
                        edge_patch_root_,
                        edge_patch_n_,
                        edge_patch_size_,
                        edge_width_,
                    ) in enumerate(
                        zip(
                            repeat(g),
                            eli,
                            repeat(edge_patch_root),
                            repeat(edge_patch_n),
                            repeat(edge_patch_size),
                            repeat(edge_width),
                        )
                    )
                ]
                with get_context("fork").Pool(processes=cpu_count()) as pool:
                    pool.map(save_edge_patch, args)

                edge_patch_n += len(eli)

            # log total number of patches
            for patch_type, n in zip(("node", "edge"), (node_patch_n, edge_patch_n)):
                mlflow.log_metric("num_{}_{}".format(patch_type, sub), n)

            total_num_nodes += node_patch_n
            total_num_edges += edge_patch_n

        # log average disk space used per node and edge patch
        arg = "du -s --bytes {}/*/node".format(output_dataset_path.absolute().resolve())
        out = check_output(
            arg,
            executable="/bin/bash",
            shell=True,
        ).decode("utf-8")

        # values are reported as <num_bytes>\t<path/{train, val, test}>
        node_bytes = sum([int(x.split("\t")[0]) for x in out.strip("\n").split("\n")])
        arg = "du -s --bytes {}/*/edge".format(output_dataset_path.absolute().resolve())
        out = check_output(
            arg,
            executable="/bin/bash",
            shell=True,
        ).decode("utf-8")

        edge_bytes = sum([int(x.split("\t")[0]) for x in out.strip("\n").split("\n")])

        mlflow.log_metrics(
            {
                "average_storage_per_node-b": node_bytes / total_num_nodes,
                "average_storage_per_edge-b": edge_bytes / total_num_edges,
            }
        )


if __name__ == "__main__":
    # structure as
    # input folder: should have train, val, test subsets
    # output folder: save dataset_name/subset/{node, edge}
    main()
