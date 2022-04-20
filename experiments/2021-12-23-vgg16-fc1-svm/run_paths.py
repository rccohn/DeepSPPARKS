def pca_path(
    crop: int,
    whiten: int,
    local: bool,
    local_path_base: str = "/root/artifacts/",
    remote_path_base: str = "models/",
    run_id: str = "",
) -> str:
    """
    Generate local or remote path to save/load pca model from.
    Distinguishes between pca models for cropped/uncropped inputs, with/without
    whitening.

    Parameters:
        crop: int
            1 if pca was fit to cropped data, otherwise 0
        whiten: int
            1 if pca whitening was applied, otherwise 0
        local: bool
            if True, will generate path to directory on local machine.
            else, will generate a path in a remote mlflow tracking server
        local_path_base: str
            parent directory to store mlflow model in on local machine, used
            if local == True
        remote_path_base: str
            relative path to mlflow model in artifact repository, used if
            local == False
        run_id: str
            (only used if local == False)
            run id to store/load remote model from
            if specified, a full mlflow run uri will be generated
            (runs:/run_id/path/to/model)
            Otherwise, the runs:/run_id prefix will not be added

    """
    crop_str = "".join(("no_" * (1 - crop), "crop"))
    whiten_str = "".join(("no_" * (1 - whiten), "whiten"))
    rel_path = "".join(("pca-", crop_str, whiten_str))
    if local:
        base = local_path_base
    else:
        if run_id:
            # generate mlflow uri runs:/run_id/path/to/mlflow_model
            run_str = "".join(("runs:/", run_id, "/"))
        else:
            run_str = run_id

        base = "".join(
            (
                run_str,
                remote_path_base,
            )
        )

    full_path = "".join((base, rel_path))

    return full_path
