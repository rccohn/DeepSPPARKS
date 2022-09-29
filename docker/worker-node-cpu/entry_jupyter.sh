#! /bin/bash

# add non-root user with user ID that matches user on host
# this allows files saved to bind mounts to be opened and updated 
# without having to change the permissions after running experiments
adduser --disabled-password --gecos "" --uid ${DOCKER_UID} mlflow
export HOME=/home/mlflow

chown mlflow /home/jupyter
mkdir -p /home/mlflow/.jupyter/lab
mv /home/jupyter-user-settings /home/mlflow/.jupyter/lab/user-settings

# configure jupyter password
JUPYTER_PASS_HASH=$(python -c "from notebook.auth import passwd; \
    print(passwd('${MLFLOW_TRACKING_PASSWORD}'))")

# conda run correctly activates the conda environment, but buffers stdout when running jupyter
# Running jupyter with the python binary in the file seems to work, but if you start encountering errors,
# try running python with "conda run -n env python <args>"
exec /usr/local/sbin/su-exec mlflow python -m jupyterlab `# start jupyter` \
	--no-browser `# browser cannot run inside container, access from host` \
	--ip 0.0.0.0 `#host ip` \
	--NotebookApp.token='' `# disable token (local container -> no security threat)` \
	--NotebookApp.password=${JUPYTER_PASS_HASH} \
	--notebook-dir /home/jupyter `# location to start jupyter` \
	$@ # add extra arguments from command line
