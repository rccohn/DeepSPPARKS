# instruction to start jupyter inside docker
# note that external directories need to be mounted to docker container in 
# order to access them 

# conda run correctly activates the conda environment, but buffers stdout when running jupyter
# Running jupyter with the python binary in the file seems to work, but if you start encountering errors,
# try running python with "conda run -n env python <args>"
python -m jupyterlab `# start jupyter` \
	--no-browser `# browser cannot run inside container, access from host`\
        --ip localhost `#host ip` \
        --port ${JUPYTER_PORT_HOST} `# port ` \
	--NotebookApp.token='' `# disable token (local container -> no security threat)`\
	--NotebookApp.password='' `# disable password (local container -> no security threat)` \
	--notebook-dir ${HOME}/volumes/host-files/${CONTAINER_NOTEBOOK_DIR} `# location to start jupyter`

