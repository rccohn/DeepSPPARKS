# instruction to start jupyter inside docker
# note that external directories need to be mounted to docker container in 
# order to access them 


conda run -n env python -m jupyterlab `# start jupyter` \
	--no-browser `# browser cannot run inside container, access from host`\
        --ip localhost `#host ip` \
        --port ${JUPYTER_PORT_HOST} `# port ` \
	--NotebookApp.token='' `# disable token (local container -> no security threat)`\
	--NotebookApp.password='' `# disable password (local container -> no security threat)` \
	--notebook-dir ${HOME}/jupyter-home/${CONTAINER_NOTEBOOK_DIR} `# location to start jupyter`

