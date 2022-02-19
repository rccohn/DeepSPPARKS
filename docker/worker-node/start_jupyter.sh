# instruction to start jupyter inside docker
# note that external directories need to be mounted to docker container in 
# order to access them 


python -m jupyterlab `# start jupyter` \
	--no-browser `# browser cannot run inside container, access from host`\
        --ip localhost `#host ip` \
        --port ${JUPYTER_PORT_HOST} `# port ` \
	--NotebookApp.token='' `# disable token (local container -> no security threat)`\
	--NotebookApp.password='' `# disable password (local container -> no security threat)` \
	--allow-root `# allow jupyter to run with root account (of container)` \
	--notebook-dir $(pwd)/host-files/${CONTAINER_NOTEBOOK_DIR} `# location to start jupyter`

