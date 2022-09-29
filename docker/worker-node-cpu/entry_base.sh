#! /bin/bash
# add non-root user with user ID that matches user on host
# this allows files saved to bind mounts to be opened and updated 
# without having to change the permissions after running experiments
adduser --disabled-password --gecos "" --uid ${DOCKER_UID} mlflow

# grant ownership of mounted files and artifacts temp dir to nonroot user
# note that using /root is bad practice, TODO update paths later
chown_dirs=(
${HOME}/volumes/datasets # raw data (mount as read only)
${HOME}/volumes/processed-data  # processed dat (mount as read-write)
${HOME}/volumes/inputs  # for params.yaml and other input files
/root/
/root/artifacts 
/root/inputs
)

for f in ${chown_dirs[@]};
do
    chown mlflow ${f}
done

export HOME=/home/mlflow
# directory added when running project with "mlflow run" command
mlflow_dir=/mlflow/projects/code
if [ -d ${mlflow_dir} ]; # container built by "mlflow run"
then
    # run in the mlflow project directory
    workdir=${mlflow_dir}
    chown -R mlflow ${workdir}
else # container executed manually by user
    # run in home directory
    workdir=/home/mlflow
fi

# run project as non-root user 
# with same UID as user who called mlflow run on host machine
# more info on why we need su-exec here 
# (they use gosu, which does the same thing but is larger/written in go instead of c):
# https://denibertovic.com/posts/handling-permissions-with-docker-volumes/
cd ${workdir} && export USER=mlflow PATH && \
	exec /usr/local/sbin/su-exec mlflow "$@"
