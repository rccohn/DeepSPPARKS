# OS and architecture, needed for base images and nvidia key rotation
# operating system
ARG dist_os=ubuntu
# version major
ARG version_major=18
# version minor
ARG version_minor=04
# architecture
ARG arch=x86_64
ARG distro=${dist_os}${version_major}${version_minor}

# docker CUDA version differs from cuda used in torch --> this is ok as cudatoolkit is 
# a dependency of pytorch. Using latest docker container gives better security.
FROM nvidia/cuda:11.6.0-base-${dist_os}${version_major}.${version_minor} AS base
# build arguments: non-root user
ARG USER=deepspparks
ARG USER_UID=1000
ARG USER_GID=${USER_UID}

# need to re-declare these args or else they will disappear (new build stage)
ARG distro
ARG arch

# Set the default shell for RUN commands
SHELL ["/bin/bash", "-c"]

# Set character encoding environment variables
ENV LC_ALL=C.UTF-8 LANG=C.UTF-8

# Allow apt-get install without interaction from console
ENV DEBIAN_FRONTEND=noninteractive

# nvidia key rotation (security update, needs to be applied or else 
# apt-get update will not work) 
# for more information see:
# https://developer.nvidia.com/blog/updating-the-cuda-linux-gpg-repository-key/
# Note that their 'recommended' approach does not seem to work in Docker (runs into dpkg issues)
# but their "not recommended" approach for manually installing the key works...

RUN apt-key del "7fa2af80" \
    && apt-key adv --fetch-keys "https://developer.download.nvidia.com/compute/cuda/repos/$distro/$arch/3bf863cc.pub"

# System updates and configurations
RUN apt-get update && apt-get -y --no-install-recommends install \
		ca-certificates \
		ssh \
		wget \ 
		build-essential \
		netcat \
		iputils-ping \
		iptables \
	&& apt-get clean \
	&& apt-get autoremove \
	&& rm -rf /var/lib/apt/lists/* # clear package lists to avoid unecessary storage

# user name and UID
# make available during runtime, not only when building container
ENV USER=${USER} \
    USER_UID=${USER_UID} \
    USER_GID=${USER_GID}

# ad non-root user
# more info here: https://code.visualstudio.com/remote/advancedcontainers/add-nonroot-user
RUN groupadd --gid=${USER_GID} ${USER} \
    && useradd --uid ${USER_UID} --gid ${USER_GID} -m ${USER}

USER ${USER}

# explicitely define ${HOME}
ENV HOME=/home/${USER} 

# Install Miniconda
WORKDIR ${HOME}
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && bash Miniconda3-latest-Linux-x86_64.sh -b -p ${HOME}/miniconda/ \
	&& rm Miniconda3-latest-Linux-x86_64.sh

# Set the path env to inclcude conda command
ENV PATH=${HOME}/miniconda/condabin:${PATH}

# copy env files to root
COPY --chown=${USER}:${USER} env.yml env.yml


# Install venv
# env.yml specifies environment name as "env"
# note: the default env.yml includes several large libraries 
#       cuda, pytorch, tensorflow, etc
#      For adding smaller packages, adding another 
#      RUN pip install command will be MUCH faster
#      than altering env.yml as Docker can use the 
#      existing layers for these larger libraries.
RUN conda env create --file env.yml && rm env.yml

# Copy deepspparks library and install (note this does not include dockerfiles or experiments,
# which should be managed outside of the container (ie mlflow run)
COPY --chown=${USER}:${USER} deepspparks deepspparks
COPY --chown=${USER}:${USER} setup.py .

# currently -e flag is needed to install correctly
# TODO eventually install via wheel and remove source
RUN conda run -n env pip install -e .

# Set default python binary environment created above (ie no need to source activate)
# NOTE this works for most cases, but can break if environment activation includes setting
# environment variables or other non-python items. 
# A workaround is to use "conda run -n env <command>"
# for more info see: https://pythonspeed.com/articles/activate-conda-dockerfile/
ENV PATH=${HOME}/miniconda/envs/env/bin:${PATH}


# build for running worker nodes
# makes directories for ssh, inputs, datasets, artifacts, etc
# used to run mlflow experiments
FROM base AS build1

# make directories that experiments will use
RUN mkdir -p ${HOME}/volumes/datasets # raw data (mount as read only)
RUN mkdir ${HOME}/volumes/processed-data  # processed dat (mount as read-write)
RUN mkdir ${HOME}/volumes/inputs  # for params.yaml and other input files
RUN mkdir -p ${HOME}/temp/artifacts # temp storage for mlflow artifacts

# for data exploration and analysis
# note that container should be run with --init 
# to avoid the PID 1 zombie process issue if using jupyter
FROM base AS build2

# install jupyter
COPY --chown=${USER}:${USER} requirements-dex.txt requirements.txt
RUN  conda run -n env python -m pip install -r requirements.txt \
     && rm requirements.txt
RUN mkdir ${HOME}/.jupyter
# default settings for jupyter (dark theme, show line numbers, folding, etc)

COPY --chown=${USER}:${USER} user-settings-jupyter ${HOME}/.jupyter/lab/user-settings

# convenient so you don't have to remember path inside jupyter
ENV HOSTFILES=${HOME}/jupyter-home/volumes/host-files

# create empty directory to mount local files
RUN mkdir -p ${HOSTFILES}

# start in ~/jupyter to include temp dir, so 
# scratch notebooks/temp files can be made instead of
# writing all files to host
WORKDIR ${HOME}/jupyter-home
# sample command for starting jupyter
COPY --chown=${USER}:${USER} start_jupyter.sh .
