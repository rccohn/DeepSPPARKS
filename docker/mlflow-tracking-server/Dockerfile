# make sure python and mlflow versions match deepspparks-worker container!
FROM python:3.8.13-slim-buster  

# build arguments
ARG USER=deepspparks
ARG USER_UID=1000
ARG USER_GID=${USER_UID}

# install dependencies (ssh for sftp backend)
RUN apt-get update && apt-get -y --no-install-recommends install \
      ca-certificates  openssh-client \
	  && apt-get clean && apt-get autoremove \
	  && rm -rf /var/lib/apt/lists/*

# add non-root user
# more info here: https://code.visualstudio.com/remote/advancedcontainers/add-nonroot-user
ENV USER=${USER} \
    USER_UID=${USER_UID} \
    USER_GID=${USER_GID} \
    HOME=/home/${USER}

RUN groupadd --gid=${USER_GID} ${USER} \
    && useradd --uid ${USER_UID} --gid ${USER_GID} -m ${USER}

USER ${USER}

# add virtual environment, add to path
WORKDIR ${HOME}
ENV VENV=${HOME}/venv
ENV PATH=${VENV}/bin:${PATH}

COPY --chown=${USER}:${USER} requirements.txt requirements.txt    

RUN python3 -m venv ${VENV} && pip install --upgrade pip \
    && pip install -r requirements.txt \
    && rm requirements.txt

RUN mkdir ${HOME}/.ssh
