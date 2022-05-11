# make sure python and mlflow versions match deepspparks-worker container!
FROM python:3.8.13-slim-buster  

# install dependencies (ssh for sftp backend)
RUN apt-get update && apt-get -y --no-install-recommends install \
	  apache2-utils nginx \
	  && apt-get clean && apt-get autoremove \
	  && rm -rf /var/lib/apt/lists/*

# add non-root user
# more info here: https://code.visualstudio.com/remote/advancedcontainers/add-nonroot-user
ENV USER=mlflow \
    USER_UID=1000 \
    USER_GID=1000

ENV HOME=/home/${USER}

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

EXPOSE 5000

USER root
# set nginx permissions to run as non root user
COPY set_permissions_nginx.sh set_permissions_nginx.sh
RUN bash set_permissions_nginx.sh && rm set_permissions_nginx.sh
COPY --chown=${USER}:${USER} nginx_conf /etc/nginx/sites-enabled/default
USER ${USER}

