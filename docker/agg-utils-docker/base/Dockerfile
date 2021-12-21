FROM nvidia/cuda:11.1.1-base-ubuntu18.04

# Set the default shell
SHELL ["/bin/bash", "-c"]

# Set character encoding environment variables
ENV LC_ALL=C.UTF-8 LANG=C.UTF-8

# Allow apt-get install without interaction from console
ENV DEBIAN_FRONTEND=noninteractive

# Set the working directory to /root
WORKDIR /home/

# System updates and configurations
RUN apt-get update && apt-get -y --no-install-recommends install \
		ca-certificates \
		git \
		ssh \
		wget \ 
		build-essential && \
	apt-get clean && \
	apt-get autoremove && \
	rm -rf /var/lib/apt/lists/*

# Get network debugging tools
RUN apt update && apt-get -y install netcat iputils-ping iptables

# Install Miniconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
	bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda && \
	rm Miniconda3-latest-Linux-x86_64.sh

# Set the path env to inclcude miniconda
ENV PATH /root/miniconda/bin:$PATH

# copy env files to root
COPY env.yml env.yml

# Install venv
# env.yml must specify environment name as "env"
RUN conda env create --file env.yml

# Set conda environment as default pyhton environment (ie no need to source activate)
ENV PATH="/root/miniconda/envs/env/bin:${PATH}"

# There's some weird dependency issue with pycocotools...
# HOWEVER... installing numpy==1.19.5 and then 
# manually pip install --upgrade numpy==1.21.4 appears to work for some reason...
# thus we apply the workaround here
RUN pip install --upgrade numpy==1.21.4

# Default directory when attaching to container and/or running commands
WORKDIR /home/root

