FROM nvidia/cuda:11.1.1-base-ubuntu18.04

# Set the default shell
SHELL ["/bin/bash", "-c"]

# Set character encoding environment variables
ENV LC_ALL=C.UTF-8 LANG=C.UTF-8

# Allow apt-get install without interaction from console
ENV DEBIAN_FRONTEND=noninteractive

# Set the working directory to /home
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
# env.yml specifies environment name as "env"
# note: the default env.yml includes several large libraries 
#       cuda, pytorch, tensorflow, etc
#      For adding smaller packages, adding another 
#      RUN pip install command will be MUCH faster
#      than altering env.yml as Docker can use the 
#      existing layers for these larger libraries.
RUN conda env create --file env.yml
# Set conda environment as default pyhton environment (ie no need to source activate)
ENV PATH="/root/miniconda/envs/env/bin:${PATH}"

# There's some weird dependency issue with pycocotools and tensorflow
# installing numpy==1.19.5 causes some issue with pycocotools.mask.encode (invalid C char)
# installing numpy==1.21.4 causes issue with tensorflow (Pip failed: The conflict is caused by: The user requested numpy==1.21.4
#    tensorflow 2.5.0 depends on numpy~=1.19.2)
# HOWEVER... installing numpy==1.19.5 and then manually pip install --upgrade numpy==1.21.4 appears to work for some reason...
# thus we apply the workaround here
RUN pip install --upgrade numpy==1.21.4

# Copy sppark2graph code and install
COPY deepspparks deepspparks
COPY setup.py .

# currently -e flag is needed to install correctly
RUN pip install -e .

# make directories that experiments will use
RUN mkdir /root/.ssh # for ssh files (mount as read only, only if sftp artifact store used)
RUN mkdir -p /root/data/datasets  # raw data (mount as read only)
RUN mkdir /root/data/processed  # processed dat (mount as read-write)
RUN mkdir /root/inputs  # for params.yaml and other input files
RUN mkdir /root/artifacts # temp storage for mlflow artifacts


