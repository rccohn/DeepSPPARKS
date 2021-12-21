# start with base image, which contains CUDA, miniconda, torch, etc...
FROM rccohn/agg-nvdc-v3-base:latest

# execute commands in /home/ directory
WORKDIR /home/

# copy code to /home/
COPY AFRL_AGG AFRL_AGG

# install AFRL AGG code
RUN python -m pip install -e AFRL_AGG/

# set workdir back to /home/root
WORKDIR /home/root/

