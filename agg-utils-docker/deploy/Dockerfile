# start with base image, which contains standard libraries
FROM rccohn/agg-utils-docker-base:latest 

# execute commands in /home/ directory
WORKDIR /home/

# copy code to /home/
COPY AFRL_AGG AFRL_AGG

# install AFRL AGG code
RUN python -m pip install -e AFRL_AGG/

# set workdir back to /home/root
WORKDIR /home/root/
