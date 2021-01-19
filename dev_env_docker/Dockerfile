# start with ubuntu image
FROM ubuntu:20.10

# Set character encoding environment variables
ENV LC_ALL=C.UTF-8 LANG=C.UTF-8

# Allow apt-get install without interaction from console
ENV DEBIAN_FRONTEND=noninteractive

# Set the workign directory to /root
WORKDIR /root

# System updates and configurations
RUN apt-get update && apt-get -y --no-install-recommends install \
                ca-certificates \
                git \
                ssh \
                wget && \
        apt-get clean && \
        apt-get autoremove && \
        rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
        bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda && \
        rm Miniconda3-latest-Linux-x86_64.sh

# Set the path env to include miniconda
ENV PATH /root/miniconda/bin:$PATH

# Copy environment files 
COPY env.yaml /root/env.yaml
COPY install_extra_libraries.sh /root/install_extra_libraries.sh

# make shell script executable
RUN ["chmod", "+x", "/root/install_extra_libraries.sh"]

# Install base libraries
RUN conda env create --file /root/env.yaml

# Install additional torch libraries (cannot be in)
RUN ["bash", "/root/install_extra_libraries.sh"]

# copy environment activation to bashrc
RUN echo "source activate env" >> .bashrc 

# copy entrypoint script
COPY start.sh /root/start.sh

# make start script executable
RUN ["chmod", "+x", "/root/start.sh"]

# start session
ENTRYPOINT [ "/root/start.sh" ]

