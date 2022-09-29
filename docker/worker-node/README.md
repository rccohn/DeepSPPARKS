Note:
The environment in this container works, but note the following:
  - It runs as UId 1000. If you have a different UID there may be permission issues while bind mounting files
  - Some of the paths used to mount files have changed, so some experiments may run into file not found error. I'll update this when I can, but in the meantime, see the example with worker-node-cpu in the getting-started section of this repository for a working example.
  
Containers for running mlFlow experiments and data analasis/exploration with jupyter. Supports GPU-accelerated computation
with nvidia container runtime.

There are two options:
  - rccohn/agg-nvdc-v3-base:latest contains directories for datasets, artifacts, 
    and inputs used to run mlflow projects. 
  - rccohn/agg-nvdc-v3-base:dex contains jupyter lab and can be used for data analysis, exploration, and visualization.



Note: deepspparks source is needed IN THE BUILD CONTEXT to work. Use the included tool (./container)  to build the container instead of running docker build by yourself. './container -b' builds the installation for running mlflow projects. './container -b --dex' builds the version for data exploration and analysis. './container' copies and Installs deepspparks as a module so it can be accessed with import deepsppark.
