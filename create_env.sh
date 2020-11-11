# create virtual environment with python 3.8.5
/usr/bin/python3.8 -m venv AGG_env
source AGG_env/bin/activate

# cuda 10.2 is installed on my system
# install pytorch and torchvision compatibile with cuda 10.2 and versions compatible with pytorch geometric
pip install torch==1.6.0 torchvision==0.7.0

CUDA=cu102
TORCH=1.6.0

pip install torch-scatter==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-${TORCH}.html
pip install torch-sparse==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-${TORCH}.html
pip install torch-cluster==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-${TORCH}.html
pip install torch-spline-conv==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-${TORCH}.html
pip install torch-geometric
