TORCH="1.7.0"
CUDA="cpu"

source activate env && \
    conda install -y  pytorch==1.7.0 cpuonly -c pytorch && \
    pip install --no-index torch-scatter -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html && \ 
    pip install --no-index torch-sparse -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html && \ 
    pip install --no-index torch-cluster -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html && \
    pip install --no-index torch-spline-conv -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html && \
    pip install torch-geometric
    
 

