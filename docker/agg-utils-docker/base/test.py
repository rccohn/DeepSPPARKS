# module imports
import torch
import numpy as np
import pandas as pd

# pandas can create df without numpy error
df = pd.DataFrame({'x': np.arange(15)})
print(df)


import torch_geometric

# RLE can encode/decode masks without "unexpected C char ID" error 
x = np.asfortranarray(np.ones(shape=(8,8), dtype=np.uint8))
from pycocotools.mask import encode, decode

rle = encode(x)
print('encode\n', rle)

rled = decode(rle)
print('decode\n',rled)
