# quick check to ensure there aren't major dependency issues and simple 
# code can actually run
import torch
# verify cuda is detected
print('cuda: ', torch.cuda.is_available())
import numpy as np
import pandas as pd

# verify pandas works with no numpy error
df = pd.DataFrame({'x': np.arange(15)})
print(df)

# verify tensorflow loads
import tensorflow

import torch_geometric

# verify RLE encode and decode work without numpy C char error
x = np.asfortranarray(np.ones(shape=(8,8), dtype=np.uint8))
from pycocotools.mask import encode, decode

rle = encode(x)
print('encode\n', rle)

rled = decode(rle)
print('decode\n',rled)
