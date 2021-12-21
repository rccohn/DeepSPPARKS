# similar test from ../base/test.py
# but adds simple test to make sure graph can be created 

import torch
print('cuda: ', torch.cuda.is_available())
import numpy as np
import pandas as pd


df = pd.DataFrame({'x': np.arange(15)})
print(df)

import tensorflow

import torch_geometric

x = np.asfortranarray(np.ones(shape=(8,8), dtype=np.uint8))
from pycocotools.mask import encode, decode

rle = encode(x)
print('encode\n', rle)

rled = decode(rle)
print('decode\n',rled)

from src.graphs import Graph

g = Graph()

g.add_node(1)
g.add_node(3)
g.add_edge(1,3)

print(g.adj_)
g.to_json('graph.json')
g2 = Graph.from_json('graph.json')
print(g2.adj_)
print('done!')


