import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

nClass = 4000
text_dim = 256
cond_embed = nn.Embedding(nClass, text_dim)

tensor = torch.from_numpy(np.array([0,0,0,0,0,0,0,0,0,0,-1,1,0,0]).astype(np.float32)).requires_grad_()

print(tensor.view(-1,1))
print(tensor.shape)

dense = nn.Linear(14, 14)
print(dense(tensor))

input_tensor = torch.LongTensor([[0,0,1,9], [1,1,1,0]])
embedding_layer = nn.Embedding(10, 64)
print(embedding_layer(input_tensor).shape)
