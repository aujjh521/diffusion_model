import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

nClass = 4000
text_dim = 256
cond_embed = nn.Embedding(nClass, text_dim)

tensor = torch.from_numpy(np.array([0,0,0,0,0,0,0,0,0,0,-1,1,0,0]).astype(np.float32)).requires_grad_()

print(tensor)

dense = nn.Linear(14, 14)
print(dense(tensor))



