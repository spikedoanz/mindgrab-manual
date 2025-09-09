#from tinygrad.nn.state import torch_load

import torch
from torch.nn import Sequential

state_dict = torch.load("weights/tissue_fast/model.pth", weights_only=False)
if isinstance(state_dict, Sequential):
    state_dict = state_dict._modules
for k,v in state_dict.items():
    print(k, v)
