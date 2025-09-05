from tinygrad.nn.state import torch_load

state_dict = torch_load("model.pth")
print("Available keys:", list(state_dict.keys()))
