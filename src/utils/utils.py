
import torch

def to_device(device: torch.device, *tensors: torch.Tensor):
    if device is not None:
        device_tensors = [tensor.to(device) for tensor in tensors]
        return device_tensors if len(device_tensors) != 1 else device_tensors[0]
