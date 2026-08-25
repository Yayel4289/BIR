import torch
import torch.nn as nn

class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_hyperparam: float):
        ctx.lambda_hyperparam = lambda_hyperparam
        return x

    @staticmethod
    def backward(ctx, grad_outputs):
        return grad_outputs * -ctx.lambda_hyperparam, None
        # same nbr of outputs as nbr of inputs in forward
