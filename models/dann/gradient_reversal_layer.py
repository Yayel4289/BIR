import torch
import torch.nn as nn

class GradientReversalLayer(nn.Module):
    def __init__(self, lambda_hyperparam: float):
        super(GradientReversalLayer, self).__init__()
        self.lambda_hyperparam = lambda_hyperparam

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_hyperparam)

class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_hyperparam: float): # ctx = context, just saving lambda
        ctx.lambda_hyperparam = lambda_hyperparam
        return x

    @staticmethod
    def backward(ctx, *grad_outputs):
        return grad_outputs * -ctx.lambda_hyperparam, None
        # same nbr of outputs as nbr of inputs in forward
