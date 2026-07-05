"""Sequential: chains modules so the output of one feeds the next."""

from tensorforge.nn.module import Module


class Sequential(Module):
    def __init__(self, *modules):
        self.modules = list(modules)

    def forward(self, x):
        for module in self.modules:
            x = module(x)
        return x
