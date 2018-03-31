import numpy as np


class Evaluator(object):
    def __init__(self, f, shp):
        self.f, self.shp = f, shp

    def loss(self, x):
        loss_, self.grad_values = self.f([x.reshape(self.shp)])
        return loss_.astype(np.float64)

    def grads(self, x):
        return self.grad_values.flatten().astype(np.float64)