import numpy as np
import chainer.functions as F
import chainer.links as L
from chainer import Chain, optimizers


class Regression(Chain):

    def __init__(self):
        super(Regression, self).__init__(
            l=L.Linear(1, 1)
        )

    def train(self, x, y_hat):
        y = self.forward(x)
        loss = F.mean_squared_error(y, y_hat)
        return loss

    def forward(self, x):
        y = self.l(x)
        return y


def target_func(x):
    return 0.5 * x + 0.3

model = Regression()
optimizer = optimizers.SGD(0.1)
optimizer.setup(model)

for epoch in range(100):
    xx = [[np.random.normal(0, 1)] for i in range(100)]
    x = np.array(xx).astype(np.float32)
    y_hat = target_func(x)

    loss = model.train(x, y_hat)
    model.cleargrads()
    loss.backward()
    optimizer.update()

    if epoch % 10 == 0:
        print(epoch, loss.data)

print("a = %f, b = %f" % (model.l.W.data[0], model.l.b.data[0]))
