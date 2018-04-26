#-*- coding: utf-8 -*-

import numpy as np
import chainer.functions as F
import chainer.links as L
from chainer import Chain, Variable, optimizers


class SimpleNN(Chain):

    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__(
            l1=L.Linear(input_size, hidden_size),
            l2=L.Linear(hidden_size, output_size),
        )

    def train(self, x, y_hat):
        y = self.forward(x)
        loss = F.mean_squared_error(y, y_hat)
        return loss

    def forward(self, x):
        h = F.relu(self.l1(x))
        o = self.l2(h)
        return o


def target_func(x):
    y = 0.5 * x[0] - 0.8 * x[1] + 0.3
    return y


model = SimpleNN(2, 3, 1)
optimizer = optimizers.Adam(0.05)
optimizer.setup(model)

for epoch in range(1000):

    xx = [[np.random.uniform(0, 1), np.random.uniform(0, 1)] for i in range(1000)]
    yy = [[target_func(x)] for x in xx]
    x = np.array(xx).astype(np.float32)
    y_hat = np.array(yy).astype(np.float32)

    model.cleargrads()
    loss = model.train(x, y_hat)
    loss.backward()
    optimizer.update()

    if epoch % 100 == 0:
        print(epoch, loss.data)

# テスト
xx = [[x1 / 10.0, x2 / 10.0] for x1 in range(0, 10) for x2 in range(0, 10)]
x = np.array(xx).astype(np.float32)
y = model.forward(x)
y_hat = [[target_func(xt)] for xt in xx]
for y, y_hat in zip(y.data.tolist(), y_hat):
    print("%.2f <--> %.2f" % (y[0], y_hat[0]))
