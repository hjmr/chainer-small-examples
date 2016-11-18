#!/usr/bin/env python3

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

    def __call__(self, x, y_hat):
        y = self.forward(x)
        loss = F.mean_squared_error(y, y_hat)
        return loss

    def forward(self, x):
        h = F.relu(self.l1(x))
        o = self.l2(h)
        return o


def target_func(x):
    return x[0] * 0.3 + x[1] * -1.0 + 0.5


num_points = 1000
x_data = []
y_data = []
for i in range(num_points):
    x1 = np.random.normal(0.0, 1.0)
    x2 = np.random.normal(0.0, 1.0)
    y_hat = target_func([x1, x2]) + np.random.normal(0.0, 0.03)
    x_data.append([x1, x2])
    y_data.append([y_hat])

x_data = np.array(x_data).astype(np.float32)
y_data = np.array(y_data).astype(np.float32)

model = SimpleNN(2, 3, 1)
optimizer = optimizers.Adam(0.05)
optimizer.setup(model)

bs = 100
for epoch in range(1000):
    accum_loss = None
    perm = np.random.permutation(num_points)
    for i in range(0, num_points, bs):
        x_sample = x_data[perm[i:(i + bs) if(i + bs < num_points) else num_points]]
        y_sample = y_data[perm[i:(i + bs) if(i + bs < num_points) else num_points]]

        model.cleargrads()
        loss = model(x_sample, y_sample)
        accum_loss = loss if accum_loss is None else accum_loss + loss

    accum_loss.backward()
    optimizer.update()

    if epoch % 100 == 0:
        print(epoch, accum_loss.data)

# テスト
xx = [[x1 / 10.0, x2 / 10.0] for x1 in range(0, 10) for x2 in range(0, 10)]
x = Variable(np.array(xx).astype(np.float32), volatile='on')
y = model.forward(x)
y_hat = [[target_func(xt)] for xt in xx]
for y, y_hat in zip(y.data.tolist(), y_hat):
    print("%.2f <--> %.2f" % (y[0], y_hat[0]))
