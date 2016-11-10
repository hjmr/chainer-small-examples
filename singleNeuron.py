import numpy as np
import chainer.functions as F
import chainer.links as L
from chainer import Chain, Variable, optimizers


class SingleNeuron(Chain):

    def __init__(self, input_size, output_size):
        super(SingleNeuron, self).__init__(
            l=L.Linear(input_size, output_size),
        )

    def train(self, x, y_hat):
        y = self.forward(x)
        loss = F.mean_squared_error(y, y_hat)
        return loss

    def forward(self, x):
        y = F.relu(self.l(x))
        return y


def target_func(x):
    return x[0] * 0.3 + x[1] * -1.0 + 0.5


num_points = 1000
vectors_set = []
for i in range(num_points):
    x1 = np.random.normal(0.0, 1.0)
    x2 = np.random.normal(0.0, 1.0)
    y = target_func([x1, x2])
    vectors_set.append([x1, x2, y])

x_data = np.array([[v[0], v[1]] for v in vectors_set]).astype(np.float32)
y_data = np.array([[v[2]] for v in vectors_set]).astype(np.float32)

model = SingleNeuron(2, 1)
optimizer = optimizers.Adam()
optimizer.setup(model)

bs = 100  # batch size
for epoch in range(1000):
    accum_loss = None
    perm = np.random.permutation(num_points)
    for i in range(0, num_points, bs):
        x_sample = Variable(x_data[perm[i:(i + bs) if(i + bs < num_points) else num_points]])
        y_sample = Variable(y_data[perm[i:(i + bs) if(i + bs < num_points) else num_points]])

        model.cleargrads()
        loss = model.train(x_sample, y_sample)
        loss.backward()
        optimizer.update()
        accum_loss = loss if accum_loss is None else accum_loss + loss

    if epoch % 100 == 0:
        print(epoch, accum_loss.data)
        accum_loss = None
