import numpy as np
import chainer.functions as F
from chainer import Variable


def target_func(x):
    y = 0.5 * x + 0.3
    return y

alpha = 0.1
a = Variable(np.array([1.0]).astype(np.float32))
b = Variable(np.array([1.0]).astype(np.float32))

for epoch in range(100):
    xx = np.random.normal(0, 1.0)
    x = np.array([xx]).astype(np.float32)
    y = a * x + b
    y_hat = target_func(x)

    err = F.mean_squared_error(y, y_hat)
    err.backward(True)
    a = a - alpha * a.grad
    b = b - alpha * b.grad

    if epoch % 10 == 0:
        print(epoch, err.data)

print("a = %f, b = %f" % (a.data, b.data))
