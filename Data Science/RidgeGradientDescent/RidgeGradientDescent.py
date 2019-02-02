import numpy as np
import matplotlib.pyplot as plt


def generate_data(n):
    np.random.seed(0)
    d = 100
    w = np.zeros(d)
    for i in range(0,10):
        w[i] = 1.0
    #
    trainx = np.random.normal(size=(n,d))
    e = np.random.normal(size=(n))
    trainy = np.dot(trainx, w) + e
    #
    return trainx, trainy


def ridge_regression_GD(x, y, C, step):
    losses = []
    n, m = x.shape
    max_iter = 10000

    xPrime = np.hstack([np.ones((n, 1)), x])
    w = np.zeros(xPrime.shape[1])

    for i in range(max_iter):
        der = -2 * np.dot(xPrime.T, y - (np.dot(xPrime, w))) + 2 * C * w
        w = w - step * der

        loss = np.dot((y - np.dot(xPrime, w)).T, (y - np.dot(xPrime, w))) + C * np.dot(w.T, w)
        losses.append(loss)

    b = w[0]
    w = w[1:]

    return w, b, losses



# Generate 200 data points
n = 200
x,y = generate_data(n)
# Set regularization constant and step
C = 1.0
step = 0.0001
# Run gradient descent solver
w,b,losses = ridge_regression_GD(x,y,C,step)
print("The intercept is", b)
print("The first 10 meaningful dimensions coefficients are",w[0:10])
print("The loss is convergent to",losses[-1])
# Plot the losses
plt.plot(losses,'r')
plt.xlim((0,100))
plt.xlabel('Iterations', fontsize=14)
plt.ylabel('Loss', fontsize=14)
plt.show()
