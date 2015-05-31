import numpy as np

import matplotlib.pyplot as plt
import theano
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from load import mnist

srng = RandomStreams()


def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)


def init_weights(shape):
    return theano.shared(floatX(np.random.randn(*shape) * 0.01))


def rectify(X):
    return T.maximum(X, 0.)


def softmax(X):
    e_x = T.exp(X - X.max(axis=1).dimshuffle(0, 'x'))
    return e_x / e_x.sum(axis=1).dimshuffle(0, 'x')


def RMSprop(cost, params, lr=0.001, rho=0.9, epsilon=1e-6):
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    for p, g in zip(params, grads):
        acc = theano.shared(p.get_value() * 0.)
        acc_new = rho * acc + (1 - rho) * g ** 2
        gradient_scaling = T.sqrt(acc_new + epsilon)
        g = g / gradient_scaling
        updates.append((acc, acc_new))
        updates.append((p, p - lr * g))
    return updates


def dropout(X, p=0.):
    if p > 0:
        retain_prob = 1 - p
        X *= srng.binomial(X.shape, p=retain_prob, dtype=theano.config.floatX)
        X /= retain_prob
    return X


def model(X, w_h, w_h2, w_o, p_drop_input, p_drop_hidden):
    X = dropout(X, p_drop_input)
    h = rectify(T.dot(X, w_h))

    h = dropout(h, p_drop_hidden)
    h2 = rectify(T.dot(h, w_h2))

    h2 = dropout(h2, p_drop_hidden)
    py_x = softmax(T.dot(h2, w_o))
    return h, h2, py_x


def normalize(matrix):
    matrix_min = np.min(matrix)
    # move min values to 0
    matrix -= matrix_min
    # divide by new max to bring everything in range 0 -1
    matrix /= np.max(matrix)
    return matrix


trX, teX, trY, teY = mnist(onehot=True)

# plt.imshow(teX[0, :].reshape((28, 28)), cmap=plt.cm.gray, interpolation="nearest")
# plt.show()



X = T.fmatrix()
Y = T.fmatrix()

w_h = init_weights((784, 625))
w_h2 = init_weights((625, 625))
w_o = init_weights((625, 10))

noise_h, noise_h2, noise_py_x = model(X, w_h, w_h2, w_o, 0.2, 0.5)
h, h2, py_x = model(X, w_h, w_h2, w_o, 0., 0.)
y_x = T.argmax(py_x, axis=1)

cost = T.mean(T.nnet.categorical_crossentropy(noise_py_x, Y))
params = [w_h, w_h2, w_o]
updates = RMSprop(cost, params, lr=0.001)

train = theano.function(inputs=[X, Y], outputs=cost, updates=updates, allow_input_downcast=True)
predict = theano.function(inputs=[X], outputs=y_x, allow_input_downcast=True)

for i in range(1):
    for start, end in zip(range(0, len(trX), 128), range(128, len(trX), 128)):
        # for start, end in zip(range(0, len(trX) / 20, 128), range(128, len(trX) / 20, 128)):
        cost = train(trX[start:end], trY[start:end])
    print np.mean(np.argmax(teY, axis=1) == predict(teX))

# First layer
# Pick an input and find the first layer intermediate state for that input
# find the most activated units of the first layer and then display those units
# purely by themselves and after activation from the input

# Second layer
# Pick an input and find the second layer intermediate state for that input
# find the most activated units of the second layer and for each unit of the
# second layer display the the sum of all images from the first layer after
# rectification each multiplied by the input weight of the second layer


def rectify(x):
    return x if x > 0 else 0


def add_to_plot(plot, matrix, amount_to_plot, plot_offset):
    for pos in range(amount_to_plot):
        sub = plot.subplot(height, width, pos + plot_offset + 1)
        sub.axis("off")
        sub.imshow(matrix[:, pos].reshape((28, 28)), cmap=plt.cm.gray, interpolation="none", vmin=0, vmax=1)


def get_sorted_weights(layer_input, weights):
    layer_output = np.sum(layer_input.transpose() * weights, 0)
    return weights[:, layer_output.argsort()[::-1]]


rectify = np.vectorize(rectify)

inputDigit = teX[0, :].reshape((1, 784))

layer1weights = w_h.get_value()
layer1 = inputDigit.transpose() * layer1weights
layer1_output = np.sum(layer1, axis=0, keepdims=True)
print "layer1_activations.shape", layer1_output.shape
# sort ith highest scoring columns first
layer1 = layer1[:, layer1_output.squeeze().argsort()[::-1]]

layer2weights = w_h2.get_value()

sortedl2Weights = get_sorted_weights(layer1_output, layer2weights)

# leave for later
# layer1weights_rectified = layer1weights[:, layer2input == 0]

l2imagecount = 10

l2posimages = np.empty((784, l2imagecount))
l2posimagesnew = np.empty((784, l2imagecount))
l2negimages = np.empty((784, l2imagecount))
l2negimagesnew = np.empty((784, l2imagecount))

l2Images = np.empty_like(layer1weights)


def calculate_next_block(a, b):
    next_block = np.empty((b.shape[0], a.shape[1]))
    for x in range(a.shape[1]):
        next_block[:, x] = np.sum(a[:, x] * b, 1)
    return next_block


def calculate_images_atlayer(layers):
    next_block = layers[0];
    for x in range(1, len(layers)):
        next_block = calculate_next_block(next_block, layers[x])

    return next_block


l2Images = calculate_images_atlayer([layer2weights, layer1weights])
print l2Images.shape

# for x in range(layer2weights.shape[1]):
# l2Images[:, x] = np.sum(layer2weights[:, x] * layer1weights, 1)

# layer2_output = np.sum(layer1_output.transpose() * layer2weights, 0)

# sorted l2 images is now correct i needed to add the input digit into the mix for sorting

sortedl2Images = l2Images[:, np.sum(inputDigit.transpose() * l2Images, 0).argsort()[::-1]]
# sortedl2Images = l2Images[:, layer2_output.argsort()[::-1]]

# for some reason an extra dimension is added if we use argsort with dimensions so we make
# layer2 output dimensionless by squeezing it
for x in range(l2imagecount):
    # Sum columns to create images
    l2posimages[:, x] = np.sum(sortedl2Weights[:, x] * layer1weights, 1)
    l2posimagesnew[:, x] = sortedl2Images[:, x]

for x in range(l2imagecount):
    # Sum across rows to create 1 image
    l2negimages[:, x] = np.sum(sortedl2Weights[:, -(x + 1)] * layer1weights, 1)
    l2negimagesnew[:, x] = sortedl2Images[:, -(x + 1)]

height = 8
width = 5
plot_amount = 10
plt.figure(figsize=(height, width))
add_to_plot(plt, normalize(l2posimages), plot_amount, plot_amount * 0)
# add_to_plot(plt, normalize(inputDigit.transpose() * l2posimages), plot_amount, plot_amount * 1)
add_to_plot(plt, normalize(l2posimagesnew), plot_amount, plot_amount * 1)
add_to_plot(plt, normalize(l2negimages), plot_amount, plot_amount * 2)
# add_to_plot(plt, normalize(inputDigit.transpose() * l2negimages), plot_amount, plot_amount * 3)
add_to_plot(plt, normalize(l2negimagesnew), plot_amount, plot_amount * 3)
plt.show()

# height = 4
# width = 5
# plot_amount = 10
# plt.figure(figsize=(height, width))
# add_to_plot(plt, normalize(secondLayerImages).T, plot_amount, plot_amount*0)
# add_to_plot(plt, normalize(inputDigit.T * secondLayerImages.T), plot_amount, plot_amount*1)
# plt.show()

layer1weights = layer1weights[:, layer1_output.squeeze().argsort()[::-1]]

height = 8
width = 5
plot_amount = 10
plt.figure(figsize=(height, width))
add_to_plot(plt, normalize(layer1weights), plot_amount, plot_amount * 0)
add_to_plot(plt, normalize(layer1), plot_amount, plot_amount * 1)
add_to_plot(plt, normalize(layer1weights[:, ::-1]), plot_amount, plot_amount * 2)
add_to_plot(plt, normalize(layer1[:, ::-1]), plot_amount, plot_amount * 3)
plt.show()