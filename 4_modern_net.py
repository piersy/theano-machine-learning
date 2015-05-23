import matplotlib.pyplot as plt
import theano
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import numpy as np
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

for i in range(2):
    for start, end in zip(range(0, len(trX), 128), range(128, len(trX), 128)):
    # for start, end in zip(range(0, len(trX) / 20, 128), range(128, len(trX) / 20, 128)):
        cost = train(trX[start:end], trY[start:end])
    print np.mean(np.argmax(teY, axis=1) == predict(teX))

intermediate = teX[0, :].reshape((1, 784)).transpose() * w_h.get_value()

# max = intermediate.max()
# min = intermediate.min()
#
# print "min:"+min.astype('str')
# print "max:"+max.astype('str')

# intermediate += np.absolute(min)

print "max:" + intermediate.max().astype('str')
print "min:" + intermediate.min().astype('str')
print intermediate.shape


# I find the intermediates state sum to get intermediate activation values then plug that into the w_h and argsort


# add row of indices to bottom of matrix so that when sorted we know the index to look back into the w_h
# intermediate = np.append(intermediate, [np.arange(intermediate.shape[1])], 0)
# print "With indexes:", intermediate.shape

# print "Without indexes:", intermediate[:-1, :].shape

# print "Sum without indexes", np.sum(intermediate[:-1, :], axis=1)

sumshape = np.sum(intermediate, axis=0)
print "sum shape: ", sumshape.shape

# sort ith highest scoring columns first
intermediate = normalize(intermediate[:, sumshape.argsort()[::-1]])
w_h_value = normalize(w_h.get_value()[:, sumshape.argsort()[::-1]])
# intermediate = intermediate[np.sum(intermediate[:-1, :], axis=1).argsort()]
# intermediate /= intermediate.max()
# print "sorted intermediate: ", intermediate.shape
# plt.imshow(teX[0, :].reshape((28, 28)), cmap=plt.cm.gray, interpolation="nearest", vmin=0, vmax=1)
# plt.show()



print "w_h_value max: ",  np.max(intermediate)
print "w_h_value min: ",  np.min(intermediate)

def add_to_plot(plot, matrix, amount_to_plot, plot_offset):
    for pos in range(amount_to_plot):
        print pos
        sub = plt.subplot(height, width, pos+plot_offset+1)
        sub.axis("off")
        sub.imshow(matrix[:, pos].reshape((28, 28)), cmap=plt.cm.gray, interpolation="none", vmin=0, vmax=1)

height = 8
width = 5
plot_amount = 10
plt.figure(figsize=(height, width))
add_to_plot(plt, w_h_value, plot_amount, plot_amount*0)
add_to_plot(plt, intermediate, plot_amount, plot_amount*1)
add_to_plot(plt, w_h_value[:, ::-1], plot_amount, plot_amount*2)
add_to_plot(plt, intermediate[:, ::-1], plot_amount, plot_amount*3)
# for i in range(width * height / 2):
#     print "I: ", i
#     sub = plt.subplot(height, width, i+1)
#     sub.axis("off")
#     sub.imshow(w_h_value[:, i].reshape((28, 28)), cmap=plt.cm.gray, interpolation="none", vmin=0, vmax=1)
#
# for i in range((width * height / 2)+1, (width * height)+1):
#     print "new I: ", i
#     sub = plt.subplot(height, width, i)
#     sub.axis("off")
#     sub.imshow(intermediate[:, i].reshape((28, 28)), cmap=plt.cm.gray, interpolation="none", vmin=0, vmax=1)

plt.show()


# height = 1
# width = 10
#
# plt.figure(figsize=(height, width))
# for i in range(width*height):
# sub = plt.subplot(height, width, i)
# t = (((w_o.get_value()[:, i].reshape((625, 1)) * np.transpose(w_h2.get_value())).sum(axis=0)).reshape((625, 1)) * np.transpose(w_h.get_value())).sum(axis=0)
#     sub.axis("off")
#     sub.imshow(t.reshape((28, 28)), cmap=plt.cm.gray, interpolation="nearest")
# plt.show()