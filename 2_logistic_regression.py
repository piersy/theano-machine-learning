import matplotlib.pyplot as plt
import theano
from theano import tensor as T
import numpy as np
from load import mnist

def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)

def init_weights(shape):
    return theano.shared(floatX(np.random.randn(*shape) * 0.01))

def model(X, w):
    return T.nnet.softmax(T.dot(X, w))

trX, teX, trY, teY = mnist(onehot=True)

X = T.fmatrix()
Y = T.fmatrix()
# 10 possible classifications 784 is inputvector size sqrt(784) is input image side length
w = init_weights((784, 10))
py_x = model(X, w)
y_pred = T.argmax(py_x, axis=1)

# categorical_crossentropy basically tells theano to minimize the correct classification as defined by Y
cost = T.mean(T.nnet.categorical_crossentropy(py_x, Y))
gradient = T.grad(cost=cost, wrt=w)
update = [[w, w - gradient * 0.05]]

train = theano.function(inputs=[X, Y], outputs=cost, updates=update, allow_input_downcast=True)
predict = theano.function(inputs=[X], outputs=y_pred, allow_input_downcast=True)

for i in range(100):
    for start, end in zip(range(0, len(trX), 128), range(128, len(trX), 128)):
        # print "start:"+str(start)
        # print "end:"+str(end)
        cost = train(trX[start:end], trY[start:end])
    print i, np.mean(np.argmax(teY, axis=1) == predict(teX))

# show the learnt stuff

plt.figure(figsize=(1, 10))
for i in range(10):
    sub = plt.subplot(1, 10, i)
    t = w.get_value()[:, i].reshape((28, 28))
    sub.axis("off")
    sub.imshow(t, cmap=plt.cm.gray, interpolation="nearest")
plt.show()

# sub.imshow(true_face.reshape(image_shape),
#            cmap=plt.cm.gray,
#            interpolation="nearest")

# f = open('ramp.png', 'wb')      # binary mode is important
