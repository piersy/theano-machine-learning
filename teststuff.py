import numpy as np


# sorting
a = np.array([[3, 1, 1], [2, 2, 2], [1, 3, 3]])
print "unsorted"
print a
print "Sort column: ", a[:, 0]
print "sorted"
print a[a[:, 0].argsort()]
print np.sum(a)

# appending  a column on to a matrix
print np.arange(3)
a = np.append(a, np.transpose([np.arange(3)]), 1)
print "appending"
print a

# getting matrix sizes
print a.shape[0]

# slicing
# all but the final column
print a[:, :-1]
print np.arange(10)


# summing
ones = np.ones((2, 4))
print ones
print "sum axis 0 ", np.sum(ones, 0)
print "sum axis 1 ", np.sum(ones, 1)

# creating arrays
print np.arange(1, 11)

# iterating over vaules from arrays
for pos in np.nditer(np.arange(10)):
    print pos

# Multidimensional matrix multiplications
a = np.array([1, 2, 3])
a = np.vstack((a, a*2, a*3))
print a
b = np.ones((3, 3)) * 2

# well this is the einsum we are
# multiplying each column of a by b to create a new matrix and stacking them in a new dimension
print "einsum"
print np.einsum('ij,jk->ijk', a, b)


# print a.shape
# print a[np.newaxis].shape
# print a[np.newaxis, :].shape



# not too shure whats happenin here
# print a[np.newaxis, :, :] * b
# print a.T[:, np.newaxis] * b.T
# # a, b[:, :, np.newaxis]
