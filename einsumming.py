import numpy as np


# 3 * 3
a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
# 3 * 2
b = np.array([[1, 1], [1, 1], [1, 1]])
# 2 * 3
c = np.array([[1, 2, 3], [2, 4, 7]])

# This shows that when slicing what is returned is considered a row and will not be broadcast against a matrix with matching column height
# a = np.ones((3, 3))
# for x in range(3):
#     # Sum columns to create images
#     print a[:, x].reshape((3, 1)) * b
#     print a[:, x] * c
#     print

# This is multiplying  each column as a row aross layer 1 weights and summing the columns to produce a result
# l2posimages[:, x] = np.sum(sortedl2Weights[:, x] * layer1weights, 1)
# econdLayerImagesIntermediate = np.einsum('ij,ki->jik', sorted2ndlayerweights, layer1weights)
# secondLayerImages = np.sum(np.einsum('ij,ki->jik', sorted2ndlayerweights, layer1weights), 1)

result = np.empty_like(c)
for x in range(3):
    # Sum columns to create images
    print a[:, x] * c
    print np.sum(a[:, x] * c, 1, keepdims=True)

    result[:, x] = np.sum(a[:, x] * c, 1)

print
print result


# this einsum doesnt actually perform the calculations from above, i don't think einsumming can help me here :(
print np.einsum('ij,ki->jki', a, c)
print
print np.sum(np.einsum('ij,ki->jki', a, c), 0)

