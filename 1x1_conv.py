# Implementation of a 1x1 convolution via. matrix multiplications. We use the
# following shapes for inputs and outputs. We assume number of batches == 1:
#
# weights = (960, 1, 1, 160)
# inputs  = (7, 7, 160)
# bias    = (960,)
# output  = (7, 7, 960)
#
#

input_S = 0.15070965886116028
input_Z = 134
weights_S = 0.0015944414772093296
weights_Z = 127
bias_S = 0.00024029774067457765
bias_Z = 0
output_S = 0.023528477177023888
output_Z = 0

weights_shape = (960, 1, 1, 160)
input_shape = (7, 7, 160)
output_shape = (7, 7, 960)
bias_shape = (960,)

import numpy as np

weights = np.random.randint(0, 255, size=weights_shape, dtype='uint8')
input = np.random.randint(0, 255, size=input_shape, dtype='uint8')
bias = np.random.randint(-7000,7000, size=bias_shape, dtype='int32')


# assume stride = (1,1)

# a weight vector for output channel c is going to be the vector given by fixing
# the first index of weights. I.e.,
#
# W_c = weights[c][:][:][:]

# The input matrix for some channel c, column y is going to be
#
# I_yc = input[:][y][:]

print 'computing "optimized" 1x1 conv2d'

weights_i32 = np.array(weights, dtype='int32')
input_i32 = np.array(input, dtype='int32')

out_final = list()
for c in range(output_shape[2]):

    # these can be prepared by the model owner
    W_c = weights_i32[c][:][:][:].flatten()
    W_c_ = W_c - weights_Z
    B_c = bias[c]
    out = list()
    for y in range(input_shape[1]):

        # can be prepared by input owner.
        In_yc = input_i32[:][y][:]

        # SS subtraction
        In_yc_ = In_yc - input_Z

        # matrix mult
        out.append(In_yc_.dot(W_c_))

    # process locally at each person
    out_final.append(np.vstack(out))
print 'done\n'

# process locally
# x0 = np.array(out_final).reshape(output_shape)
# print x0.shape
print 'computing reference 1x1 conv2d'

int32 = np.int32

output = np.zeros(output_shape)
for out_y in range(output_shape[0]):  # h_out
    for out_x in range(output_shape[1]):  # w_out
        for out_c in range(output_shape[2]):  # c_out
            acc = int32(0)
            # in_x_origin == out_x because stride == 1, pad_w == 0
            # ditto for in_y_origin
            for wy in range(weights_shape[1]):
                for wx in range(weights_shape[2]):
                    for c in range(weights_shape[3]):
                        in_x = out_x + wx
                        in_y = out_y + wy
                        if (0 <= in_x < input_shape[1]) \
                           and (0 <= in_y < input_shape[0]):

                            iv = input[in_y][in_x][c] - input_Z
                            wv = weights[out_c][wy][wx][c] - weights_Z
                            acc += iv*wv

            # acc += bias[out_c]
            output[out_y][out_x][out_c] = acc
            other = out_final[out_c][out_y][out_x]
            if acc != other:
                print 'acc=%s, other[out_y][out_x][out_c]=%s' % (
                    acc, other
                )
                print 'out_c=%s, out_x=%s, out_y=%s', (
                    out_c, out_x, out_y
                )
                print other[0][0][0]
                print output[0][0][0]
                exit(0)

print 'done'
