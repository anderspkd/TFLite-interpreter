
import numpy as np
import math

def p_arr(arr):
    print '---------------------------------'
    print arr
    print '---------------------------------'

inputs_shape = (224, 224, 3)
weights_shape = (32, 3, 3, 3)
bias_shape = (32, )

nweights = np.prod(weights_shape)
ninputs = np.prod(inputs_shape)
weights = np.random.randint(0, 255, size=weights_shape, dtype='uint8')
weights = np.array(weights, dtype='int32')
inputs = np.random.randint(0, 255, size=inputs_shape, dtype='uint8')
inputs = np.array(inputs, dtype='int32')
bias = np.random.randint(-7000,7000, size=bias_shape, dtype='int32')

input_S = 0.0078125
input_Z = 128
weights_S = 0.03396892547607422
weights_Z = 122
bias_S = 0.00026538223028182983
bias_Z = 0
output_S = 0.023528477177023888
output_Z = 0

print 'weights:'
p_arr(weights)
print 'inputs:'
p_arr(inputs)

stride = (2, 2)
# sort of crude, but w/e
padding = (weights_shape[1] // 2, weights_shape[2] // 2)

output_shape = (int(math.ceil(float(inputs_shape[0])/stride[0])),
                int(math.ceil(float(inputs_shape[1])/stride[1])),
                weights_shape[0])
print 'shape of output:', output_shape

# this is the expensive part.

print 'prepping input ...'
inputs_per_channel = list()
for c in range(inputs_shape[2]):
    Z = list()
    for i in range(output_shape[0]):
        for j in range(output_shape[1]):
            # (x, y) is the top-left corner of the window that gives the output
            # at (i, j).
            x = i * stride[0]
            y = j * stride[1]
            z = list()
            # calculate window
            for a in range(weights_shape[1]):
                for b in range(weights_shape[2]):
                    xx = x + a - padding[0]
                    yy = y + b - padding[1]
                    if (0 <= xx < inputs_shape[0]) and \
                       (0 <= yy < inputs_shape[1]):
                        z.append(inputs[xx][yy][c] - input_Z)
                    else:
                        z.append(0)
            Z.append(z)
    inputs_per_channel.append(Z)
inputs_per_channel = np.array(inputs_per_channel, dtype='int32')

print 'done'

# this step effectively turns the shape of the weights from (Co, H, W, Ci) to
# (Co, Ci, H*W)
flat_weights = list()
for c in range(weights_shape[0]):
    ww = list()
    for c_ in range(weights_shape[3]):
        w = list()
        for a in range(weights_shape[1]):
            for b in range(weights_shape[2]):
                w.append(weights[c][a][b][c_])
        ww.append(np.array(w, dtype='uint8'))
    flat_weights.append(ww)

# now we can perform the actual convolution
output = list()
for c in range(weights_shape[0]):
    output_for_c = np.zeros(output_shape[:-1])
    for c_ in range(inputs_shape[2]):
        W_flat = flat_weights[c][c_]
        I_c = np.array(inputs_per_channel[c_], dtype='int32')
        r = I_c.dot(W_flat).reshape((output_shape[0], output_shape[1]))
        output_for_c += r
    output.append(output_for_c)

print np.array(output, dtype='int32')
