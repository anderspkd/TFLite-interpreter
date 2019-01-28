# implementation of a 3x3 conv computation using matrices

import numpy as np
import math

print 'computing "optimized" 1x1 conv2d'

def print_channel(x, ch, padlen=5):
    sh = x.shape
    for i in range(sh[0]):
        s = ''
        for j in range(sh[1]):
            c = str(x[i][j][ch])
            p = '' if len(c) >= padlen else ' '*(padlen-len(c))
            s += str(x[i][j][ch]) + p
        print s

input_S = 0.0078125
input_Z = 128
weights_S = 0.03396892547607422
weights_Z = 122
bias_S = 0.00026538223028182983
bias_Z = 0
output_S = 0.023528477177023888
output_Z = 0

stride = (2,2)

weights_shape = (32, 3, 3, 3)
input_shape = (14, 14, 3)
output_shape = (input_shape[0]/stride[0], input_shape[1]/stride[1], weights_shape[0])
bias_shape = (weights_shape[0],)

weights = np.random.randint(0, 255, size=weights_shape, dtype='uint8')
input = np.random.randint(0, 255, size=input_shape, dtype='uint8')
bias = np.random.randint(-7000,7000, size=bias_shape, dtype='int32')

# convert weights to i32 arrays to avoid overflows.
weights_i32 = np.array(weights, dtype='int32')
input_i32 = np.array(input, dtype='int32')

# Transforming a specific weight tensor into the appropriate shape can be done
# using `flatten`.

# transforming the weight tensor requires a bit more fiddeling, though. We just
# do it in the most naive way, since this can be pre-processed....

input_i32_padded = np.pad(input_i32, ((1,1), (1,1), (0,0)), 'constant',
                          constant_values=input_Z)

input_i32_reshaped = list()
for j in range(0, input_shape[1], 2):
    rows = []
    for i in range(0, input_shape[0], 2):
        row = []
        for c in range(input_shape[2]):
            for a in range(weights_shape[1]):
                for b in range(weights_shape[2]):
                    row.append(input_i32_padded[i+a][j+b][c])
            # row.append(input_i32_padded[i    ][j    ][c])
            # row.append(input_i32_padded[i    ][j + 1][c])
            # row.append(input_i32_padded[i    ][j + 2][c])
            # row.append(input_i32_padded[i + 1][j    ][c])
            # row.append(input_i32_padded[i + 1][j + 1][c])
            # row.append(input_i32_padded[i + 1][j + 2][c])
            # row.append(input_i32_padded[i + 2][j    ][c])
            # row.append(input_i32_padded[i + 2][j + 1][c])
            # row.append(input_i32_padded[i + 2][j + 2][c])
        rows.append(row)
    input_i32_reshaped.append(rows)

input_i32_reshaped = np.array(input_i32_reshaped)
print 'transformed input:', input_i32_reshaped.shape

printme = True
out_final = list()
for c in range(output_shape[2]):
    W_c = weights_i32[c].flatten()
    W_c_ = W_c - weights_Z
    out = list()
    for y in range(input_i32_reshaped.shape[0]):
        In_yc = input_i32_reshaped[y][:][:]
        In_yc_ = In_yc - input_Z
        r = In_yc_.dot(W_c_)
        out.append(r)
        if printme:
            print 'In_yc_:'
            print In_yc_[0]
            print 'W_c_:'
            print W_c_
            print 'r:'
            print r[0]
            printme = False

    out_final.append(np.vstack(out))

print np.array(out_final).shape

print 'computing reference conv2d ...'

int32 = np.int32

output = np.zeros(output_shape, dtype='int32')
for out_y in range(output_shape[0]):  # h_out
    for out_x in range(output_shape[1]):  # w_out
        for out_c in range(output_shape[2]):  # c_out
            acc = int32(0)
            in_x_origin = (out_x * 2) - 1
            in_y_origin = (out_y * 2) - 1
            vals = ([],[])
            s = ''
            for wy in range(weights_shape[1]):
                for wx in range(weights_shape[2]):
                    for c in range(weights_shape[3]):
                        in_x = in_x_origin + wx
                        in_y = in_y_origin + wy
                        if (0 <= in_x < input_shape[1]) \
                           and (0 <= in_y < input_shape[0]):
                            iv = input[in_y][in_x][c] - input_Z
                            wv = weights[out_c][wy][wx][c] - weights_Z
                            vals[0].append(iv)
                            vals[1].append(wv)
                            acc += iv*wv

            # acc += bias[out_c]
            output[out_y][out_x][out_c] = acc
            other = out_final[out_c][out_y][out_x]
            # print acc, other
            if acc != other:
                print 'acc=%s, other[out_y][out_x][out_c]=%s' % (
                    acc, other
                )
                print 'out_c=%s, out_x=%s, out_y=%s' % (
                    out_c, out_x, out_y
                )
                print 'weights (%s): %s' % (len(vals[1]), vals[1])
                print 'inputs (%s): %s' % (len(vals[0]), vals[0])
                exit(0)

print 'done'

# # out_final = list()
# # for c in range(output_shape[2]):

# #     # these can be prepared by the model owner
# #     W_c = weights_i32[c][:][:][:].flatten()
# #     W_c_ = W_c - weights_Z
# #     B_c = bias[c]
# #     out = list()
# #     for y in range(input_shape[1]):

# #         # can be prepared by input owner.
# #         In_yc = input_i32[:][y][:]

# #         # SS subtraction
# #         In_yc_ = In_yc - input_Z

# #         # matrix mult
# #         out.append(In_yc_.dot(W_c_))

# #     # process locally at each person
# #     out_final.append(np.vstack(out))
# # print 'done\n'

# # # process locally
# # # x0 = np.array(out_final).reshape(output_shape)
# # # print x0.shape
# # print 'computing reference 1x1 conv2d'

# # int32 = np.int32

# # output = np.zeros(output_shape)
# # for out_y in range(output_shape[0]):  # h_out
# #     for out_x in range(output_shape[1]):  # w_out
# #         for out_c in range(output_shape[2]):  # c_out
# #             acc = int32(0)
# #             # in_x_origin == out_x because stride == 1, pad_w == 0
# #             # ditto for in_y_origin
# #             for wy in range(weights_shape[1]):
# #                 for wx in range(weights_shape[2]):
# #                     for c in range(weights_shape[3]):
# #                         in_x = out_x + wx
# #                         in_y = out_y + wy
# #                         if (0 <= in_x < input_shape[1]) \
# #                            and (0 <= in_y < input_shape[0]):

# #                             iv = input[in_y][in_x][c] - input_Z
# #                             wv = weights[out_c][wy][wx][c] - weights_Z
# #                             acc += iv*wv

# #             # acc += bias[out_c]
# #             output[out_y][out_x][out_c] = acc
# #             other = out_final[out_c][out_y][out_x]
# #             if acc != other:
# #                 print 'acc=%s, x0[out_y][out_x][out_c]=%s' % (
# #                     acc, other
# #                 )
# #                 print 'out_c=%s, out_x=%s, out_y=%s', (
# #                     out_c, out_x, out_y
# #                 )
# #                 print x0[0][0][0]
# #                 print output[0][0][0]
# #                 exit(0)

# # print 'done'
