# Sum-of-Product operators
#
# Implementation of the convolution etc operators that focus on a
# sum-of-products style approach.
#
# Because we want to be constitent with the interface in fake_mpc_interpreter.py
# we have to move the arguments around a bit.

import quantization
import numpy as np
from sys import stdout


def p(string):
    stdout.write(string)
    stdout.flush()


def offset_with_size(dim0, dim1, dim2):
    def f(i0, i1, i2):
        # array offset into a 3D tensor
        return ((dim1 * i0 + i1) * dim2) + i2
    return f


def _flatten_weights(weights, shape):
    # helper function for conv2d and dwconv2d. Flattens the weights tensor along
    # rows for each output channel.
    oc, wh, ww, ic = shape
    data = np.zeros((oc, ic * wh * ww), dtype='int64')
    offset = offset_with_size(ic, wh, ww)
    for c in range(oc):
        for c_ in range(ic):
            for x in range(ww):
                for y in range(wh):
                    data[c][offset(c_, y, x)] = weights[c][y][x][c_]
    return data


def _get_window_for_channel(inputs, x, y, size_x, size_y, channel):
    # returns the window in channel `channel` of `inputs` with origin `x` and
    # `y`. `size_x` and `size_y` denote the size of the window. If the window
    # happen to be out of bounds, we pad with 0s
    window = list()
    max_x = inputs.shape[2]
    max_y = inputs.shape[1]
    for j in range(size_y):
        for i in range(size_x):
            x_ = x + i
            y_ = y + j
            if (0 <= x_ < max_x) and (0 <= y_ < max_y):
                window.append(inputs[0][y_][x_][channel])
            else:
                window.append(0)
    return np.array(window, dtype='int64')


def _get_full_window(inputs, x, y, size_x, size_y):
    # as above, except we extract all windows across all input channels. This is
    # used in conv2d, whereas the above is used in dwconv2d.
    windows = list()
    for c in range(inputs.shape[3]):
        window = _get_window_for_channel(inputs, x, y, size_x, size_y, c)
        windows.append(window)
    return np.hstack(windows)


def conv2d(options, inputs, weights, bias, output):
    shift, multiplier = quantization.compute_multiplier_for_conv2d(
        weights.scale, inputs.scale, output.scale
    )
    padding_h, padding_w = (weights.shape[1] // 2, weights.shape[2] // 2)
    # we flatten the weights and account for the offset before we input them to
    # the _conv2d function. Of course, this can be performed in a preprocessing
    # step.
    p('flattening weights ....')
    _wd = np.array(weights.data, dtype='int64')
    wd = _flatten_weights(_wd - weights.zero_point, weights.shape)
    print 'done'
    p('computing conv2d ....')
    x = _conv2d(
        # input
        inputs.data,
        inputs.zero_point,
        inputs.shape,
        # output
        output.zero_point,
        output.shape,
        # weights
        wd,
        weights.shape,
        # bias
        bias.data,
        # options
        options.stride,
        (padding_h, padding_w),
        shift,
        multiplier
    )
    print 'done'
    return x


def dwconv2d(options, inputs, weights, bias, output):
    shift, multiplier = quantization.compute_multiplier_for_conv2d(
        weights.scale, inputs.scale, output.scale
    )
    padding_h, padding_w = (weights.shape[1] // 2, weights.shape[2] // 2)
    wd = np.array(weights.data, dtype='int64')
    p('computing depthwise convolution ....')
    x = _dwconv2d(
        # input
        inputs.data,
        inputs.zero_point,
        inputs.shape,
        # output
        output.zero_point,
        output.shape,
        # weights
        wd - weights.zero_point,
        weights.shape,
        # bias
        bias.data,
        # options
        options.stride,
        (padding_h, padding_w),
        shift,
        multiplier
    )
    print 'done'
    return x


def avgpool2d(options, inputs, output):
    p('computing average pool ....')
    x = _avgpool2d(inputs.data,
                   inputs.shape,
                   output.shape,
                   options.stride,
                   options.filter_size
    )
    print 'done'
    return x


def add(options, input1, input2, output):
    shift1, multiplier1 = quantization.compute_multiplier_for_conv2d(
        input1.scale, 1.0, output.scale
    )
    shift2, multiplier2 = quantization.compute_multiplier_for_conv2d(
        input2.scale, 1.0, output.scale
    )
    p('computing residual ....')
    x = _add(input1.data,
             input1.zero_point,
             shift1,
             multiplier1,
             input2.data,
             input2.zero_point,
             shift2,
             multiplier2,
             output.shape,
             output.zero_point
    )
    print 'done'
    return x


qmult = quantization.quantized_multiplier_mult

def _conv2d(input_data, input_offset, input_shape, output_offset, output_shape,
            weights_data, weights_shape, bias_data, stride, padding, shift,
            multiplier):
    output_w = output_shape[2]
    output_h = output_shape[1]

    # we "preprocess" the windows before we compute the convolution. It makes
    # the presentation below a bit clearer and it means we don't have to spend
    # time on extracting each window in each loop. We need a window for each
    # output coordinate.
    #
    # The reason for doing this here rather than in a proper preprocessing step
    # is because the shape of the output of the previous layer does not match
    # the layout of these extracted windows (with an exception in the case where
    # the filters are 1x1, maybe).
    p('extracting windows ....')
    windows = list()
    for out_y in range(output_h):
        ys = list()
        for out_x in range(output_w):
            xs = list()
            for out_c in range(output_shape[3]):
                x = (out_x * stride[0]) - padding[0]
                y = (out_y * stride[1]) - padding[1]
                window = _get_full_window(input_data - input_offset, x, y,
                                          weights_shape[1], weights_shape[2])
                xs.append(window)
            ys.append(xs)
        windows.append(ys)
    p('done ....')

    # Compute the convolution which at this point is nothing more than a lot of
    # dot-products with a division + clamp at the end.
    output_data = np.zeros(output_shape, dtype='int64')
    for out_y in range(output_h):
        for out_x in range(output_w):
            for out_c in range(output_shape[3]):
                # sum-of-products
                z = windows[out_y][out_x][out_c].dot(weights_data[out_c])
                # add bias
                z += bias_data[out_c]
                # divide/truncate
                z = qmult(z, multiplier, shift)
                # add output offset
                z += output_offset
                # clamp
                z = min(255, max(0, z))
                output_data[0][out_y][out_x][out_c] = z
    return output_data


def _flat_filter_for_channel(weights, channel):
    # extract and flatten the filter at `channel`. Used as a helper for
    # _dwconv2d.
    height = weights.shape[1]
    width = weights.shape[2]
    fw = np.zeros((height * width), dtype='int64')
    for x in range(width):
        for y in range(height):
            # assume depth multiplier == 1
            fw[y * height + x] = weights[0][y][x][channel]
    return fw


def _dwconv2d(input_data, input_offset, input_shape, output_offset,
              output_shape, weights_data, weights_shape, bias_data, stride,
              padding, shift, multiplier):

    output_h = output_shape[1]
    output_w = output_shape[2]
    p('extracting windows and filters ....')
    filters = [_flat_filter_for_channel(weights_data, c) for c in range(input_shape[3])]
    windows = list()
    for out_y in range(output_h):
        ys = list()
        for out_x in range(output_w):
            xs = list()
            for in_c in range(input_shape[3]):
                x = (out_x * stride[0]) - padding[0]
                y = (out_y * stride[0]) - padding[1]
                window = _get_window_for_channel(
                    input_data, x, y, weights_shape[1], weights_shape[2], in_c
                )
                xs.append(window)
            ys.append(xs)
        windows.append(ys)
    p('done ....')

    # Compute a depthwise convoltion. A depthwise convolution is very similar to
    # a regular 2D convolution with the only difference being that we do not sum
    # up the dot-products over the input channels.
    output_data = np.zeros(output_shape, dtype='int64')
    for out_y in range(output_h):
        for out_x in range(output_w):
            for in_c in range(input_shape[3]):
                # assume depth multiplier == 1
                #
                # Same deal now as in conv2d
                z = windows[out_y][out_x][in_c].dot(filters[in_c])
                z += bias_data[in_c]
                z = qmult(z, multiplier, shift)
                z += output_offset
                z = min(255, max(0, z))
                output_data[0][out_y][out_x][in_c] = z
    return output_data


def _avgpool2d(input_data, input_shape, output_shape, stride, filter_size):
    output_data = np.zeros(output_shape)
    for out_y in range(output_shape[1]):
        for out_x in range(output_shape[2]):
            for c in range(input_shape[3]):
                # assume VALID padding
                x = out_x * stride[0]
                y = out_y * stride[1]
                # filter end
                fxe = min(filter_size[1], input_shape[2] - x)
                fye = min(filter_size[0], input_shape[1] - y)
                acc = 0
                # NB: `div` can be precomputed
                div = 0
                for fy in range(0, fye):
                    for fx in range(0, fxe):
                        x_ = x + fx
                        y_ = y + fy
                        acc += input_data[0][y_][x_][c]
                        div += 1
                # compute average. The accuracy of the model does not appear to
                # suffer if this division is changed to an imprecise one.
                acc = (acc + div / 2) / div
                acc = min(255, max(0, acc))
                output_data[0][out_y][out_x][c] = acc
    return output_data


def _add(input1_data, input1_offset, input1_shift, input1_multiplier,
         input2_data, input2_offset, input2_shift, input2_multiplier,
         output_shape, output_offset):
    # This operation should be easy to evaluate in parallel as all entries can
    # be computed independently of eachother.
    output_data = np.zeros(output_shape, dtype='int64')
    input1_data = input1_data - input1_offset
    input2_data = input2_data - input2_offset
    for out_y in range(output_shape[1]):
        for out_x in range(output_shape[2]):
            for out_c in range(output_shape[3]):
                in1 = qmult(input1_data[0][out_y][out_x][out_c], input1_multiplier, input1_shift)
                in2 = qmult(input2_data[0][out_y][out_x][out_c], input2_multiplier, input2_shift)
                out = output_offset + in1 + in2
                out = min(255, max(0, 255))
                output_data[0][out_y][out_x][out_c] = out
    return output_data


if __name__ == '__main__':
    np.set_printoptions(threshold=np.nan)
    i_shape = (1,128,128,3)
    w_shape = (8,3,3,3)
    o_shape = (1,64,64,8)
    padding_h, padding_w = (w_shape[1] // 2, w_shape[2] // 2)
    # i = np.random.randint(0, 255, size=i_shape, dtype='int32')
    i = np.array(np.arange(np.prod(i_shape), dtype='uint8').reshape(i_shape), dtype='int32')
    # w = np.random.randint(0, 255, size=w_shape, dtype='int32')
    w = np.array(np.arange(np.prod(w_shape), dtype='uint8').reshape(w_shape), dtype='int32')
    # b = np.random.randint(-5000, 5000, size=(8,), dtype='int32')
    b = np.arange(o_shape[3], dtype='int32')
    w_offset = 157
    w_scale = 0.008882409892976284
    i_offset = 128
    i_scale = 0.0078125
    b_offset = 0.00006939382728887722
    b_scale = 0.0
    o_offset = 0.0
    o_scale = 0.023528477177023888
    shift, mult = quantization.compute_multiplier_for_conv2d(i_scale, w_scale, o_scale)
    w = _flatten_weights(w - w_offset, w_shape)
    print _conv2d(i, i_offset, i_shape, o_offset, o_shape,
                  w, w_shape, b, (2, 2), (padding_h, padding_w), shift, mult)
