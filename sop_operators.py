# sop ops
#
# Implementation of the convolution etc operators that focus on a
# sum-of-products style approach.
#
# Because we want to be constitent with the interface in fake_mpc_interpreter.py
# we have to move the arguments around a bit.

import quantization
import numpy as np
from sys import stdout

# np.set_printoptions(threshold=np.nan)

def p(string):
    stdout.write(string)
    stdout.flush()


def offset_with_size(dim0, dim1, dim2):
    def f(i0, i1, i2):
        # offset into a 3D tensor
        return ((dim1 * i0 + i1) * dim2) + i2
    return f


def _flatten_weights(weights, shape):
    # helper function for conv2d and dwconv2d
    oc, wh, ww, ic = shape
    data = np.zeros((oc, ic * wh * ww))
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
    window = np.zeros((size_x * size_y, ))
    max_x = inputs.shape[2]
    max_y = inputs.shape[1]
    for i in range(size_x):
        for j in range(size_y):
            x_ = x + i
            y_ = y + j
            if (0 <= x_ < max_x) and (0 <= y_ < max_y):
                window[j * size_y + i] = inputs[0][y][x][channel]
    return window


def _get_full_window(inputs, x, y, size_x, size_y):
    # as above, except we extract all windows across all input channels
    windows = list()
    for c in range(inputs.shape[3]):
        windows.append(_get_window_for_channel(
            inputs, x, y, size_x, size_y, c))
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
    _wd = np.array(weights.data, dtype='int32')
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
    wd = np.array(weights.data, dtype='int32')
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


qmult = quantization.quantized_multiplier_mult

def _conv2d(input_data, input_offset, input_shape, output_offset, output_shape,
            weights_data, weights_shape, bias_data, stride, padding, shift,
            multiplier):
    output_w = output_shape[2]
    output_h = output_shape[1]
    output_data = np.zeros(output_shape)
    for out_y in range(output_h):
        for out_x in range(output_w):
            for out_c in range(output_shape[3]):
                x = (out_x * stride[0]) - padding[0]
                y = (out_y * stride[1]) - padding[1]
                window = _get_full_window(input_data, x, y, weights_shape[1],
                                          weights_shape[2])
                window -= input_offset
                # compute sum-of-products (dot product)
                z = window.dot(weights_data[out_c])
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
    fw = np.zeros((height * width))
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
    output_data = np.zeros(output_shape)
    for out_y in range(output_h):
        for out_x in range(output_w):
            for in_c in range(input_shape[3]):
                # assume depth multiplier == 1
                x = (out_x * stride[0]) - padding[0]
                y = (out_y * stride[0]) - padding[1]
                window = _get_window_for_channel(
                    input_data, x, y, weights_shape[1], weights_shape[2], in_c)
                weights = _flat_filter_for_channel(weights_data, in_c)
                # same deal now as in conv2d
                z = window.dot(weights)
                z += bias_data[in_c]
                z = qmult(z, multiplier, shift)
                z += output_offset
                z = min(255, max(0, z))
                output_data[0][out_y][out_x][in_c] = z
    return output_data


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


def _avgpool2d(input_data, input_shape, output_shape, stride, filter_size):
    for out_y in range(output_shape[1]):
        for out_x in range(output_shape[2]):
            for in_c in range(input_shape[3]):
                # assume VALID padding
                x = out_x * stride[0]
                y = out_y * stride[1]
                # filter end
                fxe = min(filter_size[1], input_shape[2] - x)
                fye = min(filter_size[0], input_shape[1] - y)
                acc = 0
                div = 1
