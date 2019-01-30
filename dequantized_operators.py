# Dequantized versions of conv2d, dwconv2d and avgpool2d

import numpy as np


# real_val = scale * (quant_val - zero_point)
def dequantize_tensor(tensor):
    print 'dequantizing tensor:', tensor.name
    S, Z = tensor.scale, tensor.zero_point
    return (np.array(tensor.data, dtype='float32') - Z) * S


# The name of the tensor includes the activation function, so that's what we'll
# use to determine whether or not ReLU6 should be applied.
def should_apply_relu6(tensor_name):
    return 'relu6' in tensor_name.lower()


# Compute a 2D convolution. Assume `inputs` is already FP32. Dequantizes
# `weights` and `bias`, and then computes the convolution as described in
# reference_ops.h (see the float version).
def conv2d(options, inputs, weights, bias, output):
    apply_relu6 = should_apply_relu6(output.name)

    print 'applying relu6:', apply_relu6

    weights = dequantize_tensor(weights)
    bias = dequantize_tensor(bias)

    n_channels_out, weights_h, weights_w, n_channels_in = weights.shape
    _, inputs_h, inputs_w, n_channels_in_ = inputs.shape
    _, output_h, output_w, n_channels_out_ = output.shape

    assert n_channels_in_ == n_channels_in
    assert n_channels_out_ == n_channels_out

    stride_h, stride_w = options.stride
    padding_h, padding_w = (weights_h // 2, weights_w // 2)

    print 'input shape: ', inputs.shape
    print 'output shape:', output.shape
    print 'weights shape:', weights.shape

    output.data = np.zeros(output.shape, dtype='float32')

    # assume num_batches == 0
    for out_y in range(output_h):
        for out_x in range(output_w):
            for out_c in range(n_channels_out):
                in_x_origin = (out_x * stride_w) - padding_w
                in_y_origin = (out_y * stride_h) - padding_h
                acc = 0.0
                for filter_y in range(weights_h):
                    for filter_x in range(weights_w):
                        for in_c in range(n_channels_in):
                            in_x = in_x_origin + filter_x
                            in_y = in_y_origin + filter_y
                            if (0 <= in_x < inputs_w) and \
                               (0 <= in_y < inputs_h):
                                iv = inputs[0][in_y][in_x][in_c]
                                wv = weights[out_c][filter_y][filter_x][in_c]
                                acc += iv * wv
                acc += bias[out_c]
                # relu6
                if apply_relu6:
                    acc = 0.0 if acc < 0 else acc
                    acc = 6.0 if acc > 6 else acc
                output.data[0][out_y][out_x][out_c] = acc

    return output.data


# Compute a depthwise 2D convolution. Same assumptions as `conv2d`.
def dwconv2d(options, inputs, weights, bias, output):

    apply_relu6 = should_apply_relu6(output.name)

    print 'applying relu6:', apply_relu6

    weights = dequantize_tensor(weights)
    bias = dequantize_tensor(bias)

    depth_multiplier = options.depth_multiplier

    _, weights_h, weights_w, output_depth = weights.shape
    _, inputs_h, inputs_w, n_channels_in = inputs.shape
    _, output_h, output_w, output_depth_ = output.shape

    assert output_depth_ == output_depth

    stride_h, stride_w = options.stride
    padding_h, padding_w = (weights_h // 2, weights_w // 2)

    print 'input shape: ', inputs.shape
    print 'output shape:', output.shape
    print 'weights shape:', weights.shape

    output.data = np.zeros(output.shape, dtype='float32')

    for out_y in range(output_h):
        for out_x in range(output_w):
            for in_c in range(n_channels_in):
                for m in range(depth_multiplier):
                    oc = m + in_c * depth_multiplier
                    in_x_origin = (out_x * stride_w) - padding_w
                    in_y_origin = (out_y * stride_h) - padding_h
                    acc = 0.0
                    for filter_y in range(weights_h):
                        for filter_x in range(weights_w):
                            in_x = in_x_origin + filter_x
                            in_y = in_y_origin + filter_y
                            if (0 <= in_x < inputs_w) and \
                               (0 <= in_y < inputs_h):
                                iv = inputs[0][in_y][in_x][in_c]
                                wv = weights[0][filter_y][filter_x][oc]
                                acc += (iv * wv)
                    acc += bias[oc]
                    if apply_relu6:
                        acc = 0.0 if acc < 0 else acc
                        acc = 6.0 if acc > 6 else acc
                    output[0][out_y][out_x][oc] = acc

    return output.data


# Compute an 2D average pool layer.
def avgpool2d(options, input, output):
    _, input_h, input_w, n_channels_in = input.shape
    _, output_h, output_w, n_channels_out = output.shape

    n = input_h * input_w
    print 'divisor: ', n

    assert options.padding == 'VALID'
    assert output_h == output_w == 1
    assert n_channels_in == n_channels_out

    padding_h, padding_w = (0, 0)
    stride_h, stride_w = options.stride
    filter_h, filter_w = options.filter_size

    output_ = np.zeros(output.shape, dtype='float32')

    for out_y in range(output_h):
        for out_x in range(output_w):
            for c in range(n_channels_in):
                in_x_origin = (out_x * stride_w) - padding_w
                in_y_origin = (out_y * stride_h) - padding_h
                fxs = max(0, -in_x_origin)
                fxe = min(filter_w, input_w - in_x_origin)
                fys = max(0, -in_y_origin)
                fye = min(filter_h, input_h - in_y_origin)
                acc = 0.0
                fc = 0
                for filter_y in range(fys, fye):
                    for filter_x in range(fxs, fxe):
                        in_x = in_x_origin + filter_x
                        in_y = in_y_origin + filter_y
                        acc += input[0][in_y][in_x][c]
                        fc +=1
                acc = float(acc) / fc
                output_[0][out_y][out_x][c] = acc

    output.data = output_
    return output_
