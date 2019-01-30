import numpy as np
import quantization

uint8 = np.uint8
int32 = np.int32
int64 = np.int64

# This performs a rounded multiplication of v (a 32 bit integer), and a
# multiplier (should be < 1). The result is clamped to an int32.
def rounded_mult(v, multiplier):
    x = int64(v) * multiplier
    if x < quantization.INT32_MIN or x > quantization.INT32_MAX:
        return quantization.INT32_MAX
    return x


# 2D conv on quantized inputs according to the reference implementation.
def conv2d(options, inputs, weights, bias, output):

    n_channels_out, weights_h, weights_w, n_channels_in = weights.shape
    _, inputs_h, inputs_w, n_channels_in_ = inputs.shape
    _, output_h, output_w, n_channels_out_ = output.shape

    assert n_channels_in_ == n_channels_in
    assert n_channels_out_ == n_channels_out

    # dilated convolution is not supported.
    assert options.dilation_factor == (1, 1)

    # quantization params
    inputs_S, inputs_Z = inputs.scale, inputs.zero_point
    weights_S, weights_Z = weights.scale, weights.zero_point
    bias_S, bias_Z = bias.scale, bias.zero_point
    output_S, output_Z = output.scale, output.zero_point

    print 'quantization params:'
    print 'input: %s, %s' % (inputs_S, inputs_Z)
    print 'weights: %s, %s' % (weights_S, weights_Z)
    print 'bias: %s, %s' % (bias_S, bias_Z)
    print 'output: %s, %s' % (output_S, output_Z)

    # other options
    stride_h, stride_w = options.stride
    # a bit crude, but this provides the maximum padding we need, since we
    # assume "SAME" padding type.
    padding_h, padding_w = (weights_h // 2, weights_w // 2)

    multiplier = (inputs_S * weights_S) / output_S

    output.data = np.zeros(output.shape, dtype='uint8')

    for out_y in range(output_h):
        for out_x in range(output_w):
            for out_c in range(n_channels_out):
                in_x_origin = (out_x * stride_w) - padding_w
                in_y_origin = (out_y * stride_h) - padding_h
                acc = int32(0)
                for filter_y in range(weights_h):
                    for filter_x in range(weights_w):
                        for in_c in range(n_channels_in):
                            in_x = in_x_origin + filter_x
                            in_y = in_y_origin + filter_y
                            if (0 <= in_x < inputs_w) and \
                               (0 <= in_y < inputs_h):
                                iv = int32(inputs[0][in_y][in_x][in_c])
                                wv = int32(weights[out_c][filter_y][filter_x][in_c])
                                acc += (iv - inputs_Z) * (wv - weights_Z)
                acc += bias[out_c]
                acc = int32(rounded_mult(acc, multiplier))
                acc += output_Z
                acc = 255 if acc > 255 else acc
                acc = 0   if acc < 0   else acc
                output[0][out_y][out_x][out_c] = uint8(acc)
    return output.data


def dwconv2d(options, inputs, weights, bias, output):
    depth_multiplier = options.depth_multiplier

    # from here everything is essentially the same as before
    _, weights_h, weights_w, output_depth = weights.shape
    _, inputs_h, inputs_w, n_channels_in = inputs.shape
    _, output_h, output_w, output_depth_ = output.shape

    assert output_depth_ == output_depth
    assert options.dilation_factor == (1, 1)

    inputs_S, inputs_Z = inputs.scale, inputs.zero_point
    weights_S, weights_Z = weights.scale, weights.zero_point
    bias_S, bias_Z = bias.scale, bias.zero_point
    output_S, output_Z = output.scale, output.zero_point

    multiplier = (inputs_S * weights_S) / output_S

    stride_h, stride_w = options.stride
    padding_h, padding_w = (weights_h // 2, weights_w // 2)

    print 'input shape: ', inputs.shape
    print 'output shape:', output.shape
    print 'weights shape:', weights.shape

    print 'quantization params:'
    print 'input: %s, %s' % (inputs_S, inputs_Z)
    print 'weights: %s, %s' % (weights_S, weights_Z)
    print 'bias: %s, %s' % (bias_S, bias_Z)
    print 'output: %s, %s' % (output_S, output_Z)

    output.data = np.zeros(output.shape, dtype='uint8')

    for out_y in range(output_h):
        for out_x in range(output_w):
            for in_c in range(n_channels_in):
                for m in range(depth_multiplier):
                    oc = m + in_c * depth_multiplier
                    in_x_origin = (out_x * stride_w) - padding_w
                    in_y_origin = (out_y * stride_h) - padding_h
                    acc = 0
                    for filter_y in range(weights_h):
                        for filter_x in range(weights_w):
                            in_x = in_x_origin + filter_x
                            in_y = in_y_origin + filter_y
                            if (0 <= in_x < inputs_w) and \
                               (0 <= in_y < inputs_h):
                                iv = int32(inputs[0][in_y][in_x][in_c])
                                wv = int32(weights[0][filter_y][filter_x][oc])
                                acc += (iv - inputs_Z) * (wv - weights_Z)
                    acc += bias[oc]
                    acc = int32(rounded_mult(acc, multiplier))
                    acc += output_Z
                    acc = 0   if acc < 0   else acc
                    acc = 255 if acc > 255 else acc
                    output[0][out_y][out_x][oc] = uint8(acc)

    return output.data


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

    output_ = np.zeros(output.shape, dtype='uint8')

    # quantization parameters are not used at all here??
    for out_y in range(output_h):
        for out_x in range(output_w):
            for c in range(n_channels_in):
                in_x_origin = (out_x * stride_w) - padding_w
                in_y_origin = (out_y * stride_h) - padding_h
                fxs = max(0, -in_x_origin)
                fxe = min(filter_w, input_w - in_x_origin)
                fys = max(0, -in_y_origin)
                fye = min(filter_h, input_h - in_y_origin)
                acc = 0
                fc = 0
                for filter_y in range(fys, fye):
                    for filter_x in range(fxs, fxe):
                        in_x = in_x_origin + filter_x
                        in_y = in_y_origin + filter_y
                        acc += input[0][in_y][in_x][c]
                        fc +=1
                acc = (acc + fc / 2) / fc
                acc = min(255, max(0, acc))
                output_[0][out_y][out_x][c] = uint8(acc)

    output.data = output_
    return output_
