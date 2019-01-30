import numpy as np
from model import TFLiteModel, Tensor
import math
import quantization
import sys
from PIL import Image

mult_by_quant_mult = quantization.quantized_multiplier_mult

int32 = np.int32
int64 = np.int64
uint8 = np.uint8


def rounded_mult(v, multiplier):
    x = int64(v) * multiplier
    if x < quantization.INT32_MIN or x > quantization.INT32_MAX:
        return quantization.INT32_MAX
    return x


def dequantize_tensor(tensor):
    S, Z = tensor.scale, tensor.zero_point
    return (np.array(tensor.data, dtype='float32') - Z) * S


def conv2d_dequant(options, inputs, weights, bias, output):
    # assume input has already been dequantized.

    clamp_output = 'relu6' in output.name.lower()

    print 'clamping output: ', clamp_output

    weights = dequantize_tensor(weights)
    bias = dequantize_tensor(bias)

    # from here everything is essentially the same as before
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

    # reference_ops.h version. Assume num_batch == 0
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
                if clamp_output:
                    acc = 0.0 if acc < 0 else acc
                    acc = 6.0 if acc > 6 else acc
                output.data[0][out_y][out_x][out_c] = acc

    return output.data


def dwconv2d_dequant(options, inputs, weights, bias, output):
    # assume input has already been dequantized.

    clamp_output = 'relu6' in output.name.lower()

    weights = dequantize_tensor(weights)
    bias = dequantize_tensor(bias)

    depth_multiplier = options.depth_multiplier

    # from here everything is essentially the same as before
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
                    if clamp_output:
                        acc = 0.0 if acc < 0 else acc
                        acc = 6.0 if acc > 6 else acc
                    output[0][out_y][out_x][oc] = acc

    return output.data


def conv2d_reference(options, inputs, weights, bias, output):

    n_channels_out, weights_h, weights_w, n_channels_in = weights.shape
    _, inputs_h, inputs_w, n_channels_in_ = inputs.shape
    _, output_h, output_w, n_channels_out_ = output.shape

    assert n_channels_in_ == n_channels_in
    assert n_channels_out_ == n_channels_out

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


def dwconv2d_reference(options, inputs, weights, bias, output):
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
                    acc = int32(0)
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
                    # acc = int32(round(multiplier * acc))
                    acc = 0   if acc < 0   else acc
                    acc = 255 if acc > 255 else acc
                    output[0][out_y][out_x][oc] = uint8(acc)

    return output.data


def conv2d(options, inputs, weights, bias, output):

    # assume number of batches == 1
    n_channels_out, weights_h, weights_w, n_channels_in = weights.shape
    _, inputs_h, inputs_w, n_channels_in_ = inputs.shape

    assert n_channels_in_ == n_channels_in

    # quantization params
    inputs_S, inputs_Z = inputs.scale, inputs.zero_point
    weights_S, weights_Z = weights.scale, weights.zero_point
    bias_S, bias_Z = bias.scale, bias.zero_point
    output_S, output_Z = output.scale, output.zero_point

    n, qm = quantization.compute_multiplier_for_conv2d(weights_S, inputs_S, output_S)

    # other options
    stride_h, stride_w = options.stride
    # a bit crude, but this provides the maximum padding we need, since we
    # assume "SAME" padding type.
    padding_h, padding_w = (weights_h // 2, weights_w // 2)

    output_shape = (int(math.ceil(float(inputs_h)/stride_h)),
                    int(math.ceil(float(inputs_w)/stride_w)),
                    n_channels_out)

    output_h, output_w, _ = output_shape

    print 'input shape: ', inputs.shape
    print 'weights shape: ', weights.shape
    print 'output shape: ', output_shape

    sys.stdout.write('input prep ... ')
    sys.stdout.flush()

    # input prep. This is essentially the Conv function from reference_ops.h
    # except we don't perform any computation. It is possible to pre-process
    # this guy also by telling people which entries correspond to padding and
    # which doesn't. (Note that this doesn't leak information.)

    inputs_per_channel = list()
    for c in range(n_channels_in):
        Z = list()
        for i in range(output_h):
            for j in range(output_w):
                x = i * stride_h
                y = j * stride_w
                z = list()
                for a in range(weights_h):
                    for b in range(weights_w):
                        xx = x + a - padding_h
                        yy = y + b - padding_w
                        if (0 <= xx < inputs_h) and (0 <= yy < inputs_w):
                            z.append(int64(inputs[0][xx][yy][c]))
                        else:
                            z.append(int64(0))
                Z.append(z)
        inputs_per_channel.append(Z)
    inputs_per_channel = np.array(inputs_per_channel, dtype='int64')

    print 'done'

    sys.stdout.write('flattening weights ... ')
    sys.stdout.flush()
    # flatten and reshape weights. Again, this can be preprocessed.
    flat_weights = list()
    for c in range(n_channels_out):
        ww = list()
        for c_ in range(n_channels_in):
            w = list()
            for a in range(weights_h):
                for b in range(weights_w):
                    w.append(weights[c][a][b][c_])
            ww.append(np.array(w, dtype='int64'))
        flat_weights.append(ww)

    print 'done'

    sys.stdout.write('computing conv2d ... ')
    sys.stdout.flush()

    # This step is very cheap
    max_print = 10
    print_counter = 0
    output_list = list()
    for c in range(n_channels_out):
        output_for_c = np.zeros(output_shape[:-1])
        for c_ in range(n_channels_in):
            W_flat = flat_weights[c][c_]
            I_c = inputs_per_channel[c_]
            # if print_counter <= max_print:
            #     print 'I_c:\n----------------------------------'
            #     print I_c
            #     print '----------------------------------------'
            #     print 'W_flat:\n-------------------------------'
            #     print W_flat
            print_counter += 1
            r = (I_c - inputs_Z).dot(W_flat - weights_Z)
            r = r.reshape(output_shape[:-1])
            output_for_c += r
        output_list.append(output_for_c + bias[c])

    print 'done'

    sys.stdout.write('scaling results ... ')
    sys.stdout.flush()

    multiplier = (weights_S * inputs_S) / output_S
    # def f(v):
    #     x = rounded_mult(v, multiplier)
    #     x += output_Z
    #     x = 255 if x > 255 else x
    #     return 0 if x < 0 else x

    # f = np.vectorize(f)

    # output_final = np.array(output_list)
    # output_final = np.expand_dims(f(output_final), axis=0)
    # print output_final.shape
    # return output_final

    # this step is very expensive (probably the clamping and whatnot)
    output_final = np.zeros((output.shape))
    for c in range(n_channels_out):
        for i in range(output_shape[0]):
            for j in range(output_shape[1]):
                v = output_list[c][i][j]
                v = rounded_mult(v, multiplier)
                # v = mult_by_quant_mult(v, qm, n)
                v += output_Z
                v = 255 if v > 255 else v
                v = 0 if v < 0 else v
                output_final[0][i][j][c] = v

    print 'done'

    return output_final


def dwconv2d(options, inputs, weights, bias, output):
        # assume number of batches == 1
    depth_mult, weights_h, weights_w, n_channels_in = weights.shape
    _, inputs_h, inputs_w, n_channels_in_ = inputs.shape

    assert n_channels_in_ == n_channels_in

    # quantization params
    inputs_S, inputs_Z = inputs.scale, inputs.zero_point
    weights_S, weights_Z = weights.scale, weights.zero_point
    bias_S, bias_Z = bias.scale, bias.zero_point
    output_S, output_Z = output.scale, output.zero_point

    n, qm = quantization.compute_multiplier_for_conv2d(weights_S, inputs_S, output_S)

    multiplier = (weights_S * inputs_S) / output_S

    # other options
    stride_h, stride_w = options.stride
    # a bit crude, but this provides the maximum padding we need, since we
    # assume "SAME" padding type.
    padding_h, padding_w = (weights_h // 2, weights_w // 2)

    output_shape = output.shape
    _, output_h, output_w, n_channels_out = output_shape

    print 'input shape: ', inputs.shape
    print 'weights shape: ', weights.shape
    print 'output shape: ', output.shape

    sys.stdout.write('prepping input ... ')
    sys.stdout.flush()

    # same as with conv2d. Convert each layer into a matrix where each row
    # correspond to a window.
    inputs_per_channel = list()
    for c in range(n_channels_in):
        Z = list()
        for i in range(output_h):
            for j in range(output_w):
                x = i * stride_h
                y = j * stride_w
                z = list()
                for a in range(weights_h):
                    for b in range(weights_w):
                        xx = x + a - padding_h
                        yy = y + b - padding_w
                        if (0 <= xx < inputs_h) and (0 <= yy < inputs_w):
                            z.append(int64(inputs[0][xx][yy][c] - inputs_Z))
                        else:
                            z.append(int64(0))
                Z.append(z)
        inputs_per_channel.append(Z)
    inputs_per_channel = np.array(inputs_per_channel, dtype='int64')
    print inputs_per_channel.shape

    print 'done'

    # Since we don't need to sum accross channels in this case, we could easily
    # merge the computation with the above step. We can them separate here,
    # since that more precisely reflects the MPC implementation.

    sys.stdout.write('flattening weights ... ')
    sys.stdout.flush()

    flat_weights = list()
    for d in range(depth_mult):
        ww = list()
        for c_ in range(n_channels_in):
            w = list()
            for a in range(weights_h):
                for b in range(weights_w):
                    w.append(weights[d][a][b][c_] - weights_Z)
            ww.append(np.array(w, dtype='int64'))
        flat_weights.append(ww)

    print 'done'

    sys.stdout.write('computing dwconv2d ... ')
    sys.stdout.flush()

    max_print = 10
    print_counter = 0

    output_list = list()
    for d in range(depth_mult):
        for c in range(n_channels_in):
            W_flat = flat_weights[d][c_]
            I_c = inputs_per_channel[c_]
            if print_counter <= max_print:
                print 'I_c:\n----------------------------------'
                print I_c
                print '----------------------------------------'
                print 'W_flat:\n-------------------------------'
                print W_flat
            print_counter += 1
            r = I_c.dot(W_flat).reshape(output_shape[1:-1])
            r += bias[c]
            output_list.append(r)

    print 'done'

    sys.stdout.write('scaling outputs ... ')
    sys.stdout.flush()

    output_final = np.zeros(output_shape)
    for c in range(n_channels_out):
        for i in range(output_shape[1]):
            for j in range(output_shape[2]):
                v = output_list[c][i][j]
                # v = mult_by_quant_mult(v, qm, n)
                v = rounded_mult(v, multiplier)
                v += output_Z
                v = 255 if v > 255 else v
                v = 0 if v < 0 else v
                output_final[0][i][j][c] = v

    print 'done'
    return output_final



def add(options, input1, input2, output):
    # TODO: Doing something wrong here....

    print 'input1', input1.shape
    print 'input2', input2.shape

    in1_S, in1_Z = input1.scale, input1.zero_point
    in2_S, in2_Z = input2.scale, input2.zero_point
    out_S, out_Z = output.scale, output.zero_point

    n1, qm1 = quantization.compute_multiplier(in1_S/out_S)
    n2, qm2 = quantization.compute_multiplier(in2_S/out_S)
    no, qmo = quantization.compute_multiplier(out_S)


    input11 = np.array(input1.data, dtype='int32') - in1_Z
    input22 = np.array(input2.data, dtype='int32') - in2_Z

    # shifts input
#    input11 *= two_23
#    input22 *= two_23

    # ugh. mult_by_quant_mult is not vectorized.
    output_ = np.zeros(output.shape)
    for c in range(output.shape[3]):
        for i in range(output.shape[1]):
            for j in range(output.shape[2]):
                in1m = mult_by_quant_mult(input11[0][i][j][c], qm1, n1)
                in2m = mult_by_quant_mult(input22[0][i][j][c], qm2, n2)

                v = in1m + in2m
#                out = mult_by_quant_mult(v, qmo, no)
                out = v
                out += out_Z
                out = 255 if out > 255 else out
                out = 0 if out < 0 else out
                output_[0][i][j][c] = out

#    print input1.data
#    print input2.data
#    print output_
    return output_


def avgpool2d_dequant(options, input, output):
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
                acc = 0.0
                fc = 0
                for filter_y in range(fys, fye):
                    for filter_x in range(fxs, fxe):
                        in_x = in_x_origin + filter_x
                        in_y = in_y_origin + filter_y
                        acc += input[0][in_y][in_x][c]
                        fc +=1
                # acc = (float(acc) + fc / 2) / fc
                acc = float(acc) / fc
                # acc = min(6.0, max(0.0, acc))
                output_[0][out_y][out_x][c] = acc

    output.data = output_
    return output_

    # for c in range(n_channels_out):
    #     acc = np.int64(0)
    #     for i in range(input_h):
    #         for j in range(input_w):
    #             acc += int64(input[0][i][j][c] - input.zero_point)
    #     avg = (float(acc) / n)
    #     output_[0][0][0][c] = int(round(avg))

    # return output_


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

    # for c in range(n_channels_out):
    #     acc = np.int64(0)
    #     for i in range(input_h):
    #         for j in range(input_w):
    #             acc += int64(input[0][i][j][c] - input.zero_point)
    #     avg = (float(acc) / n)
    #     output_[0][0][0][c] = int(round(avg))

    # return output_


def load_image(image_path, input_shape):
    from keras.preprocessing.image import load_img
    from keras.preprocessing.image import img_to_array
    from keras.applications.imagenet_utils import decode_predictions
    image = load_img(image_path, target_size=input_shape[1:-1])
    np_image = img_to_array(image)
    image_batch = np.expand_dims(np_image, axis=0)
    return image_batch

    # img_shape = input_shape[1:-1]
    # img = Image.open(image_path)
    # img = img.resize(img_shape, Image.ANTIALIAS)
    # data = np.asarray(img, dtype='uint8')
    # data = data.reshape(input_shape)
    # return data


def run(model_path, input_image):
    model = TFLiteModel(model_path, parse_data=True, use_flat_tensors=False)
    input_shape = model.get_input().shape
    model.set_input(load_image(input_image, input_shape))

    np.set_printoptions(threshold=np.nan)

    for op in model:
        print '-------------------------------------'
        print op
        inputs = model.get_named_inputs_for_op(op)
        output = list(model.get_outputs_for_op(op))[0]  # assume single output

        # print 'inputs:'
        # print inputs
        # print 'output:'
        # print output

        # raw_input('---')

        # if inputs['_'][0].name == 'input':
        #     inputs['_'][0].data = dequantize_tensor(inputs['_'][0])

        opname = op.opname
        if 'CONV_2D' == opname:
            # output.data = np.random.randint(0,255,size=output.shape)
            # output.data = np.zeros(output.shape)
            x = conv2d_reference(op,
                       inputs['_'][0],
                       inputs['weights'][0],
                       inputs['bias'][0],
                       output)
            print x.flatten()
            output.data = x
        elif 'DEPTHWISE_CONV_2D' == opname:
            # output.data = np.random.randint(0,255,size=output.shape)
            # output.data = np.zeros(output.shape)
            x = dwconv2d_reference(op,
                         inputs['_'][0],
                         inputs['weights'][0],
                         inputs['bias'][0],
                         output)
            print x.flatten()
            output.data = x
        elif 'ADD' == opname:
#             output.data = np.random.randint(0,255,size=output.shape)

            input1 = inputs['_'][0]
            input2 = inputs['_'][1]
#            input1.data = np.ones(input1.shape, dtype = 'uint8') * 100
#            input2.data = np.ones(input2.shape, dtype = 'uint8') * 145

            x = add(op,
                    inputs['_'][0],
                    inputs['_'][1],
                    output)
            output.data = x
        elif 'AVERAGE_POOL_2D' == opname:
            x = avgpool2d(op,
                          inputs['_'][0],
                          output)
            print x.flatten()
            output.data = x
        elif 'RESIZE_BILINEAR' == opname:
            new_shape = tuple(inputs['_'][0].data)
            print new_shape
            # TODO: "bias" contains input for this layer, for some reason
            output.data = inputs['bias'][0].data.flatten()
        elif 'SPACE_TO_DEPTH' == opname:
            for i in model.get_inputs_for_op(op):
                print 'input:', i
                print 'top 5:'
                print [(x, i[x]) for x in i.data.argsort()[-5:][::-1]]
        else:
            raise NotImplementedError('Unknown opname:', opname)

    # print np.argmax(model.get_output().data)
    return model

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print 'Usage: %s [tflite file] [image]' % (sys.argv[0],)
        exit(0)
    model = run(sys.argv[1], sys.argv[2])
