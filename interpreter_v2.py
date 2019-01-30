import numpy as np
from model import TFLiteModel, Tensor
import math
import quantization
import sys
from PIL import Image
import json

if len(sys.argv) < 3:
    print 'Usage: %s [tflite file] [image]' % (sys.argv[0],)
    exit(0)

mult_by_quant_mult = quantization.quantized_multiplier_mult

def label_result(result, labels):
    with open(labels) as f:
        text = f.read()
    return json.loads(text)[str(result)]

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
    
    # Formulas from TF website
    out_height = np.ceil(float(inputs_h) / float(stride_h))
    out_width  = np.ceil(float(inputs_w) / float(stride_w))
    
    if (inputs_h % stride_h == 0):
        pad_along_height = max(weights_h - stride_h, 0)
    else:
        pad_along_height = max(weights_h - (inputs_h % stride_h), 0)
    if (inputs_w % stride_w == 0):
        pad_along_width = max(weights_w - stride_w, 0)
    else:
        pad_along_width = max(weights_w - (inputs_w % stride_w), 0)
        
    pad_top = pad_along_height // 2
    pad_bottom = pad_along_height - pad_top
    pad_left = pad_along_width // 2
    pad_right = pad_along_width - pad_left
    
    padding_h, padding_w = (weights_h // 2, weights_w // 2)

    output_shape = (int(math.ceil(float(inputs_h)/stride_h)),
                    int(math.ceil(float(inputs_w)/stride_w)),
                    n_channels_out)

    output_h, output_w, _ = output_shape

    print 'input shape: ', inputs.shape
    print 'weights shape: ', weights.shape
    print 'output shape: ', output_shape, (out_height, out_width)
    print 'total padding (h,w): ', pad_along_height, pad_along_width
    print 'padding t,b,l,r: ', pad_top, pad_bottom, pad_left, pad_right

    sys.stdout.write('input prep ... ')
    sys.stdout.flush()

    # input prep. This is essentially the Conv function from reference_ops.h
    # except we don't perform any computation. It is possible to pre-process
    # this guy also by telling people which entries correspond to padding and
    # which doesn't. (Note that this doesn't leak information.)

    inputs_per_channel = list()
    for c in range(n_channels_in):
        # Z holds a list of lists
        # Each list is the flat version of a window
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
                            z.append(inputs[0][xx][yy][c] - inputs_Z)
                        else:
                            z.append(0)
                Z.append(z)
        inputs_per_channel.append(Z)
    inputs_per_channel = np.array(inputs_per_channel, dtype='int32')

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
                    w.append(weights[c][a][b][c_] - weights_Z)
            ww.append(np.array(w, dtype='int32'))
        flat_weights.append(ww)

    print 'done'

    sys.stdout.write('computing conv2d ... ')
    sys.stdout.flush()

    # This step is very cheap
    output_list = list()
    for c in range(n_channels_out):
        output_for_c = np.zeros(output_shape[:-1])
        for c_ in range(n_channels_in):
            W_flat = flat_weights[c][c_]
            I_c = inputs_per_channel[c_]
            r = I_c.dot(W_flat)
            r = r.reshape(output_shape[:-1])
            output_for_c += r
        output_list.append(output_for_c + bias[c])

    print 'done'
#    print output_list
    sys.stdout.write('scaling results ... ')
    sys.stdout.flush()

    # this step is very expensive (probably the clamping and whatnot)
    output_final = np.zeros((output.shape))
    for c in range(n_channels_out):
        for i in range(output_shape[0]):
            for j in range(output_shape[1]):
                v = output_list[c][i][j]
                v = mult_by_quant_mult(v, qm, n)
                v += output_Z
                v = 255 if v > 255 else v
                v = 0 if v < 0 else v
                output_final[0][i][j][c] = v

    print 'done'
    print np.array(output_list)
    print np.moveaxis(output_final, -1, 1)
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
                            z.append(inputs[0][xx][yy][c] - inputs_Z)
                        else:
                            z.append(0)
                Z.append(z)
        inputs_per_channel.append(Z)
    inputs_per_channel = np.array(inputs_per_channel, dtype='int32')

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
            ww.append(np.array(w, dtype='int32'))
        flat_weights.append(ww)

    print 'done'

    sys.stdout.write('computing dwconv2d ... ')
    sys.stdout.flush()

    output_list = list()
    for d in range(depth_mult):
        for c in range(n_channels_in):
            W_flat = flat_weights[d][c_]
            I_c = inputs_per_channel[c_]
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
                v = mult_by_quant_mult(v, qm, n)
                v += output_Z
                v = 255 if v > 255 else v
                v = 0 if v < 0 else v
                output_final[0][i][j][c] = v

    print 'done'
#    print output_final
    return output_final

two_23 = pow(2, 23)
int32 = np.int32

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


def avgpool2d(options, input, output):
    _, input_h, input_w, n_channels_in = input.shape
    _, output_h, output_w, n_channels_out = output.shape

    # We make assumptions about this particular layer
    # Window size = Input size, so there is no stride and the output is 1x1
    
    n = input_h * input_w
    print 'input.shape: ', input.shape
    print 'output.shape: ', output.shape
    print 'divisor: ', n

    assert output_h == output_w == 1
    assert n_channels_in == n_channels_out

    output_ = np.zeros(output.shape)

    for c in range(n_channels_out):
        acc = int32(0)
        for i in range(input_h):
            for j in range(input_w):
                acc += input[0][i][j][c]# - input.zero_point
#        avg = (acc + n//2) // n
        avg = (float(acc) / n)
        out = int(round(avg))
        out = 255 if out > 255 else out
        out = 0 if out < 0 else out
        output_[0][0][0][c] = out
    print output_[0][0][0]
    return output_


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


def run(model_path, input_image, labels):
    model = TFLiteModel(model_path, parse_data=True, use_flat_tensors=False)
    input_shape = model.get_input().shape
    model.set_input(load_image(input_image, input_shape))

    for op in model:
        print "\n" + 80*"-" + "\n", op
        inputs = model.get_named_inputs_for_op(op)
        output = list(model.get_outputs_for_op(op))[0]  # assume single output

        opname = op.opname
        if 'CONV_2D' == opname:
            output.data = np.random.randint(0,255,size=output.shape)
            # output.data = np.zeros(output.shape)
            x = conv2d(op,
                       inputs['_'][0],
                       inputs['weights'][0],
                       inputs['bias'][0],
                       output)
            output.data = x
#            print output.data
#            return output.data
        elif 'DEPTHWISE_CONV_2D' == opname:
            output.data = np.random.randint(0,255,size=output.shape)
            # output.data = np.zeros(output.shape)
            x = dwconv2d(op,
                         inputs['_'][0],
                         inputs['weights'][0],
                         inputs['bias'][0],
                         output)
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
            
            avg = np.abs(np.mean(input1.scale*(input1.data - input1.zero_point) \
                                 + input2.scale*(input2.data - input2.zero_point) \
            - output.scale*(output.data - output.zero_point)))
            print "Average error: ", avg
            
#            return model
        elif 'AVERAGE_POOL_2D' == opname:
            x = avgpool2d(op,
                          inputs['_'][0],
                          output)
            output.data = x
        elif 'RESIZE_BILINEAR' == opname:
            new_shape = tuple(inputs['_'][0].data)
            # TODO: "bias" contains input for this layer, for some reason
            output.data = inputs['bias'][0].data.flatten()
        elif 'SPACE_TO_DEPTH' == opname:
            for i in model.get_inputs_for_op(op):
                print i
                print np.argmax(i.data)
                model.get_output().data = i.data
                
        else:
            raise NotImplementedError('Unknown opname:', opname)

    result = np.argmax(model.get_output().data) 
    result = label_result(result, labels)
    print '%s is a "%s"' % (input_image, result)
    return model



if __name__ == '__main__':
    assert len(sys.argv) == 4
    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        model = run(sys.argv[1], sys.argv[2], sys.argv[3])
