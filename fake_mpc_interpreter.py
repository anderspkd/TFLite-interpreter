import numpy as np
from model import Tensor, TFLiteModel
from quantization import compute_multiplier_for_conv2d
from PIL import Image

dummy_output = np.zeros((1,1))

# this is called by the guy with the picture. Returns a tensor of the image that
# can be shared.
#
# `height` and `width` is part of some public knowledge of the model (we need to
# know the size of our input). The final data is a tensor of shape
#
#   (1, height, width, 3)
#
# i.e., we assume that we're dealing with a RGB image. Strictly speaking, the
# number of channels is also part of the model. As is the number of batches (the
# 1 in the beginning).
def preprocess_data_owner(image_path, height, width):
    img = Image.open(image_path)
    img = img.resize((height, width), Image.ANTIALIAS)
    data = np.asarray(img, dtype='uint8')
    data = data.reshape((1, height, width, 3))
    return data


# called by the model owner
def preprocess_model_owner(model_path):
    model = TFLiteModel(model_path)

    class Operator(object):
        def __init__(self, **kwargs):
            for k in kwargs:
                setattr(self, k, kwargs[k])

    ops = list()
    for op in model:
        if op.opname in ('CONV_2D', 'DEPTHWISE_CONV_2D'):
            inputs = model.get_named_inputs_for_op(op)
            output = list(model.get_outputs_for_op(op))[0]
            # get data for the weights and add the offset.
            weights = inputs['weights'][0]
            weights_offset = weights.zero_point
            weights_data = weights.data - weights_offset
            # bias
            bias_data = inputs['bias'][0].data
            # calculate padding
            padding = (weights.shape[1] // 2, weights.shape[2] // 2)
            # calculate quantized multiplier
            input_tensor = inputs['_'][0]
            shift, multiplier = compute_multiplier_for_conv2d(
                weights.scale, input_tensor.scale, output.scale
            )
            operator = Operator(
                weights=weights_data,
                bias=bias_data,
                padding=padding,
                stride=op.stride,
                input_offset=input_tensor.zero_point,
                # weights_offset=weights.zero_point,
                output_offset=output.zero_point,
                output_shape=output.shape,
                quant_mult_shift=shift,
                quant_mult_multiplier=multiplier,
                name=op.opname
            )
            ops.append(operator)

        elif op.opname == 'AVERAGE_POOL_2D':
            operator = Operator(
                stride=op.stride,
                filter_size=op.filter_size
            )

    return ops


# Convolution.
#
# Secret shared parameters:
#  input_data: tensor with input data
#  input_offset: zero point for the input
#  output_offset: zero point for the output
#  weights_data: model data
#  bias_data: bias (also part of the model)
#  shift: used in truncation
#  multiplier: integer
#
# Public parameters
#  output_shape: shape of the output tensor
#  stride: stride used for computing the convolution
#  padding: size of the padding
def conv2d(input_data, input_offset, output_offset, output_shape,
           weights_data, bias_data, stride, padding, shift, multiplier):
    print 'conv2d:\n\t%s\n\t%s\n\t%s' % (
        input_data.shape, output_shape, weights_data.shape)

    return dummy_output


# Depthwise Convolution.
#
# Secret shared parameters:
#  input_data: tensor with input data
#  input_offset: zero point for the input
#  output_offset: zero point for the output
#  weights_data: model data
#  bias_data: bias (also part of the model)
#  shift: used in truncation
#  multiplier: integer
#
# Public parameters
#  output_shape: shape of the output tensor
#  stride: stride used for computing the convolution
#  padding: size of the padding
def dwconv2d(input_data, input_offset, output_offset, output_shape,
             weights_data, bias_data, stride, padding, shift,
             multiplier):
    print 'dwconv2d:\n\t%s\n\t%s\n\t%s' % (
        input_data.shape, output_shape, weights_data.shape)

    return dummy_output

# Average pool
#
# input_data is secret. stride and filter_size is public.
def avgpool2d(input_data, stride, filter_size):
    print 'avgpool2d'

    return dummy_output


# Intuition: Each of the two parties call their respective pre-process
# function. The inputs to this function is the output of each of these, after
# they've been shared.
def run(image_data, model_data):
    input_data = image_data
    for op in model_data:
        if op.name == 'CONV_2D':
            input_data = conv2d(
                input_data,
                op.input_offset,
                op.output_offset,
                op.output_shape,
                op.weights,
                op.bias,
                op.stride,
                op.padding,
                op.quant_mult_shift,
                op.quant_mult_multiplier
            )
        elif op.name == 'DEPTHWISE_CONV_2D':
            input_data = dwconv2d(
                input_data,
                op.input_offset,
                op.output_offset,
                op.output_shape,
                op.weights,
                op.bias,
                op.stride,
                op.padding,
                op.quant_mult_shift,
                op.quant_mult_multiplier
            )
        elif op.name == 'AVERAGE_POOL_2D':
            input_data = avgpool2d(
                input_data
            )
        else:
            raise NotImplementedError('Unknown operator:', op.name)

    # this is revealed towards the data owner.
    return input_data


if __name__ == '__main__':
    import sys
    assert len(sys.argv) > 2
    model_path = sys.argv[1]
    image_path = sys.argv[2]

    model_data = preprocess_model_owner(model_path)
    # the 128 is specific to the smaller v1 models
    image_data = preprocess_data_owner(image_path, 128, 128)

    run(image_data, model_data)
