import numpy as np
from model import TFLiteModel
from quantization import compute_multiplier_for_conv2d
from PIL import Image

# PARTIES
# P0: Data owner
# P1: Model owner
# P2: Extra party

##############################################################################
######################## LOCAL PREPROCESSING #################################
##############################################################################

# load an image from 'image_path' and reshape it so it can be used as input to a
# NN. This would be called by P0 locally in order to load its data
def load_image_data(image_path, height, width):
    img = Image.open(image_path)
    img = img.resize((height, width), Image.ANTIALIAS)
    data = np.asarray(img, dtype='uint8')
    data = data.reshape((1, height, width, 3))
    return data


# Container class describing a network layer. This includes the parameters of 
# the operation, and placeholders for inputs and output.
class Layer(object):
    def __init__(self, **kwargs):
        for k in kwargs:
            setattr(self, k, kwargs[k])


# Load the model from 'model_path', and returns a list of layers with the weights
# and biases loaded in the appropriate placeholders. This is called by P1 to 
# load and parse the model.            
def load_model_data(model_path):
    model = TFLiteModel(model_path)
    layers = list()
    for op in model:
        inputs = model.get_named_inputs_for_op(op)
        output = list(model.get_outputs_for_op(op))[0]
        if op.opname in ('CONV_2D', 'DEPTHWISE_CONV_2D'):
            weights = inputs['weights'][0]
            weights_data = weights.data - weights.zero_point
            bias_data = inputs['bias'][0].data
            input_tensor = inputs['_'][0]
            shift, multiplier = compute_multiplier_for_conv2d(
                weights.scale, input_tensor.scale, output.scale
            )
            layers.append(Layer(
                weights=weights_data,
                bias=bias_data,
                input_offset=input_tensor.zero_point,
                output_offset=output.zero_point,
                quant_mult_shift=shift,
                quant_mult_multiplier=multiplier,
                name=op.opname
            ))
        elif op.opname == 'AVERAGE_POOL_2D':
            layers.append(Layer(
                stride=op.stride,
                filter_size=op.filter_size,
                name=op.opname
            ))
        elif op.opname == 'ADD':
            # the ADD operator computes
            #
            #   output_scale * (q3 - output_offset) =
            #        input1_scale * (q1 - input1_offset) +
            #        input2_scale * (q2 - input2_offset)
            #
            # I.e., a pointwise addition of quantized values, that have
            # different parameters. To support this, we need to rescale stuff,
            # which can be done in a similar way as with the convolution above.
            shift1, multiplier1 = compute_multiplier_for_conv2d(
                inputs['_'][0].scale, 1.0, output.scale
            )
            shift2, multiplier2 = compute_multiplier_for_conv2d(
                inputs['_'][1].scale, 1.0, output.scale
            )
            layers.append(Layer(
                output_offset=output.zero_point,
                input1_offset=inputs['_'][0].zero_point,
                input2_offset=inputs['_'][1].zero_point,
                quant_mult_shift1=shift1,
                quant_mult_multiplier1=multiplier1,
                quant_mult_shift2=shift2,
                quant_mult_multiplier2=multiplier2,
                name=op.opname
            ))
    return layers

# Load the model without setting any value for the placeholders. This is called
# by all the parties so that they can fill in the placeholders with the weights
# and biases which will be shared by P1 later on, and with the input that P0
# will share as well.
def load_model_description(model_path):
    model = TFLiteModel(model_path, parse_data=False)
    layers = list()
    for op in model:
        inputs = model.get_named_inputs_for_op(op)
        output = list(model.get_outputs_for_op(op))[0]
        if op.opname in ('CONV_2D', 'DEPTHWISE_CONV_2D'):
            weights = inputs['weights'][0]
            bias = inputs['bias'][0]
            padding = (weights.shape[1] // 2, weights.shape[2] // 2)
            input_tensor = inputs['_'][0]
            layers.append(Layer(
                input_shape=input_tensor.shape,
                output_shape=output.shape,
                weights_shape=weights.shape,
                bias_shape=bias.shape,
                padding=padding,
                stride=op.stride,
                name=op.opname
            ))
        elif op.opname == 'AVERAGE_POOL_2D':
            layers.append(Layer(
                input_shape=inputs['_'][0].shape,
                output_shape=output.shape,
                stride=op.stride,
                filter_size=op.filter_size,
                name=op.opname
            ))
        elif op.opname == 'ADD':
            # no options here besides the shape of the inputs and output
            layers.append(Layer(
                input1_shape=inputs['_'][0].shape,
                input2_shape=inputs['_'][1].shape,
                output_shape=output.shape,
                name=op.opname,
            ))
    return layers

# At this point the parties are ready for input distribution. P0 shares the
# parsed input and P1 shares the parsed model. The parties allocate these 
# shares appropriately.    
    
def share_image_data(image_data):
    return image_data


def share_model_data(model_data, model_description):
    # if model_data == None, we receive shares. Otherwise we send them.
    return model_data

#############################################################################
############################## OPERATIONS ################################### 
#############################################################################
    
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

    return []


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

    return []

# Average pool
#
# input_data is secret. stride and filter_size is public.
def avgpool2d(input_data, stride, filter_size):
    print 'avgpool2d'

    return []

##############################################################################
########################### MODEL EVALUATION #################################
##############################################################################

# With the inputs and model shared and the operations implemented, the parties
# proceed to the execution of the inference. This is essentially a loop that 
# goes through each layer and executes the corresponding operation.

def evaluate_model(model, input_data):
    for op in model:
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

##############################################################################
############################### OVERVIEW #####################################
##############################################################################
    
# Overall, the execution would look like this.
def run(party_id, model_path, image_path=None):

    assert party_id in (0, 1, 2), 'unexpected party_id: %s' % (party_id, )

    # Preprocess the model owners inputs.
    # model owner
    if party_id == 0:
        model_data = load_model_data(model_path)
    else:
        model_data = None

    # end of preprocessing

    # all parties load the model description
    model_description = load_model_description(model_path)

    # data owner loads their input here.
    if party_id == 1:
        input_shape = model_description[0].input_shape
        height = input_shape[1]
        width = input_shape[2]
        image_data = load_image_data(image_path, height, width)
    else:
        image_data = None

    # At this point we can start sharing the input for the evaluation. This
    # involves the model owner creating shares of the data in `model_data` and
    # the data owner sharing `image_data`.
    #
    # We cheat here and simply give all data to everyone.
    input_data = share_image_data(image_data)
    model = share_model_data(model_data, model_description)

    # runs the evaluation loop
    evaluate_model(model, input_data)



if __name__ == '__main__':
    import sys
    assert len(sys.argv) > 3
    party_id = int(sys.argv[1])
    model_path = sys.argv[2]
    image_path = sys.argv[3]
    run(party_id, model_path, image_path)
