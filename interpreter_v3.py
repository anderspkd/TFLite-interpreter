#!/usr/bin/env python

from model import TFLiteModel
import numpy as np
import json

def pprint_op(op):
    print 80*'-'
    print op
    
def label_result(result, labels):
    with open(labels) as f:
        text = f.read()
    return json.loads(text)[str(result)]

def load_image_data(image_path, input_shape, use_keras=False):
    if use_keras:
        from keras.preprocessing.image import load_img
        from keras.preprocessing.image import img_to_array
        from keras.applications.imagenet_utils import decode_predictions
        image = load_img(image_path, target_size=input_shape[1:-1])
        np_image = img_to_array(image)
        image_batch = np.expand_dims(np_image, axis=0)
        return image_batch
    img_shape = input_shape[1:-1]
    img = Image.open(image_path)
    img = img.resize(img_shape, Image.ANTIALIAS)
    data = np.asarray(img, dtype='uint8')
    data = data.reshape(input_shape)
    return data


def run(image_path, model_path, label_path, variant):
    print 'running %s on %s using variant=%s' % (
        model_path, image_path, variant)

    model = TFLiteModel(model_path)
    input_shape = model.get_input().shape
    input_data = load_image_data(image_path, input_shape, use_keras=True)

    model.set_input(input_data)

    if variant == 'reference':
        import reference_operators
        conv2d = reference_operators.conv2d
        dwconv2d = reference_operators.dwconv2d
        avgpool2d = reference_operators.avgpool2d
    elif variant == 'dequantized':
        import dequantized_operators
        conv2d = dequantized_operators.conv2d
        dwconv2d = dequantized_operators.dwconv2d
        avgpool2d = dequantized_operators.avgpool2d
        # dequantize input_data
        model.get_input().data = dequantized_operators.dequantize_tensor(model.get_input())
    else:
        raise ValueError('unknown variant:', variant)

    for op in model:
        pprint_op(op)
        inputs = model.get_named_inputs_for_op(op)
        output = list(model.get_outputs_for_op(op))[0]  # assume a single output

        opname = op.opname
        if 'CONV_2D' == opname:
            x = conv2d(op,
                   inputs['_'][0],
                   inputs['weights'][0],
                   inputs['bias'][0],
                   output)
            output.data = x
        elif 'DEPTHWISE_CONV_2D' == opname:
            x = dwconv2d(op,
                     inputs['_'][0],
                     inputs['weights'][0],
                     inputs['bias'][0],
                     output)
            output.data = x
        elif 'AVERAGE_POOL_2D' == opname:
            x = avgpool2d(op,
                          inputs['_'][0],
                          output)
            output.data = x
        elif 'RESIZE_BILINEAR' == opname:
            # TODO: fix this shitty hack
            new_shape = tuple(inputs['_'][0].data)
            output.data = inputs['bias'][0].data.flatten()
        elif 'SPACE_TO_DEPTH' == opname:
            # TODO: we assume a mobileNet v1 model is used. So we print the
            # result here.
            for i in model.get_inputs_for_op(op):
                print 'input: ', i
                print 'top 5:'
                top5 = [(x, i[x]) for x in i.data.argsort()[-5:][::-1]]
                print top5
        else:
            raise NotImplementedError('unknown operator:', opname)
        
    result = top5[0][0]
    result = label_result(result, label_path)
    print '\n%s is a "%s"' % (image_path, result)
    return model


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='TFLite interpreter')
    parser.add_argument('--variant',
                        help='''
variant. Can be "reference", "dequantized". Default is "reference"''',
                        default="reference")
    parser.add_argument('model', help='model to use')
    parser.add_argument('image', help='image to classify')
    parser.add_argument('labels', help='labels correponding to the outputs')    

    args = parser.parse_args()

    model = run(args.image, args.model, args.labels, args.variant)
