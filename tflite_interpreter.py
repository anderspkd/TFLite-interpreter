from sys import argv

if len(argv) < 4:
    print 'Usage: {} [model] [image] [labels]'.format(argv[0])
    exit(0)

model = argv[1]
image = argv[2]
labels = argv[3]

import numpy as np
import tensorflow as tf
from PIL import Image
import json

def prep_image(image_loc, data_shape):
    img_shape = data_shape[1:-1]  # shape is (batch, height, width, channels)
    img = Image.open(image_loc)
    img = img.resize(img_shape, Image.ANTIALIAS)
    data = np.asarray(img, dtype="uint8")
    return data.reshape(data_shape)

def setup_interpreter(model_path):
    interpreter = tf.contrib.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    return interpreter, input_details, output_details

def label_result(result):
    with open(labels) as f:
        text = f.read()
    return json.loads(text)[str(result)]

if __name__ == '__main__':
    interpreter, input_details, output_details = setup_interpreter(model)
    input_shape = input_details[0]['shape']
    input_idx = input_details[0]['index']
    data = prep_image(image, input_shape)
    interpreter.set_tensor(input_idx, data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    result = np.argmax(output_data)
    result = label_result(result)
    print '%s is a "%s"' % (image, result)
