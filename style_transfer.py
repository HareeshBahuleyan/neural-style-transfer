import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth=True

sess = tf.Session(config=config)

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from vgg16_avg import VGG16_Avg
from evaluator import Evaluator
from keras import backend as K
from keras.models import Model
from keras import metrics
from scipy.misc import imsave
from scipy.optimize import fmin_l_bfgs_b


class StyleTransferModel(object):

    def __init__(self, config):
        self.original_image_path = config['original_image_path']
        self.style_image_path = config['style_image_path']
        self.image_size = config['image_size']
        self.lambda_coeff = config['lambda_coeff']
        self.iterations = config['iterations']
        self.content_layer_name = config['content_layer_name']
        self.style_loss_conv_blocks = config['style_loss_conv_blocks']
        self.style_wgts = config['style_wgts']
        self.start_img = config['start_img']
        self.rn_mean = np.array([123.68, 116.779, 103.939], dtype=np.float32) # For pre-processing the image, subtract the imagenet mean RGB pixels (required for VGG-16 model)

        if self.start_img == 'u':
            self.rand_img = lambda shape: np.random.uniform(-2.5, 2.5, shape)/100
        else:
            self.rand_img = lambda shape: np.random.normal(size=shape)

        # Lambda functions to preprocess the image and de-preprocess it while saving
        self.preproc = lambda x: (x - self.rn_mean)[:, :, :, ::-1]
        self.deproc = lambda x, s: np.clip(x.reshape(s)[:, :, :, ::-1] + self.rn_mean, 0, 255)

        # Load the VGG model - re-designed with average pooling instead of MAX pooling
        # Although the filter weights are kept the same
        self.model = VGG16_Avg(include_top=False, input_shape=[self.image_size, self.image_size, 3]) # 3 channel RGB
        self.outputs = {l.name: l.output for l in self.model.layers} # outputs from each conv block

    def load_image(self, img_path):
        im = Image.open(img_path)
        im = im.resize((self.image_size, self.image_size), resample=Image.ANTIALIAS)

        return im

    def gram_matrix(self, x):
        # The first dim is #channels and second dimension is flattened image_w X image_h
        features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
        # The dot product of this with its transpose shows the correlation
        # between each pair of channels
        # For 3 channel, the output is 3x3
        return K.dot(features, K.transpose(features)) / x.get_shape().num_elements()

    def get_style_loss(self, x, targ):  # MSE between the 2 gram matrices
        return K.mean(metrics.mse(self.gram_matrix(x), self.gram_matrix(targ)))

    # Deterministic BFGS solver/optimizer from scipy
    def solve_image(self, eval_obj, x, shp):
        for i in range(self.iterations):
            x, min_val, info = fmin_l_bfgs_b(eval_obj.loss, x.flatten(),
                                             fprime=eval_obj.grads, maxfun=20)
            x = np.clip(x, -127, 127)
            print('Current loss value:', min_val)
            imsave(f'images/results/res_at_iteration_{i+1}.png', self.deproc(x.copy(), shp)[0])
        return x

    def do_style_transfer(self):
        # Load source image
        im = self.load_image(self.original_image_path)
        img_arr = self.preproc(np.expand_dims(im, axis=0))
        shp = img_arr.shape

        # Load style image
        style = self.load_image(self.style_image_path)
        style = style.resize((shp[1], shp[2]), resample=Image.ANTIALIAS) # Make style and original image the same size
        style_arr = self.preproc(np.expand_dims(style, 0)[:, :, :, :3])

        self.content_layer = self.outputs[self.content_layer_name]
        self.style_layers = [self.outputs['block{}_conv2'.format(o)] for o in self.style_loss_conv_blocks]

        # Content model
        content_model = Model(self.model.input, self.content_layer)
        content_targ = K.variable(content_model.predict(img_arr))

        # Style model # Single input but multiple outputs
        style_model = Model(self.model.input, self.style_layers)
        style_targs = [K.variable(o) for o in style_model.predict(style_arr)]

        # Compute style loss between actual and generated for all layers - use the style_wgts for each layer
        self.style_loss = sum(self.get_style_loss(l1[0], l2[0]) * w
                                for l1, l2, w in zip(self.style_layers, style_targs, self.style_wgts))
        # Compute content loss
        self.content_loss = K.mean(metrics.mse(self.content_layer, content_targ))

        # Compute total loss
        self.total_loss = self.style_loss + self.lambda_coeff * self.content_loss
        grads = K.gradients(self.total_loss, self.model.input)
        transfer_fn = K.function([self.model.input], [self.total_loss] + grads)

        evaluator = Evaluator(transfer_fn, shp)
        x = self.rand_img(shp)
        x = self.solve_image(evaluator, x, shp)
