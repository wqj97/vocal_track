import tensorflow as tf
import numpy as np
import scipy as sp
from scipy.linalg import toeplitz


class Inference(object):
    def __init__(self, input_tensor, kwidth=5, stride=2, is_train=True):
        """
        Build model
        :param Tensor input_tensor: input Tensor
        :param int kwidth: width of down and up convolution (default: 5)
        :param int stride: stride of down and up convolution
        :param bool is_train: whether train or not (default: True)
        """
        self.stride = stride
        self.kwidth = kwidth
        self.input_tensor = input_tensor
        self.is_train = is_train

        # this list store each down_conv output layer shapes , it makes us more convenient to use up_conv function
        self.layer_shape = []

        # this list represent the kernel of each conv and deconv layer
        self.num_kernel = [16, 32, 256, 512, 1024]

    @staticmethod
    def variable_with_weight_decay(name, shape, wd=None):
        """
        get an variable with L2loss
        :param str name: variable name
        :param list shape: variable shape
        :param float wd: weight decay (default: None)
        :return: Tensor
        """

        var = tf.get_variable(name, shape, initializer=tf.contrib.layers.xavier_initializer())
        tf.summary.histogram(name, var)
        if wd:
            l2_loss = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
            tf.add_to_collection('l2_losses', l2_loss)
        return var

    @staticmethod
    def get_lpc_a(signals):
        """
        get lpc a
        :param ndarray signals: contains signals which need to be calculate
        :return: ndarray
        """
        output = []
        order = signals.shape[-1]
        for signal in signals:
            p = order + 1
            r = np.zeros(p, signal.dtype)
            nx = np.min([p, signal.size])
            x = np.correlate(signal, signal, 'full')
            r[:nx] = x[signal.size - 1:signal.size + order]
            phi = np.dot(sp.linalg.inv(toeplitz(r[:-1])), -r[1:])
            lpc = np.concatenate(([1.], phi))
            lpca = -lpc[1:][::-1].T
            output.append(lpca)
        return np.array(output).astype(np.float32).T

    def down_conv(self, input_tensor, kwidth, num_kernel, stride, wd=None, padding='SAME', name='down_conv'):
        """
        2D down convolution
        :param Tensor input_tensor: Tensor which will be convolution
        :param int kwidth: width of convolution
        :param int num_kernel: number of convolution out kernels
        :param int stride: number of convolution stride
        :param float wd: weight decay of L2loss (default: None)
        :param str padding: convolution padding type (default: SAME)
        :param str name: name of this operation (default: down_conv)
        :return: Tensor
        """
        input_shape = input_tensor.shape.as_list()  # get the input shape
        w = self.variable_with_weight_decay('filter_weight', [1, kwidth, input_shape[-1], num_kernel], wd)
        b = tf.get_variable('filter_biases', [num_kernel], initializer=tf.constant_initializer([0.1]))
        conv = tf.nn.conv2d(input_tensor, w, strides=[1, 1, stride, 1], padding=padding)
        self.layer_shape.append(conv.shape.as_list())  # this line makes our more convenient to use for up_conv function
        return tf.nn.relu(conv + b, name=name)

    def up_conv(self, input_tensor, kwidth, num_kernel, stride, output_shape, wd=None, padding='SAME', name='up_conv'):
        """
        2D up convolution
        :param Tensor input_tensor: Tensor which will be deconvolution
        :param int kwidth: width of deconvolution
        :param int num_kernel: number of deconvolution out kernels
        :param int stride: number of deconvolution stride
        :param list output_shape: output shape of up convolution
        :param float wd: weight decay of L2loss (default: None)
        :param str padding: convolution padding type (default: SAME)
        :param str name: name of this operation (default: up_conv)
        :return: Tensor
        """
        input_shape = input_tensor.shape.as_list()  # get the input shape
        w = self.variable_with_weight_decay('filter_weight', [1, kwidth, num_kernel, input_shape[-1]], wd)
        b = tf.get_variable('filter_biases', [num_kernel], initializer=tf.constant_initializer([0.1]))
        deconv = tf.nn.conv2d_transpose(input_tensor, w, strides=[1, 1, stride, 1], output_shape=output_shape,
                                        padding=padding)
        return tf.nn.relu(deconv + b, name=name)

    def build_ae_model(self):
        """
        Build an AutoEncoder model
        :return: Tensor
        """
        with tf.variable_scope('encoder'):
            # First we need to reshape input tensor into [-1, kwidth]
            layer = tf.reshape(self.input_tensor, [-1, self.input_tensor.shape.as_list()[-1]])
            # Second we need to expand the of layer dimension into [batch_size, height, width, channels] to fit 1d
            # convolution
            layer = tf.expand_dims(layer, 1)
            layer = tf.expand_dims(layer, 3)
            self.layer_shape.append(layer.shape.as_list())
            print "down convolution: "
            for index, num_kernel in enumerate(self.num_kernel):
                with tf.variable_scope("down_conv_{}".format(index)) as scope:
                    preview_shape = layer.shape
                    layer = self.down_conv(layer, self.kwidth, num_kernel, self.stride, name=scope.name)
                    print "{} -> {}".format(preview_shape, layer.shape)
        with tf.variable_scope('decoder'):
            print "up convolution"
            for index, layer_shape in enumerate(self.layer_shape[::-1][1:]):
                with tf.variable_scope("up_conv_{}".format(index)) as scope:
                    preview_shape = layer.shape
                    layer = self.up_conv(layer, self.kwidth, layer_shape[-1], self.stride,
                                         output_shape=layer_shape, name=scope.name)
                    print "{} -> {}".format(preview_shape, layer.shape)
        return tf.squeeze(layer)


if __name__ == '__main__':
    sess = tf.InteractiveSession()
    inference = Inference(tf.constant(0.1, dtype=tf.float32, shape=[2, 10, 64]))
    data = inference.build_ae_model()
    tf.global_variables_initializer().run()
    data = data.eval()
    voice_value = tf.constant(0.1, dtype=tf.float32, shape=[20, 64]).eval()
    voice_value = inference.get_lpc_a(voice_value, 64)
    print np.shape(voice_value)
    print np.shape(data)
    print tf.matmul(voice_value, data).eval()
