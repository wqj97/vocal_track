import tensorflow as tf


class Inference(object):
    def __init__(self, input_tensor, is_train=True):
        """
        Build model
        :param Tensor input_tensor: input Tensor
        :param bool is_train: whether train or not (default: True)
        """
        self.input = input_tensor
        self.is_train = is_train

        # this list store each down_conv output layer shapes , it makes us more convenient to use up_conv function
        self.layer_shape = [input_tensor.shape.as_list()]

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
        if wd:
            l2_loss = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
            tf.add_to_collection('l2_losses', l2_loss)
        return var

    def down_conv(self, input_tensor, kwidth, num_kernel, stride, wd=None, padding='SAME', name='down_conv'):
        """
        1D down convolution
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
        w = self.variable_with_weight_decay('filter_weight', [kwidth, input_shape[-1], num_kernel], wd)
        b = tf.get_variable('filter_biases', [num_kernel], initializer=tf.constant_initializer([0.1]))
        conv = tf.nn.conv1d(input_tensor, w, stride=stride, padding=padding)
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
        deconv = tf.nn.conv2d_transpose(input_tensor, w, strides=[1, stride, 1, 1], output_shape=output_shape,
                                        padding=padding)
        return tf.nn.relu(deconv + b, name=name)
