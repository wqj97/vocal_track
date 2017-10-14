import tensorflow as tf


class Losses(object):
    """
    get Losses
    1st: get lpca of enhanced data
    2nd: matmul lpca with generated_voice
    3ed: loss = mean((matmul_Data - labels)^2)
    """

    def __init__(self, generated_voice, lpca, labels):
        predict_data = tf.matmul(generated_voice, lpca)
        self.loss = tf.reduce_mean(tf.pow(predict_data - labels, 2)) + tf.add_n(tf.get_collection('l2_losses'))

    def get_loss(self):
        return self.loss
