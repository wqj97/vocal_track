import tensorflow as tf
import os


class Reader(object):
    def __init__(self, path, canvas_size, batch_size, threads=8, pattem='*.tfrecords', preemph=0.):
        """
        This is Reader
        :param path: datasets path
        :param pattem: file pattem (default: *.tfrecords)
        :param preemph: Pre-emphasis weight (default: 0)
        """
        self.preemph = preemph
        self.canvas_size = canvas_size
        self.batch_size = batch_size
        self.theads = threads
        self.filename_queue = tf.train.string_input_producer(tf.gfile.Glob(os.path.join(path, pattem)))

    @staticmethod
    def pre_emph(x, coeff=0.95):
        x0 = tf.reshape(x[0], [1, -1])
        diff = x[1:] - coeff * x[:-1]
        concat = tf.cast(tf.concat(0, [x0, diff]), tf.float32)
        return concat

    def read_and_decode(self, preemph=0.):
        reader = tf.TFRecordReader()

        _, serialized_example = reader.read(self.filename_queue)

        features = tf.parse_single_example(
            serialized_example,
            features={
                'wav_raw': tf.FixedLenFeature([], tf.string)
            })
        wave = tf.decode_raw(features['wav_raw'], tf.int32)
        wave.set_shape(self.canvas_size)

        wave = (2. / 65535.) * tf.cast((wave - 32767), tf.float32) + 1.  # normalization

        if preemph > 0:
            wave = self.pre_emph(wave, preemph)
        tf.train.shuffle_batch([wave], batch_size=self.batch_size, num_threads=self.theads,
                               capacity=10 * self.batch_size, min_after_dequeue=1000 + self.batch_size)
        return wave
