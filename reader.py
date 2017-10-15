import tensorflow as tf
import os


class Reader(object):
    def __init__(self, path, canvas_size, batch_size, window_size=64, threads=8, pattem='*.tfrecords'):
        """
        This is Reader
        :param str path: datasets path
        :param int canvas_size: canvas_size for each tfrecord line
        :param int batch_size: number of batch size for each train step
        :param int window_size: slice each canvas into parts
        :param int threads: number of read threads
        :param str pattem: file pattem (default: *.tfrecords)
        """
        self.kwidth = window_size
        self.canvas_size = canvas_size
        self.batch_size = batch_size
        self.threads = threads
        self.filename_queue = tf.train.string_input_producer(tf.gfile.Glob(os.path.join(path, pattem)))

    def get_batch(self):
        """
        get train batch
        reade and decode data.
        :returns:
        :var Tensor train_collect: train batch
        :var Tensor label_collect: label batch
        """
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

        train_collect = []
        label_colloect = []
        for i in xrange(wave.shape[0] - self.kwidth - 1):
            train_collect.append(wave[i:i + self.kwidth])
            label_colloect.append([wave[i + self.kwidth]])

        train_collect, label_colloect = tf.train.shuffle_batch([train_collect, label_colloect],
                                                               batch_size=self.batch_size, num_threads=self.threads,
                                                               capacity=1000 + 10 * self.batch_size,
                                                               min_after_dequeue=1000)
        return train_collect, label_colloect


if __name__ == '__main__':
    """
    test the batch output
    train_collect: [batch_size, 2 ** 10 - kwidth - 1, kwidth] -> [1, 959, 64]
    label_collect: [batch_size, 2 ** 10 - kwidth - 1, 1] -> [1, 959, 1]
    """
    import numpy as np

    sess = tf.InteractiveSession()
    reader = Reader('data', 2 ** 10, 1)
    batch = reader.get_batch()
    tf.train.start_queue_runners()

    print np.shape(batch[0])  # print shape of train collect
    print np.shape(batch[1])  # print shape of label collect
    print sess.run([batch[0][0][1], batch[1][0][0]])  # test whether we get correct train data and it's label
