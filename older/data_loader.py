from __future__ import print_function

from ops import *


def pre_emph(x, coeff=0.95):
    x0 = tf.reshape(x[0], [1, ])
    diff = x[1:] - coeff * x[:-1]
    concat = tf.concat(0, [x0, diff])
    return concat


def de_emph(y, coeff=0.95):
    if coeff <= 0:
        return y
    x = np.zeros(y.shape[0], dtype=np.float32)
    x[0] = y[0]
    for n in range(1, y.shape[0], 1):
        x[n] = coeff * x[n - 1] + y[n]
    return x


def read_and_decode(filename_queue, canvas_size, preemph=0.):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            'wav_raw': tf.FixedLenFeature([], tf.string)
        })
    wave = tf.decode_raw(features['wav_raw'], tf.int32)
    wave.set_shape(canvas_size)
    wave = (2. / 65535.) * tf.cast((wave - 32767), tf.float32) + 1.

    if preemph > 0:
        wave = tf.cast(pre_emph(wave, preemph), tf.float32)
    return wave


if __name__ == '__main__':
    sess = tf.InteractiveSession()
    wave, noisy = read_and_decode(tf.train.string_input_producer(['data/segan.tfrecords']), 2 ** 14)
    noisybatch = tf.train.shuffle_batch([wave,
                                         noisy],
                                        batch_size=2,
                                        num_threads=2,
                                        capacity=1000 + 3 * 2,
                                        min_after_dequeue=1000,
                                        name='wav_and_noisy')
    tf.train.start_queue_runners(sess)
    print(noisybatch)
