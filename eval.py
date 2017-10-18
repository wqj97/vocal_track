import tensorflow as tf
import inference
import reader
import numpy as np


class Eval(object):
    def __init__(self, sess, data_sets_path, canvas_size, batch_size, window_size, threads, kwidth, stride,
                 save_path):
        self.sess = sess
        self.wav_data, self.label_data = reader.Reader(data_sets_path, canvas_size, batch_size, window_size,
                                                       threads).get_batch()
        self.wav_data_evaled = tf.placeholder(tf.float32,
                                              [batch_size, canvas_size - window_size - 1, window_size])
        self.generate = inference.Inference(self.wav_data, kwidth, stride, False)
        self.generator = self.generate.build_ae_model()

        tf.global_variables_initializer().run()
        tf.train.start_queue_runners()

        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint(save_path))

    def eval(self):
        import matplotlib.pyplot as plt
        train_collect, label_collect = self.sess.run([self.wav_data, self.label_data])
        generate_data = self.generator.eval(feed_dict={
            self.wav_data: train_collect
        })
        lpca_data = self.generate.get_lpc_a(generate_data)
        predict_data = np.matmul(generate_data, lpca_data)

        plt.subplot(311)
        plt.title('data figure')
        plt.plot(generate_data[:, 0], 'r', label='generate data')
        plt.plot(train_collect[0][:, 0], 'b', label='raw data')
        plt.legend()
        plt.subplot(312)
        plt.title('predict figure')
        plt.plot(predict_data[0], 'r', label='predict')
        plt.plot(np.reshape(label_collect, [-1]), 'b', label='label')
        plt.legend()
        plt.subplot(313)
        plt.title('loss')
        plt.plot(pow(predict_data[0] - np.reshape(label_collect, [-1]), 2), label='(predic - data)^2')
        plt.legend()
        plt.show()
