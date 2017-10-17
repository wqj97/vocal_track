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
        import pylab as plt
        train_collect, label_collect = self.sess.run([self.wav_data, self.label_data])
        generate_data = self.generator.eval(feed_dict={
            self.wav_data: train_collect
        })
        lpca_data = self.generate.get_lpc_a(generate_data)
        predict_data = np.matmul(generate_data, lpca_data)
        print "generate data:"
        print generate_data
        print "predict data:"
        print predict_data
        plt.subplot(311)
        plt.title('data figure')
        plt.plot(generate_data[0], 'r', label='generate data')
        plt.plot(label_collect[0], 'b', label='label')
        plt.subplot(312)
        plt.title('predict figure')
        plt.plot(label_collect[0], 'b', label='label')
        plt.plot(predict_data[0], 'r', label='predict')
        plt.subplot(313)
        plt.title('loss')
        plt.plot(pow(predict_data[0] - label_collect[0], 2), label='(predic - data)^2')
        plt.show()
