import tensorflow as tf
import inference
import losses
import reader
import datetime


class Train(object):
    """
    Define train script
    """

    def __init__(self, sess, learning_rate, data_sets_path, batch_size, canvas_size, window_size, threads, max_steps,
                 save_path, optimizer, kwidth, stride, is_train, beta1, summary_step, saver_step):
        self.saver_step = saver_step
        self.summary_step = summary_step
        self.beta1 = beta1
        self.is_train = is_train
        self.stride = stride
        self.kwidth = kwidth
        self.sess = sess
        self.optimizer = optimizer
        self.save_path = save_path
        self.max_step = max_steps
        self.learning_rate = learning_rate
        self.window_size = window_size
        self.wav_data, self.label_data = reader.Reader(data_sets_path, canvas_size, batch_size, window_size,
                                                       threads).get_batch()

    def get_optimizer(self, optimizer_name):
        if optimizer_name == 'adam':
            return tf.train.AdamOptimizer(self.learning_rate, self.beta1)
        elif optimizer_name == 'rmsprop':
            return tf.train.RMSPropOptimizer(self.learning_rate)

    def train_on_gpu(self):
        return tf.Operation

    def train_on_single_gpu(self):
        global_step = tf.get_variable(
            'global_step', [],
            initializer=tf.constant_initializer(0), trainable=False)

        generate = inference.Inference(self.wav_data, self.kwidth, self.stride, self.is_train)

        lpca = tf.placeholder(tf.float32, [None, self.window_size])
        input_data = tf.placeholder(tf.float32, [None, self.window_size])
        label_data = tf.placeholder(tf.float32, [None, self.window_size])

        loss = losses.Losses(input_data, lpca, label_data).get_loss()
        tf.summary.scalar('losses', loss)

        train_op = self.get_optimizer(self.optimizer).minimize(loss)

        saver = tf.train.Saver()
        summary_writer = tf.summary.FileWriter(self.save_path)
        summary_op = tf.summary.merge_all()
        for i in xrange(self.max_step):

            generate_data, label = self.sess.run([generate, self.label_data])
            lpca_data = generate.get_lpc_a(generate_data, self.window_size)
            train_op.eval(feed_dict={
                lpca: lpca_data,
                input_data: generate_data,
                label_data: label
            })

            global_step += 1
            if i % self.summary_step or i + 1 == self.max_step:
                summary_data = summary_op.eval(feed_dict={
                    lpca: lpca_data,
                    input_data: generate_data,
                    label_data: label
                })
                summary_writer.add_summary(summary_data, global_step)
            if i % self.saver_step or i + 1 == self.max_step:
                saver.save(self.sess, self.save_path + 'model.ckpt', global_step)
