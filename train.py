import tensorflow as tf
import inference
import losses
import reader


class Train(object):
    """
    Define train script
    """

    def __init__(self, sess, learning_rate, data_sets_path, batch_size, canvas_size, window_size, threads, max_steps,
                 save_path, optimizer, kwidth, stride, is_train, beta1, summary_step, saver_step):
        self.bach_size = batch_size
        self.saver_step = saver_step
        self.summary_step = summary_step
        self.canvas_size = canvas_size
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
        global_step = tf.get_variable(
            'global_step', [],
            initializer=tf.constant_initializer(0), trainable=False)
        wav_data = tf.placeholder(tf.float32,
                                  [self.bach_size, self.canvas_size - self.window_size, self.window_size])

        generate = inference.Inference(wav_data, self.kwidth, self.stride, self.is_train)
        generator = generate.build_seae_model()
        lpca = tf.placeholder(tf.float32, [self.window_size, None], name="lpca")
        label_data = tf.placeholder(tf.float32, [self.bach_size, self.canvas_size - self.window_size, 1],
                                    name="label")

        loss = losses.Losses(generator, lpca, label_data).get_loss()
        tf.summary.scalar('losses', loss)

        train_op = self.get_optimizer(self.optimizer).minimize(loss, var_list=tf.trainable_variables(),
                                                               global_step=global_step)

        tf.global_variables_initializer().run()
        tf.train.start_queue_runners()

        saver = tf.train.Saver(var_list=tf.trainable_variables())
        summary_writer = tf.summary.FileWriter(self.save_path, graph=self.sess.graph)
        summary_op = tf.summary.merge_all()
        for i in xrange(self.max_step):

            train_collect, label_collect = self.sess.run([self.wav_data, self.label_data])
            generate_data = generator.eval(feed_dict={
                wav_data: train_collect
            })
            lpca_data = generate.get_lpc_a(generate_data)
            train_op.run(feed_dict={
                wav_data: train_collect,
                lpca: lpca_data,
                label_data: label_collect
            })

            if i % self.summary_step == 0 or i + 1 == self.max_step:
                summary_data = summary_op.eval(feed_dict={
                    wav_data: train_collect,
                    lpca: lpca_data,
                    label_data: label_collect
                })

                summary_writer.add_summary(summary_data, global_step.eval())
            if i % self.saver_step == 0 or i + 1 == self.max_step:
                saver.save(self.sess, self.save_path + 'model', global_step)

    def train_on_single_gpu(self):
        global_step = tf.get_variable(
            'global_step', [],
            initializer=tf.constant_initializer(0), trainable=False)
        wav_data = tf.placeholder(tf.float32,
                                  [self.bach_size, self.canvas_size - self.window_size - 1, self.window_size])

        generate = inference.Inference(wav_data, self.kwidth, self.stride, self.is_train)
        generator = generate.build_ae_model()

        lpca = tf.placeholder(tf.float32, [self.window_size, None], name="lpca")
        label_data = tf.placeholder(tf.float32, [self.bach_size, self.canvas_size - self.window_size - 1, 1],
                                    name="label")

        loss = losses.Losses(generator, lpca, label_data).get_loss()
        tf.summary.scalar('losses', loss)

        train_op = self.get_optimizer(self.optimizer).minimize(loss, var_list=tf.trainable_variables(),
                                                               global_step=global_step)

        tf.global_variables_initializer().run()
        tf.train.start_queue_runners()

        saver = tf.train.Saver(var_list=tf.trainable_variables())
        summary_writer = tf.summary.FileWriter(self.save_path, graph=self.sess.graph)
        summary_op = tf.summary.merge_all()
        for i in xrange(self.max_step):

            train_collect, label_collect = self.sess.run([self.wav_data, self.label_data])
            generate_data = generator.eval(feed_dict={
                wav_data: train_collect
            })
            lpca_data = generate.get_lpc_a(generate_data)
            train_op.run(feed_dict={
                wav_data: train_collect,
                lpca: lpca_data,
                label_data: label_collect
            })

            if i % self.summary_step == 0 or i + 1 == self.max_step:
                summary_data = summary_op.eval(feed_dict={
                    wav_data: train_collect,
                    lpca: lpca_data,
                    label_data: label_collect
                })
                summary_writer.add_summary(summary_data, global_step.eval())
            if i % self.saver_step == 0 or i + 1 == self.max_step:
                saver.save(self.sess, self.save_path + 'model', global_step)
