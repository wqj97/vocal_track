import tensorflow as tf
import train

tf.flags.DEFINE_string('buckets', './data', 'Path where date set are')
tf.flags.DEFINE_string('checkpointDir', './saves', 'Path where model and summary saved')
tf.flags.DEFINE_string('optimizer', 'adam', 'use which optimizer, support adam, rmsprop')
tf.flags.DEFINE_float('learning_rate', 1e-3, 'Learning rate')
tf.flags.DEFINE_float('Adam_beta_1', 0.5, 'Learning rate')
tf.flags.DEFINE_integer('max_steps', 10000, "How many steps we need to execute")
tf.flags.DEFINE_integer('batch_size', 32, "batch size")
tf.flags.DEFINE_integer('threads', 8, "execute threads")
tf.flags.DEFINE_integer('canvas_size', 64, "Voice fragment length")
tf.flags.DEFINE_integer('window_size', 2 ** 10, "datasets window size")
tf.flags.DEFINE_integer('summary_frequency', 20, "After how many steps to summary data")
tf.flags.DEFINE_integer('save_frequency', 1000, "After how many steps to save model")
tf.flags.DEFINE_integer('kwidth', 5, "width of convolution")
tf.flags.DEFINE_integer('stride', 2, "stride of convolution")
tf.flags.DEFINE_boolean('log_device_placement', False, "Log option used device")
tf.flags.DEFINE_boolean('is_train', True, "Log option used device")
FLAGS = tf.flags.FLAGS

sess = tf.InteractiveSession(config=tf.ConfigProto(
    allow_soft_placement=True,
    log_device_placement=FLAGS.log_device_placement))
train_script = train.Train(sess=sess,
                           learning_rate=FLAGS.learning_rate,
                           data_sets_path=FLAGS.buckets,
                           batch_size=FLAGS.batch_size,
                           canvas_size=FLAGS.canvas_size,
                           window_size=FLAGS.window_size,
                           threads=FLAGS.threads,
                           max_steps=FLAGS.max_steps,
                           save_path=FLAGS.checkpointDir,
                           optimizer=FLAGS.optimizer,
                           kwidth=FLAGS.kwidth,
                           stride=FLAGS.stride,
                           is_train=FLAGS.is_train,
                           beta1=FLAGS.beta1,
                           summary_step=FLAGS.summary_frequency,
                           saver_step=FLAGS.save_frequency
                           )

train_script.train_on_single_gpu()
