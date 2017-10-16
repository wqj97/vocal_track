from __future__ import print_function

import tensorflow as tf
from data_loader import read_and_decode
from generator import *
import numpy as np
from scipy.linalg import hankel
import matplotlib.pyplot as plt

import scipy as sp
import scipy.signal as sig

from pylab import *



def save_trainable_vars(sess,filename,**kwargs):
    """save a .npz archive in `filename`  with
    the current value of each variable in tf.trainable_variables()
    plus any keyword numpy arrays.
    """
    save={}
    for v in tf.trainable_variables():
        save[str(v.name)] = sess.run(v)
    save.update(kwargs)
    np.savez(filename,**save)

def load_trainable_vars(sess,filename):
    """load a .npz archive and assign the value of each loaded
    ndarray to the trainable variable whose name matches the
    archive key.  Any elements in the archive that do not have
    a corresponding trainable variable will be returned in a dict.
    """
    other={}
    try:
        tv=dict([ (str(v.name),v) for v in tf.trainable_variables() ])
        for k,d in np.load(filename).items():
            if k in tv:
                print('restoring ' + k)
                sess.run(tf.assign( tv[k], d) )
            else:
                other[k] = d
    except IOError:
        pass
    return other





flags = tf.app.flags

flags.DEFINE_string("e2e_dataset", "data/segan.tfrecords", "TFRecords"
                                                           " (Def: data/"
                                                           "segan.tfrecords.")
flags.DEFINE_integer("canvas_size", 2 ** 10, "Canvas size (Def: 2^14).")

flags.DEFINE_integer("batch_size", 1, "Batch size (Def: 150).")
flags.DEFINE_float("d_learning_rate", 0.01, "D learning_rate (Def: 0.0002)")


FLAGS = flags.FLAGS

def lpc2(signal, order):

    order = int(order)

    if signal.ndim > 1:
        raise ValueError("Array of rank > 1 not supported yet")
    if order > signal.size:
        raise ValueError("Input signal must have a lenght >= lpc order")

    if order > 0:
        p = order + 1
        r = np.zeros(p, signal.dtype)
        # Number of non zero values in autocorrelation one needs for p LPC
        # coefficients
        nx = np.min([p, signal.size])
        x = np.correlate(signal, signal, 'full')
        r[:nx] = x[signal.size-1:signal.size+order]
        phi = np.dot(sp.linalg.inv(sp.linalg.toeplitz(r[:-1])), -r[1:])
        return np.concatenate(([1.], phi)), None, None
    else:
        return np.ones(1, dtype = signal.dtype), None, None



def main(_):


    print(
        "Deep Linear Prediction Method: I will make predictions via my deep network instead of traditional linear method!")
    file_queue = tf.train.string_input_producer([FLAGS.e2e_dataset])
    get_wav, get_noisy = read_and_decode(file_queue, FLAGS.canvas_size)
    wavbatch, \
    noisybatch = tf.train.shuffle_batch([get_wav,
                                         get_noisy],
                                        batch_size=FLAGS.batch_size,
                                        num_threads=2,
                                        capacity=1000 + 3 * FLAGS.batch_size,
                                        min_after_dequeue=1000,
                                        name='wav_and_noisy')
    lambdaG=100
    lambdaprediction = 1
    savefile = 'DeepLPcoeff.npz'
    learning_rate = 0.0001

    deltamaxstep=50
    maxstep = 10
    test_epochs=100

    training_epochs =10#5000
    display_step = int(maxstep/10)#500
    p = 8
    p=16



    rng = np.random


    FLAGS.canvas_size=FLAGS.canvas_size+p

    # tf Graph input (only pictures)
    X = tf.placeholder(tf.float32, [FLAGS.canvas_size, p])
    #X0 = tf.placeholder(tf.float32, [FLAGS.canvas_size, p])


    Y = tf.placeholder(tf.float32, [FLAGS.canvas_size, 1])



    class param:
        def __init__(self):
            self.g_enc_depths = ''  # 名称
            self.d_num_fmaps = ''  # 尺寸
            self.bias_downconv=False
            self.deconv_type='deconv'
            self.bias_deconv=False
            #self.list = []  # 列表

    aparam = param()  # 定义结构对象



    aparam.g_enc_depths = [16 , 32]#, 32, 64]#, 64, 128, 128, 256, 256, 512, 1024]
    # Define D fmaps
    #aparam.d_num_fmaps = [16, 32, 32, 64, 64, 128, 128, 256, 256, 512, 1024]

    generator = AEGenerator(aparam)

    G= generator(X,is_ref=False, z_on=False)

    G=tf.squeeze(G)










    # Set model weights
    W = tf.placeholder(tf.float32,[p, 1])  # rng.randn(p,1)
    # b = tf.Variable(rng.randn(1), name="lastbias", dtype=tf.float32) #tf.zeros([p,1])

    # Construct a linear model
    # pred = tf.add(tf.matmul(X, W), b) # tf.multiply is wrong

    #W0 = tf.Variable(rng.randn(p, 1), name="lastweight0", dtype=tf.float32)  # rng.randn(p,1)

    #y_pred0=tf.matmul(X0,W0)


    # Prediction
    y_pred = tf.matmul(G, W) # what if i use lpca for w initialization

    # Define loss and optimizer, minimize the squared error
    cost = tf.reduce_mean(tf.pow(Y - y_pred, 2))
    #cost0 = tf.reduce_mean(tf.pow(Y - y_pred0, 2))

    #cost=lambdaG*tf.reduce_mean(tf.pow(G-X,2))+lambdaprediction*cost0




    optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)
    #optimizer0 = tf.train.RMSPropOptimizer(learning_rate0).minimize(cost0)



   # optimizertest = tf.train.RMSPropOptimizer(learning_rate).minimize(cost, var_list=[W])

    #init = tf.global_variables_initializer()


    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        sess.run(tf.global_variables_initializer())


        state = load_trainable_vars(sess, savefile)  # must load AFTER the initializer

        # must use this same Session to perform all training
        # if we start a new Session, things would replay and we'd be training with our validation set (no no)

        #done = state.get('done', [])
        log = str(state.get('log', ''))
        print(log)
        step = 0


        windowme01=sess.run(G, feed_dict={X: 0.1*np.ones([FLAGS.canvas_size, p])})
        windowme02 = sess.run(G, feed_dict={X: 0.2 * np.ones([FLAGS.canvas_size, p])})
        windowme03 = sess.run(G, feed_dict={X: 0.3 * np.ones([FLAGS.canvas_size, p])})
        windowme04 = sess.run(G, feed_dict={X: 0.4 * np.ones([FLAGS.canvas_size, p])})
        windowme05 = sess.run(G, feed_dict={X: 0.5 * np.ones([FLAGS.canvas_size, p])})
        windowme06 = sess.run(G, feed_dict={X: 0.6 * np.ones([FLAGS.canvas_size, p])})

        windowme1=sess.run(G, feed_dict={X: np.ones([FLAGS.canvas_size, p])})

        windowme2 = sess.run(G, feed_dict={X: 2*np.ones([FLAGS.canvas_size, p])})


        windowme20 = sess.run(G, feed_dict={X: 20*np.ones([FLAGS.canvas_size, p])})

        windowme40 = sess.run(G, feed_dict={X: 40* np.ones([FLAGS.canvas_size, p])})

        plt.figure(0)

        plt.plot(windowme01[0, :],label='0.1')
        plt.plot(windowme02[0, :],label='0.2')
        plt.plot(windowme03[0, :],label='0.3')
        plt.plot(windowme04[0, :],label='0.4')
        plt.plot(windowme05[0, :],label='0.5')
        plt.plot(windowme06[0, :],label='0.6')

        plt.plot(windowme1[0,:],label='1' )

        plt.plot(windowme2[0, :],label='2')

        plt.plot(windowme20[0, :],label='20')

        plt.plot(windowme40[0, :],label='40')
        plt.legend()

        plt.show()
        #return
        #init.run()

        try:
            while not coord.should_stop():


             #+++++++++++++++++++++++



               maxstep = maxstep + deltamaxstep


               for i in range(test_epochs):
                 inputdata, noisybatch0 = sess.run([wavbatch, noisybatch])
                 inputdata = np.squeeze(inputdata)
                 xt=inputdata
                 num_sample = len(xt)

                 def nextpow2(x):
                     return np.ceil(np.log2(x))

                 zpf = 3
                 Nfft = int(2 ** nextpow2(num_sample * zpf))

                 Org_XW = fft(xt, Nfft)

                 test_X = np.asarray(hankel(np.append(np.zeros(p), inputdata), np.zeros((p, 1))))

                 # print(train_X.shape)
                 ##print(inputdata)
                 test_Y = np.asarray([np.append(inputdata, np.zeros((p, 1)))])

                 a, _, _ = lpc2(inputdata, p)
                 b = -a[1:]
                 lpca = np.asarray([b[::-1]]).T

                 # print('linear prediction coeff=',lpca)
                 test_Y = test_Y.T
                 test_G =sess.run(G, feed_dict={X: test_X})

                 invX = np.linalg.pinv(test_G)

                 myW = np.dot(invX, test_Y)
                 # print(Org_LP_coef)

                 my_est = np.dot(test_G, myW)

                 my_est = my_est[0:-p]


                 plt.figure(1)
                 plt.subplot(221)
                 plt.plot(test_Y[0:-p], label='Original data')
                 plt.plot(test_Y[0:-p]-my_est, 'r',label='my residue line')
                 # plt.plot(sess.run(y_pred0, feed_dict={X0: train_X, Y: train_Y}), label='LP line')
                 plt.plot(test_Y[0:-p]-np.matmul(test_X[0:-p], lpca), 'b--',label='LP residue line')
                 plt.legend()
                 print("LPC error is ", np.mean(np.square(test_Y[0:-p]-np.matmul(test_X[0:-p], lpca))))
                 print("my error is", np.mean(np.square(test_Y[0:-p]-my_est)))


                 plt.subplot(222)
                 plt.plot(lpca, 'r--', label='LP coef')
                 plt.plot(myW, 'b', label='deep LP coef')
                 plt.legend()
                 #plt.show()
                 #
                 Fs = 16000
                 myDLPcoef = np.append(1, -myW[::-1])

                 w0, Org_h0 = sig.freqz(1, myDLPcoef, Nfft, whole=True)
                 Org_F0 = Fs * w0 / (2 * np.pi)
                 Org_LP_coef=a
                 w, Org_h = sig.freqz(1, Org_LP_coef, Nfft, whole=True)
                 Org_F = Fs * w / (2 * np.pi)

                 Org_mag = abs(Org_XW)

                 Org_mag = 20 * np.log10(Org_mag)

                 f = np.asarray(range(Nfft)).astype(np.float32) * Fs / Nfft

                 plt.subplot(212)
                 plt.plot(f, Org_mag, 'k-', label='signal')

                 plt.plot(Org_F, 20 * np.log10(abs(Org_h)), 'b--', label='lpc')
                 plt.plot(Org_F0, 20 * np.log10(abs(Org_h0)), label='mylpc')

                 plt.xlim((0, Fs / 2))
                 #plt.ylim((-40, 50))
                 plt.legend()

                 filtercoeff = np.append(0, -Org_LP_coef[1:])
                 # print(filtercoeff)
                 est_x = sig.lfilter(filtercoeff, 1, xt)  # Estimated signal
                 e = xt - est_x

                 # plt.figure(6)
                 # plt.subplot(211)
                 # plt.plot(xt, 'r--', label='original')
                 # plt.plot(est_x, 'b', label='est')
                 # plt.plot(my_est, label='my est')
                 # plt.legend()
                 # plt.subplot(212)
                 # plt.plot(e, 'r--', label='residue')
                 # plt.plot(xt - np.squeeze(my_est), label='my residue')
                 # plt.legend()

                 # plt.figure(7)
                 #
                 # plt.plot(-Org_LP_coef[1:], label='LP')
                 #
                 # plt.plot(myW[::-1], label='my Deep LP')
                 # plt.legend()

                 plt.show()
                 plt.close('all')
                 break
                 #print('W=',sess.run(W))


        except Exception as e:
            print(e)
            coord.request_stop()
        except IOError as e:

            coord.should_stop()
        else:
            pass

        finally:
            pass

        coord.request_stop()

        coord.join(threads)




        # coord.request_stop()

        # coord.join(threads)


if __name__ == '__main__':
    tf.app.run()
