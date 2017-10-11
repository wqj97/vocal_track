from __future__ import print_function

import argparse
import os
import timeit

import numpy as np
import scipy.io.wavfile as wavfile
import tensorflow as tf


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def slice_signal(signal, window_size, stride=0.5):
    """ Return windows of the given signal by sweeping in stride fractions
        of window
    """
    assert signal.ndim == 1, signal.ndim
    n_samples = signal.shape[0]
    offset = int(window_size * stride)
    slices = []
    for beg_i, end_i in zip(range(0, n_samples, offset),
                            range(window_size, n_samples + offset,
                                  offset)):
        if end_i - beg_i < window_size:
            break
        slice_ = signal[beg_i:end_i]
        if slice_.shape[0] == window_size:
            slices.append(slice_)
    return np.array(slices, dtype=np.int32)


def read_and_slice(filename, wav_canvas_size, stride=0.5):
    fm, wav_data = wavfile.read(filename)
    if fm != 16000:
        raise ValueError('Sampling rate is expected to be 16kHz!')
    signals = slice_signal(wav_data, wav_canvas_size, stride)
    return signals


def encoder_proc(wav_filename, out_file, wav_canvas_size):
    """ Read and slice the wav and noisy files and write to TFRecords.
        out_file: TFRecordWriter.
    """
    wav_signals = read_and_slice(wav_filename, wav_canvas_size)
    for wav in wav_signals:
        wav_raw = wav.tostring()
        example = tf.train.Example(features=tf.train.Features(feature={
            'wav_raw': _bytes_feature(wav_raw)
        }))
        out_file.write(example.SerializeToString())


def main(opts):
    if not os.path.exists(opts.save_path):
        # make save path if it does not exist
        os.makedirs(opts.save_path)
    # set up the output filepath
    out_filepath = os.path.join(opts.save_path, opts.out_file)
    if os.path.splitext(out_filepath)[1] != '.tfrecords':
        # if wrong extension or no extension appended, put .tfrecords
        out_filepath += '.tfrecords'
    else:
        out_filename, ext = os.path.splitext(out_filepath)
        out_filepath = out_filename + ext
    # check if out_file exists and if force flag is set
    if os.path.exists(out_filepath) and not opts.force_gen:
        raise ValueError('ERROR: {} already exists. Set force flag (--force-gen) to '
                         'overwrite. Skipping this speaker.'.format(out_filepath))
    elif os.path.exists(out_filepath) and opts.force_gen:
        print('Will overwrite previously existing tfrecords')
        os.unlink(out_filepath)
    beg_enc_t = timeit.default_timer()
    out_file = tf.python_io.TFRecordWriter(out_filepath)
    wav_files = tf.gfile.Glob(os.path.join(opts.wav_datasets_path, '*.wav'))
    for wav_file in wav_files:
        encoder_proc(wav_file, out_file, opts.canvas_size)

    out_file.close()
    end_enc_t = timeit.default_timer() - beg_enc_t

    print('Total processing and writing time: {} s'.format(end_enc_t))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert the set of txt and '
                                                 'wavs to TFRecords')
    parser.add_argument('--cfg', type=str, default='cfg/e2e_maker.cfg',
                        help='File containing the description of datasets '
                             'to extract the info to make the TFRecords.')
    parser.add_argument('--save_path', type=str, default='data/',
                        help='Path to save the dataset')
    parser.add_argument('--out_file', type=str, default='segan.tfrecords',
                        help='Output filename')
    parser.add_argument('--force-gen', dest='force_gen', action='store_true',
                        help='Flag to force overwriting existing dataset.')
    parser.add_argument('--wav_datasets_path', type=str, default='data/clean_trainset_wav_16k/',
                        help='Output filename')
    parser.add_argument('--canvas_size', type=int, default=2 ** 10,
                        help='Output filename')
    parser.set_defaults(force_gen=True)
    opts = parser.parse_args()
    main(opts)
