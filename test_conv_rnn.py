import utils
from audio_processing import AudioProcessor
from convRNN import ConvRNN, Conv2dNN
import tensorflow as tf
import numpy as np
from time import time
from matplotlib import pyplot as plt
from conv.input_data import AudioProcessor as A_proc

def qwetest_conv_rnn():
    # We want to see all the logging messages for this tutorial.
    tf.logging.set_verbosity(tf.logging.INFO)
    with tf.Session() as sess:
        model_settings = utils.get_default_model_settings()
        model_settings['wanted_words'] = 'yes,no'

        data_dir = model_settings['data_dir']
        wanted_words = model_settings['wanted_words']
        validation_percentage = model_settings['validation_percentage']
        testing_percentage = model_settings['testing_percentage']
        model_settings['strip_window_size_ms'] = 30.0
        model_settings['strip_window_stride_ms'] = 30.0
        model_settings['window_size_ms'] = 30.0
        model_settings['window_stride_ms'] = 10.0


        proc = AudioProcessor(model_settings,
                                      data_dir,
                                      wanted_words=wanted_words.split(','),
                                      validation_percentage=validation_percentage,
                                      testing_percentage=testing_percentage)
        model_settings = proc._model_settings
        conv_rnn = Conv2dNN("conv2d", model_settings)
        conv_rnn.train(proc)

qwetest_conv_rnn()