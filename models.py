# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Model definitions for simple speech recognition.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import tensorflow as tf


def prepare_model_settings(label_count, sample_rate, clip_duration_ms,
                           window_size_ms, window_stride_ms,
                           dct_coefficient_count):
  """Calculates common settings needed for all models.

  Args:
    label_count: How many classes are to be recognized.
    sample_rate: Number of audio samples per second.
    clip_duration_ms: Length of each audio clip to be analyzed.
    window_size_ms: Duration of frequency analysis window.
    window_stride_ms: How far to move in time between frequency windows.
    dct_coefficient_count: Number of frequency bins to use for analysis.

  Returns:
    Dictionary containing common settings.
  """
  desired_samples = int(sample_rate * clip_duration_ms / 1000)
  window_size_samples = int(sample_rate * window_size_ms / 1000)
  window_stride_samples = int(sample_rate * window_stride_ms / 1000)
  length_minus_window = (desired_samples - window_size_samples)
  if length_minus_window < 0:
    spectrogram_length = 0
  else:
    spectrogram_length = 1 + int(length_minus_window / window_stride_samples)
  fingerprint_size = dct_coefficient_count * spectrogram_length
  return {
      'desired_samples': desired_samples,
      'window_size_samples': window_size_samples,
      'window_stride_samples': window_stride_samples,
      'spectrogram_length': spectrogram_length,
      'dct_coefficient_count': dct_coefficient_count,
      'fingerprint_size': fingerprint_size,
      'label_count': label_count,
      'sample_rate': sample_rate,
  }


def create_model(fingerprint_input, model_settings, model_architecture,
                 is_training, runtime_settings=None):
  """Builds a model of the requested architecture compatible with the settings.

  There are many possible ways of deriving predictions from a spectrogram
  input, so this function provides an abstract interface for creating different
  kinds of models in a black-box way. You need to pass in a TensorFlow node as
  the 'fingerprint' input, and this should output a batch of 1D features that
  describe the audio. Typically this will be derived from a spectrogram that's
  been run through an MFCC, but in theory it can be any feature vector of the
  size specified in model_settings['fingerprint_size'].

  The function will build the graph it needs in the current TensorFlow graph,
  and return the tensorflow output that will contain the 'logits' input to the
  softmax prediction process. If training flag is on, it will also return a
  placeholder node that can be used to control the dropout amount.

  See the implementations below for the possible model architectures that can be
  requested.

  Args:
    fingerprint_input: TensorFlow node that will output audio feature vectors.
    model_settings: Dictionary of information about the model.
    model_architecture: String specifying which kind of model to create.
    is_training: Whether the model is going to be used for training.
    runtime_settings: Dictionary of information about the runtime.

  Returns:
    TensorFlow node outputting logits results, and optionally a dropout
    placeholder.

  Raises:
    Exception: If the architecture type isn't recognized.
  """
  if model_architecture == 'crnn_gru':
    return create_GRU_crnn_model(fingerprint_input, model_settings, is_training)
  elif model_architecture ==  'crnn_lstm':
    return create_LSTM_crnn_model(fingerprint_input, model_settings, is_training)
  elif model_architecture == 'rcnn_gru':
    return create_GRU_rcnn_model(fingerprint_input, model_settings, is_training)
  elif model_architecture == 'rcnn_lstm':
    return create_LSTM_rcnn_model(fingerprint_input, model_settings, is_training)
  
  else:
    raise Exception('model_architecture argument "' + model_architecture +
                    '" not recognized, should be one of "single_fc", "conv",' +
                    ' "low_latency_conv, or "low_latency_svdf"')


def load_variables_from_checkpoint(sess, start_checkpoint):
  """Utility function to centralize checkpoint restoration.

  Args:
    sess: TensorFlow session.
    start_checkpoint: Path to saved checkpoint on disk.
  """
  saver = tf.train.Saver(tf.global_variables())
  saver.restore(sess, start_checkpoint)




def create_GRU_crnn_model(fingerprint_input, model_settings, is_training):
    """Builds a Convolutional Recurrent model.

    This model is an improved version of CNN for speech command recognition
    which has recurrent layers at the end of convolutional layers.

    Here's the layout of the graph:

    (fingerprint input)
        v
    [Conv2D]<-(weights)
        v
    [BiasAdd]<-(bias)
        v
     [Relu]
        v
    [Recurrent cell]
        v
    [FC layer]<-(weights)
        v
    [BiasAdd]
        v
    [softmax]
        v

    Args: 
      fingerprint_input: TensorFlow node that will output audio feature vectors.
      model_settings: Dictionary of information about the model.
      is_training: Whether the model is going to be used for training.

    Returns:
      TensorFlow node outputting logits results, and optionally a dropout
      placeholder.
    """
    
    if is_training:
        dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')
    input_frequency_size = model_settings['dct_coefficient_count']
    input_time_size = model_settings['spectrogram_length']
    fingerprint_4d = tf.reshape(fingerprint_input,
                                [-1, input_time_size, input_frequency_size, 1])

    layer_norm = False
    bidirectional = True
    
    # CNN Model
    #first_filter_width = 8
    first_filter_width = 1
    first_filter_height = 5
    #first_filter_count = 64
    first_filter_count = 32
    #stride_x = 1
    stride_x = 4
    stride_y = 1
    
    first_weights = tf.Variable(
            tf.truncated_normal(
                [first_filter_height, first_filter_width, 1, first_filter_count],
                stddev=0.01))

    first_bias = tf.Variable(tf.zeros([first_filter_count]))
    
    first_conv = tf.nn.conv2d(fingerprint_4d, first_weights,
                              [1, stride_y, stride_x, 1], 'VALID') + first_bias

    
    first_relu = tf.nn.relu(first_conv)
    if is_training:
        first_dropout = tf.nn.dropout(first_relu, dropout_prob)
    else:
        first_dropout = first_relu

    first_conv_output_width = int(math.floor((input_frequency_size - first_filter_width + stride_x) /
                                              stride_x))
    first_conv_output_height = int(math.floor((input_time_size - first_filter_height + stride_y) /
                                               stride_y))

    # GRU Model
    num_layers = 2
    #RNN_units = 128
    RNN_units = 8


    flow = tf.reshape(first_dropout, [-1, first_conv_output_height,
                                      first_conv_output_width * first_filter_count])
        
    forward_cell, backward_cell = [], []
    
    
    for i in range(num_layers):
        forward_cell.append(tf.contrib.rnn.GRUCell(RNN_units))
        backward_cell.append(tf.contrib.rnn.GRUCell(RNN_units))

    outputs, output_state_fw, output_state_bw = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(forward_cell, backward_cell, flow, dtype=tf.float32)
    flow_dim = first_conv_output_height * RNN_units * 2
    flow = tf.reshape(outputs, [-1, flow_dim])
    
    #fc_output_channels = 256
    fc_output_channels = 64
    fc_weights = tf.get_variable('fcw', shape=[flow_dim,fc_output_channels],
                                 initializer=tf.contrib.layers.xavier_initializer())

    fc_bias = tf.Variable(tf.zeros([fc_output_channels]))
    fc = tf.nn.relu(tf.matmul(flow, fc_weights) + fc_bias)
    
    if is_training:
        final_fc_input = tf.nn.dropout(fc, dropout_prob)
    else:
        final_fc_input = fc

    label_count = model_settings['label_count']

    final_fc_weights = tf.Variable(tf.truncated_normal([fc_output_channels, label_count], stddev=0.01))
    final_fc_bias = tf.Variable(tf.zeros([label_count]))
    final_fc = tf.matmul(final_fc_input, final_fc_weights) + final_fc_bias
    
    if is_training:
        return final_fc, dropout_prob
    else:
        return final_fc




def create_LSTM_crnn_model(fingerprint_input, model_settings, is_training):
    """Builds a Convolutional Recurrent model.

    This model is an improved version of CNN for speech command recognition
    which has recurrent layers at the end of convolutional layers.

    Here's the layout of the graph:

    (fingerprint input)
        v
    [Conv2D]<-(weights)
        v
    [BiasAdd]<-(bias)
        v
     [Relu]
        v
    [Recurrent cell]
        v
    [FC layer]<-(weights)
        v
    [BiasAdd]
        v
    [softmax]
        v

    Args: 
      fingerprint_input: TensorFlow node that will output audio feature vectors.
      model_settings: Dictionary of information about the model.
      is_training: Whether the model is going to be used for training.

    Returns:
      TensorFlow node outputting logits results, and optionally a dropout
      placeholder.
    """
    
    if is_training:
        dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')
    input_frequency_size = model_settings['dct_coefficient_count']
    input_time_size = model_settings['spectrogram_length']

    
    fingerprint_4d = tf.reshape(fingerprint_input,
                                [-1, input_time_size, input_frequency_size, 1])

    layer_norm = False
    bidirectional = True


    # CNN Model
    #first_filter_width = 8
    first_filter_width = 5
    first_filter_height = 20
    #first_filter_count = 64
    first_filter_count = 32
    #stride_x = 1
    stride_x = 8
    stride_y = 2
    
    first_weights = tf.Variable(
            tf.truncated_normal(
                [first_filter_height, first_filter_width, 1, first_filter_count],
                stddev=0.01))

    first_bias = tf.Variable(tf.zeros([first_filter_count]))
    
    first_conv = tf.nn.conv2d(fingerprint_4d, first_weights,
                              [1, stride_y, stride_x, 1], 'VALID') + first_bias



    first_relu = tf.nn.relu(first_conv)
    if is_training:
        first_dropout = tf.nn.dropout(first_relu, dropout_prob)
    else:
        first_dropout = first_relu

    first_conv_output_width = int(math.floor((input_frequency_size - first_filter_width + stride_x) /
                                              stride_x))
    first_conv_output_height = int(math.floor((input_time_size - first_filter_height + stride_y) /
                                               stride_y))


    # GRU Model
    #num_layers = 2
    num_layers = 3
    #RNN_units = 128
    RNN_units = 8


    flow = tf.reshape(first_dropout, [-1, first_conv_output_height,
                                      first_conv_output_width * first_filter_count])
        

    forward_cell, backward_cell = [], []
    
    
    for i in range(num_layers):
        forward_cell.append(tf.contrib.rnn.LSTMCell(RNN_units))
        backward_cell.append(tf.contrib.rnn.LSTMCell(RNN_units))

    outputs, output_state_fw, output_state_bw = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(forward_cell, backward_cell, flow, dtype=tf.float32)
    flow_dim = first_conv_output_height * RNN_units * 2
    
    
    flow = tf.reshape(outputs, [-1, flow_dim])
    
    #fc_output_channels = 256
    fc_output_channels = 64
    fc_weights = tf.get_variable('fcw', shape=[flow_dim,fc_output_channels],
                                 initializer=tf.contrib.layers.xavier_initializer())

    fc_bias = tf.Variable(tf.zeros([fc_output_channels]))
    fc = tf.nn.relu(tf.matmul(flow, fc_weights) + fc_bias)


    
    if is_training:
        final_fc_input = tf.nn.dropout(fc, dropout_prob)
    else:
        final_fc_input = fc

    label_count = model_settings['label_count']

    final_fc_weights = tf.Variable(tf.truncated_normal([fc_output_channels, label_count], stddev=0.01))
    final_fc_bias = tf.Variable(tf.zeros([label_count]))
    final_fc = tf.matmul(final_fc_input, final_fc_weights) + final_fc_bias

    if is_training:
        return final_fc, dropout_prob
    else:
        return final_fc




def create_GRU_rcnn_model(fingerprint_input, model_settings, is_training):
    """Builds a Convolutional Recurrent model.

    This model is an improved version of CNN for speech command recognition
    which has recurrent layers at the end of convolutional layers.

    Here's the layout of the graph:

    (fingerprint input)
        v
    [Conv2D]<-(weights)
        v
    [BiasAdd]<-(bias)
        v
     [Relu]
        v
    [Recurrent cell]
        v
    [FC layer]<-(weights)
        v
    [BiasAdd]
        v
    [softmax]
        v

    Args: 
      fingerprint_input: TensorFlow node that will output audio feature vectors.
      model_settings: Dictionary of information about the model.
      is_training: Whether the model is going to be used for training.

    Returns:
      TensorFlow node outputting logits results, and optionally a dropout
      placeholder.
    """
    
    if is_training:
        dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')
    
    
    
    
    input_frequency_size = model_settings['dct_coefficient_count']
    input_time_size = model_settings['spectrogram_length']
    
    layer_norm = False
    bidirectional = True
    

    # RNN Model
    num_layers = 2
    RNN_units = 128


    flow = tf.reshape(fingerprint_input, [-1, input_time_size, input_frequency_size])
        

    forward_cell, backward_cell = [], []
    
    
    for i in range(num_layers):
        forward_cell.append(tf.contrib.rnn.GRUCell(RNN_units))
        backward_cell.append(tf.contrib.rnn.GRUCell(RNN_units))

    outputs, output_state_fw, output_state_bw = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(forward_cell, backward_cell, flow, dtype=tf.float32)
    
    

    #flow_dim = 3840
    flow_dim = input_time_size * RNN_units * 2
    flow = tf.reshape(outputs, [-1, flow_dim])

    
    fc_output_channels = 1960
    fc_weights = tf.get_variable('fcw', shape=[flow_dim,fc_output_channels],
                                 initializer=tf.contrib.layers.xavier_initializer())

    fc_bias = tf.Variable(tf.zeros([fc_output_channels]))
    fc = tf.nn.relu(tf.matmul(flow, fc_weights) + fc_bias)


    if is_training:
        cnn_input = tf.nn.dropout(fc, dropout_prob)
    else:
        cnn_input = fc



    # CNN Model

    fingerprint_4d = tf.reshape(cnn_input,
                                [-1, input_time_size, input_frequency_size, 1])

    first_filter_width = 8
    first_filter_height = 20
    first_filter_count = 64
    stride_x = 1
    stride_y = 2

    
    first_weights = tf.Variable(
            tf.truncated_normal(
                [first_filter_height, first_filter_width, 1, first_filter_count],
                stddev=0.01))

    first_bias = tf.Variable(tf.zeros([first_filter_count]))
    
    first_conv = tf.nn.conv2d(fingerprint_4d, first_weights,
                              [1, stride_y, stride_x, 1], 'VALID') + first_bias


    first_relu = tf.nn.relu(first_conv)
    if is_training:
        first_dropout = tf.nn.dropout(first_relu, dropout_prob)
    else:
        first_dropout = first_relu

    first_conv_output_width = int(math.floor((input_frequency_size - first_filter_width + stride_x) /
                                              stride_x))
    first_conv_output_height = int(math.floor((input_time_size - first_filter_height + stride_y) /
                                               stride_y))


    first_dropout_shape = first_dropout.get_shape().as_list()
    output_channels = first_dropout_shape[1] * first_dropout_shape[2] * first_dropout_shape[3] #31680
    
    flattened_conv = tf.reshape(first_dropout, [-1, output_channels])



    label_count = model_settings['label_count']

    final_fc_weights = tf.Variable(tf.truncated_normal([output_channels, label_count], stddev=0.01))
    final_fc_bias = tf.Variable(tf.zeros([label_count]))
    final_fc = tf.matmul(flattened_conv, final_fc_weights) + final_fc_bias


    if is_training:
        return final_fc, dropout_prob
    else:
        return final_fc




def create_LSTM_rcnn_model(fingerprint_input, model_settings, is_training):
    """Builds a Convolutional Recurrent model.

    This model is an improved version of CNN for speech command recognition
    which has recurrent layers at the end of convolutional layers.

    Here's the layout of the graph:

    (fingerprint input)
        v
    [Conv2D]<-(weights)
        v
    [BiasAdd]<-(bias)
        v
     [Relu]
        v
    [Recurrent cell]
        v
    [FC layer]<-(weights)
        v
    [BiasAdd]
        v
    [softmax]
        v

    Args: 
      fingerprint_input: TensorFlow node that will output audio feature vectors.
      model_settings: Dictionary of information about the model.
      is_training: Whether the model is going to be used for training.

    Returns:
      TensorFlow node outputting logits results, and optionally a dropout
      placeholder.
    """
    
    if is_training:
        dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')
    
    
    
    
    input_frequency_size = model_settings['dct_coefficient_count']
    input_time_size = model_settings['spectrogram_length']
    
    layer_norm = False
    bidirectional = True
    

    # RNN Model
    num_layers = 2
    RNN_units = 128


    flow = tf.reshape(fingerprint_input, [-1, input_time_size, input_frequency_size])
        

    forward_cell, backward_cell = [], []
    
    
    for i in range(num_layers):
        forward_cell.append(tf.contrib.rnn.LSTMCell(RNN_units))
        backward_cell.append(tf.contrib.rnn.LSTMCell(RNN_units))

    outputs, output_state_fw, output_state_bw = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(forward_cell, backward_cell, flow, dtype=tf.float32)
    
    

    #flow_dim = 3840
    flow_dim = input_time_size * RNN_units * 2
    flow = tf.reshape(outputs, [-1, flow_dim])

    
    fc_output_channels = 1960
    fc_weights = tf.get_variable('fcw', shape=[flow_dim,fc_output_channels],
                                 initializer=tf.contrib.layers.xavier_initializer())

    fc_bias = tf.Variable(tf.zeros([fc_output_channels]))
    fc = tf.nn.relu(tf.matmul(flow, fc_weights) + fc_bias)


    if is_training:
        cnn_input = tf.nn.dropout(fc, dropout_prob)
    else:
        cnn_input = fc



    # CNN Model

    fingerprint_4d = tf.reshape(cnn_input,
                                [-1, input_time_size, input_frequency_size, 1])

    first_filter_width = 8
    first_filter_height = 20
    first_filter_count = 64
    stride_x = 1
    stride_y = 2

    
    first_weights = tf.Variable(
            tf.truncated_normal(
                [first_filter_height, first_filter_width, 1, first_filter_count],
                stddev=0.01))

    first_bias = tf.Variable(tf.zeros([first_filter_count]))
    
    first_conv = tf.nn.conv2d(fingerprint_4d, first_weights,
                              [1, stride_y, stride_x, 1], 'VALID') + first_bias


    first_relu = tf.nn.relu(first_conv)
    if is_training:
        first_dropout = tf.nn.dropout(first_relu, dropout_prob)
    else:
        first_dropout = first_relu

    first_conv_output_width = int(math.floor((input_frequency_size - first_filter_width + stride_x) /
                                              stride_x))
    first_conv_output_height = int(math.floor((input_time_size - first_filter_height + stride_y) /
                                               stride_y))


    first_dropout_shape = first_dropout.get_shape().as_list()
    output_channels = first_dropout_shape[1] * first_dropout_shape[2] * first_dropout_shape[3] #31680
    
    flattened_conv = tf.reshape(first_dropout, [-1, output_channels])



    label_count = model_settings['label_count']

    final_fc_weights = tf.Variable(tf.truncated_normal([output_channels, label_count], stddev=0.01))
    final_fc_bias = tf.Variable(tf.zeros([label_count]))
    final_fc = tf.matmul(flattened_conv, final_fc_weights) + final_fc_bias


    if is_training:
        return final_fc, dropout_prob
    else:
        return final_fc


