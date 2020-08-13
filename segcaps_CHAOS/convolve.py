import tensorflow as tf
import keras.backend as K


def convolove(input_tensor, kernel, strides, padding, num_caps, num_atom):
    """
    Convolve several kernels with tensor
    :param x: tf tensor, [num_caps_input, batch_size, height, width, num_atom_input]
    :param kernel: tf tensor, [num_caps_input, kernel_size, kernel_size, num_atom_input, num_atom_out,  num_atom * num_capsule]
    :return: tf tensor, [batch_size, num_caps_input, height, width, num_caps_input, num_atom_output]
    """
    # get shapes
    num_caps_input, batch_size, height, width, _ = input_tensor.shape

    # convert tensors to list of tensors
    w_tf_list = tf.split(kernel, num_caps_input)
    inputs_splits_tf = tf.split(input_tensor, num_caps_input)

    # convolve
    votes = [
        K.conv2d(x=tf.squeeze(x, [0]), kernel=tf.squeeze(kernel_i, [0]), strides=(strides, strides), padding=padding,
                 data_format='channels_last')
        for x, kernel_i in zip(inputs_splits_tf, w_tf_list)]

    # reshape
    votes = [K.reshape(x, (-1, 1, votes[0].shape[1], votes[0].shape[2], num_caps, num_atom)) for x in votes]
    votes = tf.concat(votes, axis=1)
    _, _, conv_height, conv_width, _, _ = votes.get_shape()
    return votes, conv_height, conv_width
