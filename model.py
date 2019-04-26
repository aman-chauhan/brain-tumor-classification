from keras.layers import GlobalMaxPooling2D
from keras.layers import BatchNormalization
from keras.layers import TimeDistributed
from keras.layers import Concatenate
from keras.layers import Activation
from keras.layers import Lambda
from keras.layers import Conv2D
from keras.layers import Input
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import Model
from keras import backend as K


def get_dense_block(prev_layer, count, name, id):
    name = '{}_dense'.format(name)
    for i in range(1, count + 1):
        conv = Conv2D(filters=64, kernel_size=1, strides=1, padding='same',
                      kernel_initializer='he_uniform',
                      bias_initializer='he_uniform',
                      name='{}_{}_conv_{}a'.format(name, id, i))
        conv1 = conv(prev_layer if i == 1 else layer)
        norm1 = BatchNormalization(scale=False,
                                   name='{}_{}_norm_{}a'.format(name, id, i))(conv1)
        relu1 = Activation(activation='relu',
                           name='{}_{}_relu_{}a'.format(name, id, i))(norm1)
        conv2 = Conv2D(filters=16, kernel_size=3, strides=1, padding='same',
                       kernel_initializer='he_uniform',
                       bias_initializer='he_uniform',
                       name='{}_{}_conv_{}b'.format(name, id, i))(relu1)
        norm2 = BatchNormalization(scale=False,
                                   name='{}_{}_norm_{}b'.format(name, id, i))(conv2)
        relu2 = Activation(activation='relu',
                           name='{}_{}_relu_{}b'.format(name, id, i))(norm2)
        concat = Concatenate(name='{}_{}_concat_{}'.format(name, id, i))
        layer = concat([prev_layer if i == 1 else layer, relu2])
    return layer


def get_reduce_block(prev_layer, reduce_filters, reduce, name, id):
    name = '{}_reduce'.format(name)
    reduce_a = Conv2D(filters=reduce_filters,
                      kernel_size=1, strides=1, padding='same',
                      kernel_initializer='he_uniform',
                      bias_initializer='he_uniform',
                      name='{}_{}_a_conv'.format(name, id))(prev_layer)
    norm_a = BatchNormalization(scale=False,
                                name='{}_{}_a_norm'.format(name, id))(reduce_a)
    relu_a = Activation(activation='relu',
                        name='{}_{}_a_relu'.format(name, id))(norm_a)
    reduce_b = Conv2D(filters=reduce_filters, kernel_size=3,
                      strides=2 if reduce else 1, padding='same',
                      kernel_initializer='he_uniform',
                      bias_initializer='he_uniform',
                      name='{}_{}_b_conv'.format(name, id))(relu_a)
    norm_b = BatchNormalization(scale=False,
                                name='{}_{}_b_norm'.format(name, id))(reduce_b)
    relu_b = Activation(activation='relu',
                        name='{}_{}_b_relu'.format(name, id))(norm_b)
    return relu_b


def get_scale_block(prev_layer, name, id):
    name = '{}_increase'.format(name)
    transpose = Conv2DTranspose(filters=K.int_shape(prev_layer)[-1],
                                kernel_size=3, strides=2, padding='same',
                                kernel_initializer='he_uniform',
                                bias_initializer='he_uniform',
                                name='{}_{}_transpose'.format(name, id))(prev_layer)
    norm = BatchNormalization(scale=False,
                              name='{}_{}_norm'.format(name, id))(transpose)
    relu = Activation(activation='relu',
                      name='{}_{}_relu'.format(name, id))(norm)
    return relu


def get_encoder(input_size, name):
    input_layer = Input(batch_shape=(None, input_size, input_size, 1),
                        name='{}_input'.format(name))
    reduce_0 = get_reduce_block(input_layer, 32, True, name, 0)
    reduce_1 = get_reduce_block(reduce_0, 64, True, name, 1)
    dense_1 = get_dense_block(reduce_1, 4, name, 1)
    reduce_2 = get_reduce_block(dense_1, 64, True, name, 2)
    dense_2 = get_dense_block(reduce_2, 4, name, 2)
    reduce_3 = get_reduce_block(dense_2, 64, True, name, 3)
    dense_3 = get_dense_block(reduce_3, 4, name, 3)
    reduce_4 = get_reduce_block(dense_3, 64, True, name, 4)
    dense_4 = get_dense_block(reduce_4, 4, name, 4)
    reduce_5 = get_reduce_block(dense_4, 64, True, name, 5)
    dense_5 = get_dense_block(reduce_5, 4, name, 5)
    output_layer = GlobalMaxPooling2D(name='{}_output'.format(name))(dense_5)
    return Model(inputs=input_layer,
                 outputs=output_layer,
                 name='{}'.format(name))


def get_classifier(input_size, name):
    input_layer = Input(batch_shape=(None, None, input_size, input_size, 1),
                        name='input')
    encoder = TimeDistributed(get_encoder(input_size, 'cnn'),
                              name='encoder')(input_layer)
    lstm1 = LSTM(units=128, return_sequences=True,
                 name='lstm_{}'.format(1))(encoder)
    lstm2 = LSTM(units=64, return_sequences=True,
                 name='lstm_{}'.format(2))(lstm1)
    lstm3 = LSTM(units=32, name='lstm_{}'.format(3))(lstm2)
    output_layer = Dense(units=5, activation='softmax',
                         kernel_initializer='he_uniform',
                         bias_initializer='he_uniform',
                         name='classes')(lstm3)
    return Model(inputs=input_layer, outputs=output_layer, name=name)


def make_docs(name):
    from keras.utils import plot_model
    import os

    model = get_classifier(256, name)
    plot_model(model, to_file=os.path.join('docs', 'model.png'),
               show_shapes=True, show_layer_names=True, rankdir='TB')
    plot_model(model.layers[1].layer, to_file=os.path.join('docs', 'cnn.png'),
               show_shapes=True, show_layer_names=True, rankdir='TB')
