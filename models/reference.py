# reference implementation for ease of understanding
# import libraries
# transfer learning model
from keras.applications.resnet50 import ResNet50, preprocess_input
# keras layers
from keras.layers import GlobalMaxPooling2D
from keras.layers import BatchNormalization
from keras.layers import MaxPooling2D
from keras.layers import UpSampling2D
from keras.layers import Concatenate
from keras.layers import Activation
from keras.layers import Dropout
from keras.layers import Lambda
from keras.layers import Conv2D
from keras.layers import Input
from keras.layers import Dense
from keras.layers import LSTM
# keras utils
from keras.models import Model
from keras import backend as K


# transfer learning layers
def get_resnet(num_of_layers):
    '''
    Returns the non-trainable ResNet layers as a Keras Model

    Given number of layers, this function iterates over the ResNet50 model
    and returns the model till the layers where it matches the corresponding
    layer name. Here 'add_{}'.format(num_of_layers) is used to identify the
    end layer.

    Parameters
    ----------
    num_of_layers: int
        number of layers you want to select from the ResNet50 model

    Returns
    -------
    Model
        a non-trainable Keras Model instance containing the layers
        from ResNet50 model
    '''
    resnet = ResNet50(weights='imagenet', include_top=False,
                      input_shape=(256, 256, 3))
    i = 0
    while(True):
        if resnet.layers[i].name == 'add_{}'.format(num_of_layers):
            break
        i += 1
    model = Model(inputs=resnet.layers[0].input,
                  outputs=resnet.layers[i].output,
                  name='resnet')
    for layer in model.layers:
        layer.trainable = False
    model.compile('adadelta', 'mse')
    return model


# dense block
def get_dense_block(prev_shape, count, name, id):
    '''
    Returns a Dense block as a Keras Model

    Given the count of dense branches, this function returns a Dense block.
    A Dense block essentially allows you to add more feature maps while
    retaining the original layers. Inspired from the brilliant paper by
    Huang et al published at arXiv. https://arxiv.org/abs/1608.06993

    Parameters
    ----------
    prev_shape: tuple of int
        shape of the previous layer output, or output of K.int_shape
        on the previous layer
    count: int
        the number of forward connections in the Dense block.
    name: str
        the name of the model defined in the calling function
    id: int
        the number of this block in namespace of model in calling function

    Returns
    -------
    Model
        a Keras model instance for the configured Dense block
    '''
    name = '{}_dense'.format(name)
    input_layer = Input(batch_shape=prev_shape,
                        name='{}_{}_input'.format(name, id))
    for i in range(1, count + 1):
        conv = Conv2D(filters=64, kernel_size=1, strides=1, padding='same',
                      kernel_initializer='he_uniform',
                      bias_initializer='he_uniform',
                      name='{}_{}_conv_{}a'.format(name, id, i))
        conv1 = conv(input_layer if i == 1 else layer)
        norm1 = BatchNormalization(name='{}_{}_norm_{}a'.format(name, id, i))(conv1)
        relu1 = Activation(activation='relu',
                           name='{}_{}_relu_{}a'.format(name, id, i))(norm1)
        conv2 = Conv2D(filters=16, kernel_size=3, strides=1, padding='same',
                       kernel_initializer='he_uniform',
                       bias_initializer='he_uniform',
                       name='{}_{}_conv_{}b'.format(name, id, i))(relu1)
        norm2 = BatchNormalization(name='{}_{}_norm_{}b'.format(name, id, i))(conv2)
        relu2 = Activation(activation='relu',
                           name='{}_{}_relu_{}b'.format(name, id, i))(norm2)
        concat = Concatenate(name='{}_{}_concat_{}'.format(name, id, i))
        layer = concat([input_layer if i == 1 else layer, relu2])
    return Model(inputs=input_layer,
                 outputs=layer,
                 name='{}_{}'.format(name, id))


# reduce block
def get_reduce_block(prev_shape, reduce_filters, reduce, name, id):
    '''
    Returns a Reduce block as a Keras Model

    Given the `reduce_filters` count, this block reduces the number of filters
    in 2 steps. A 1x1 convolution to reduce the filters. Followed by either
    a size preserving 3x3 convolution or size reducing 3x3 convolution based
    on the `reduce` argument. Inspired from the weight saving convolutions
    suggested in the development of Inception-V3, published by Szegeddy et al
    in arXiv. https://arxiv.org/abs/1512.00567

    Parameters
    ----------
    prev_shape: a tuple of int
        shape of the previous layer output, or output of K.int_shape
        on the previous layer
    reduce_filters: int
        the number of target filters, usually lesser than the number
        of filters in previous layer
    reduce: bool
        True to reduce the X-Y dimensions of the batch, False to preserve
    name: str
        the name of the model defined in the calling function
    id: int
        the number of this block in namespace of model in calling function

    Returns
    -------
    Model
        a Keras model instance for the configured Reduce block
    '''
    name = '{}_reduce'.format(name)
    input_layer = Input(batch_shape=prev_shape,
                        name='{}_{}_input'.format(name, id))
    reduce_a = Conv2D(filters=reduce_filters,
                      kernel_size=1, strides=1, padding='same',
                      kernel_initializer='he_uniform',
                      bias_initializer='he_uniform',
                      name='{}_{}_a_conv'.format(name, id))(input_layer)
    norm_a = BatchNormalization(name='{}_{}_a_norm'.format(name, id))(reduce_a)
    relu_a = Activation(activation='relu',
                        name='{}_{}_a_relu'.format(name, id))(norm_a)
    reduce_b = Conv2D(filters=reduce_filters, kernel_size=3,
                      strides=1, padding='same',
                      kernel_initializer='he_uniform',
                      bias_initializer='he_uniform',
                      name='{}_{}_b_conv'.format(name, id))(relu_a)
    norm_b = BatchNormalization(name='{}_{}_b_norm'.format(name, id))(reduce_b)
    output_layer = Activation(activation='relu',
                              name='{}_{}_b_relu'.format(name, id))(norm_b)
    if reduce:
        output_layer = MaxPooling2D(pool_size=(3, 3),
                                    strides=(2, 2),
                                    padding='same',
                                    name='{}_{}_b_pool'.format(name, id))(output_layer)
    return Model(inputs=input_layer,
                 outputs=output_layer,
                 name='{}_{}'.format(name, id))


# increase block
def get_scale_block(prev_shape, name, id):
    '''
    Returns a Conv2D block and UpSampling2D layer as a Keras Model

    This block performs a Convolution followed by UpSampling transformation,
    leading to doubling the X-Y dimension of the batch.

    Parameters
    ----------
    prev_shape: a tuple of int
        shape of the previous layer output, or output of K.int_shape
        on the previous layer
    name: str
        the name of the model defined in the calling function
    id: int
        the number of this block in namespace of model in calling function

    Returns
    -------
    Model
        a Keras model instance for the configured Transposed Conv2D block
    '''
    name = '{}_increase'.format(name)
    input_layer = Input(batch_shape=prev_shape,
                        name='{}_{}_input'.format(name, id))
    conv = Conv2D(filters=K.int_shape(input_layer)[-1],
                  kernel_size=3, strides=1, padding='same',
                  kernel_initializer='he_uniform',
                  bias_initializer='he_uniform',
                  name='{}_{}_conv'.format(name, id))(input_layer)
    norm = BatchNormalization(name='{}_{}_norm'.format(name, id))(conv)
    relu = Activation(activation='relu',
                      name='{}_{}_relu'.format(name, id))(norm)
    output_layer = UpSampling2D(size=(2, 2),
                                name='{}_{}_upsm'.format(name, id))(relu)
    return Model(inputs=input_layer,
                 outputs=output_layer,
                 name='{}_{}'.format(name, id))


# fully connected block
def get_fully_connected_block(prev_shape, units, dropout_rate, name, id):
    '''
    Returns a Fully Connected Block with specified units and dropout rate.

    This block returns a Dense layer, followed by BatchNormalization,
    ReLu Activation and Dropout.

    Parameters
    ----------
    prev_shape: a tuple of int
        shape of the previous layer output, or output of K.int_shape
        on the previous layer
    units: int
        number of units in the Dense Layer
    dropout_rate: float
        dropout rate for the Dense Layer during training
    name: str
        the name of the model defined in the calling function
    id: int
        the number of this block in namespace of model in calling function

    Returns
    -------
    Model
        a Keras Model instance for Fully Connected block
    '''
    name = '{}_fc'.format(name)
    input_layer = Input(batch_shape=prev_shape,
                        name='{}_{}_input'.format(name, id))
    fc = Dense(units=units, kernel_initializer='he_uniform',
               bias_initializer='he_uniform',
               name='{}_{}_fc'.format(name, id))(input_layer)
    norm = BatchNormalization(name='{}_{}_norm'.format(name, id))(fc)
    relu = Activation(activation='relu',
                      name='{}_{}_relu'.format(name, id))(norm)
    output_layer = Dropout(rate=dropout_rate,
                           name='{}_{}_drop'.format(name, id))(relu)
    return Model(inputs=input_layer,
                 outputs=output_layer,
                 name='{}_{}'.format(name, id))


# encoder
def get_encoder(prev_shape, num_of_layers, name):
    '''
    Returns the Encoder as a Keras Model

    Given the `num_of_layers` count, this function extracts those number of
    layers from the Transfer Learning model, ResNet 50 here, and builds a
    Feature Encoder using those layers and multiple Reduce and Dense blocks.

    Parameters
    ----------
    prev_shape: a tuple of int
        shape of the previous layer output, or output of K.int_shape
        on the previous layer
    num_of_layers: int
        number of layers you want to select from the ResNet50 model
    name: str
        the name of the model defined in the calling function

    Returns
    -------
    Model
        a Keras model instance for the configured Encoder
    '''
    name = '{}_encode'.format(name)
    input_layer = Input(batch_shape=prev_shape,
                        name='{}_input'.format(name))
    resnet = get_resnet(num_of_layers)(input_layer)
    reduce_1 = get_reduce_block(K.int_shape(resnet),
                                128, True, name, 1)(resnet)
    dense_1 = get_dense_block(K.int_shape(reduce_1), 8, name, 1)(reduce_1)
    reduce_2 = get_reduce_block(K.int_shape(dense_1),
                                128, True, name, 2)(dense_1)
    dense_2 = get_dense_block(K.int_shape(reduce_2), 8, name, 2)(reduce_2)
    reduce_3 = get_reduce_block(K.int_shape(dense_2),
                                128, True, name, 3)(dense_2)
    dense_3 = get_dense_block(K.int_shape(reduce_3), 8, name, 3)(reduce_3)
    reduce_4 = get_reduce_block(K.int_shape(dense_3),
                                128, False, name, 4)(dense_3)
    output_layer = get_dense_block(K.int_shape(reduce_4), 8, name, 4)(reduce_4)
    return Model(inputs=input_layer,
                 outputs=output_layer,
                 name='{}'.format(name))


# decoder
def get_decoder(prev_shape, name):
    '''
    Returns the Decoder as a Keras Model

    This functions builds a Decoder/Generator using multiple
    Reduce and Increase blocks.

    Parameters
    ----------
    prev_shape: a tuple of int
        shape of the previous layer output, or output of K.int_shape
        on the previous layer
    name: str
        the name of the model defined in the calling function

    Returns
    -------
    Model
        a Keras model instance for the configured Decoder
    '''
    name = '{}_decode'.format(name)
    input_layer = Input(batch_shape=prev_shape,
                        name='{}_input'.format(name))
    shape_1 = K.int_shape(input_layer)
    reduce_1 = get_reduce_block(shape_1, shape_1[-1] // 2,
                                False, name, 1)(input_layer)
    scale_1 = get_scale_block(K.int_shape(reduce_1), name, 1)(reduce_1)

    shape_2 = K.int_shape(scale_1)
    reduce_2 = get_reduce_block(shape_2, shape_2[-1] // 2,
                                False, name, 2)(scale_1)
    scale_2 = get_scale_block(K.int_shape(reduce_2), name, 2)(reduce_2)

    shape_3 = K.int_shape(scale_2)
    reduce_3 = get_reduce_block(shape_3, shape_3[-1] // 2,
                                False, name, 3)(scale_2)
    scale_3 = get_scale_block(K.int_shape(reduce_3), name, 3)(reduce_3)

    shape_4 = K.int_shape(scale_3)
    reduce_4 = get_reduce_block(shape_4, shape_4[-1] // 2,
                                False, name, 4)(scale_3)
    scale_4 = get_scale_block(K.int_shape(reduce_4), name, 4)(reduce_4)

    shape_5 = K.int_shape(scale_4)
    reduce_5 = get_reduce_block(shape_5, shape_5[-1] // 2,
                                False, name, 5)(scale_4)
    output_layer = get_scale_block(K.int_shape(reduce_5), name, 5)(reduce_5)
    return Model(inputs=input_layer,
                 outputs=output_layer,
                 name='{}'.format(name))


# AutoEncoder
def get_autoencoder(name):
    '''
    Returns the AutoEncoder as a Keras Model

    This functions builds an AutoEncoder using the Encoder and Decoder

    Parameters
    ----------
    name: str
        the name of the model defined in the calling function

    Returns
    -------
    Model
        a Keras model instance for the configured AutoEncoder
    '''
    input_layer = Input(batch_shape=(None, 256, 256, 3), name='input')
    encoder = get_encoder(K.int_shape(input_layer), 3, name)(input_layer)
    decoder = get_decoder(K.int_shape(encoder), name)(encoder)
    output_layer = Conv2D(filters=1, kernel_size=3, strides=1,
                          padding='same', activation='sigmoid',
                          kernel_initializer='he_uniform',
                          bias_initializer='he_uniform',
                          name='output')(decoder)
    return Model(inputs=input_layer, outputs=output_layer, name=name)


# Classifier
def get_classifier(name):
    '''
    Returns the Classifier as a Keras Model

    This function builds a Classifier using the Encoder and Fully Connected
    Blocks.

    Parameters
    ----------
    name: str
        the name of the model defined in the calling function

    Returns
    -------
    Model
        a Keras model instance for the configured Classifier
    '''
    input_layer = Input(batch_shape=(None, 256, 256, 3), name='input')
    encoder = get_encoder(K.int_shape(input_layer), 3, name)(input_layer)
    features = GlobalMaxPooling2D(name='features')(encoder)
    fc_block1 = get_fully_connected_block(K.int_shape(features),
                                          256, 0.2, name, 1)(features)
    fc_block2 = get_fully_connected_block(K.int_shape(fc_block1),
                                          256, 0.2, name, 2)(fc_block1)
    output_layer = Dense(units=5, activation='softmax',
                         kernel_initializer='he_uniform',
                         bias_initializer='he_uniform',
                         name='classes')(fc_block2)
    return Model(inputs=input_layer, outputs=output_layer, name=name)


# Paraclassifier
def get_paraclassifier(name):
    '''
    Returns the Paraclassifier as a Keras Model

    This function builds a LSTM based Paraclassifier on top of predictions
    from the Classifier using LSTM blocks.

    Parameters
    ----------
    name: str
        the name of the model defined in the calling function

    Returns
    -------
    Model
        a Keras Model instance for the configured Paraclassifier
    '''
    input_layer = Input(batch_shape=(None, None, 256), name='parainput')
    lstm1 = LSTM(units=128, return_sequences=True,
                 name='{}_lstm_{}'.format(name, 1))(input_layer)
    lstm2 = LSTM(units=64, return_sequences=True,
                 name='{}_lstm_{}'.format(name, 2))(lstm1)
    lstm3 = LSTM(units=32, name='{}_lstm_{}'.format(name, 3))(lstm2)
    output_layer = Dense(units=5, activation='softmax',
                         kernel_initializer='he_uniform',
                         bias_initializer='he_uniform',
                         name='paraclass')(lstm3)
    return Model(inputs=input_layer, outputs=output_layer, name=name)


def get_preprocess():
    '''
    Returns the reference to the preprocess function of the transfer layers.

    Parameters
    ----------
    None

    Returns
    -------
    function
        reference to the preprocess function of the transfer layers.
    '''
    return preprocess_input


def make_docs(name):
    '''
    Generates the doc files showing the structure of the AutoEncoder

    This function uses Keras vis_utils to generate the images for each block
    in the AutoEncoder.

    Parameters
    ----------
    name: str
        the name of the model to show on the images

    Returns
    -------
    None
    '''
    from keras.utils import plot_model
    import os

    model = get_autoencoder(name)
    path = os.path.join('docs', 'reference')
    if not os.path.exists(path):
        os.mkdir(path)
    plot_model(model, to_file=os.path.join(path, 'autoencoder.png'),
               show_shapes=True, show_layer_names=True, rankdir='TB')
    plot_model(model.layers[1], to_file=os.path.join(path, 'encoder.png'),
               show_shapes=True, show_layer_names=True, rankdir='TB')
    plot_model(model.layers[2], to_file=os.path.join(path, 'decoder.png'),
               show_shapes=True, show_layer_names=True, rankdir='TB')
    plot_model(model.layers[1].layers[1], to_file=os.path.join(path, 'resnet.png'),
               show_shapes=True, show_layer_names=True, rankdir='TB')
    plot_model(model.layers[1].layers[2], to_file=os.path.join(path, 'reduce_true.png'),
               show_shapes=True, show_layer_names=True, rankdir='TB')
    plot_model(model.layers[1].layers[8], to_file=os.path.join(path, 'reduce_false.png'),
               show_shapes=True, show_layer_names=True, rankdir='TB')
    plot_model(model.layers[1].layers[3], to_file=os.path.join(path, 'dense_8.png'),
               show_shapes=True, show_layer_names=True, rankdir='TB')
    plot_model(model.layers[2].layers[2], to_file=os.path.join(path, 'increase.png'),
               show_shapes=True, show_layer_names=True, rankdir='TB')
