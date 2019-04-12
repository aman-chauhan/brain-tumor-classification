# import libraries
# transfer learning model
from keras.applications.inception_v3 import InceptionV3, preprocess_input
# keras layers
from keras.layers import BatchNormalization
from keras.layers import Conv2DTranspose
from keras.layers import UpSampling2D
from keras.layers import Concatenate
from keras.layers import Activation
from keras.layers import Dropout
from keras.layers import Lambda
from keras.layers import Conv2D
from keras.layers import Input
from keras.layers import Dense
# keras utils
from keras.models import Model
from keras import backend as K


# transfer learning layers
def get_inception(num_of_layers, input_size):
    '''
    Returns the non-trainable Inception-V3 layers as a Keras Model

    Given number of layers, this function iterates over the Inception-V3
    model and returns the model till the layers where it matches the
    corresponding layer name. Here 'add_{}'.format(num_of_layers) is used
    to identify the end layer.

    Parameters
    ----------
    num_of_layers: int
        number of layers you want to select from the Inception-V3 model

    Returns
    -------
    Model
        a non-trainable Keras Model instance containing the layers
        from Inception-V3 model
    '''
    inception = InceptionV3(weights='imagenet', include_top=False,
                            input_shape=(input_size, input_size, 3))
    i = 0
    while(True):
        if inception.layers[i].name == 'mixed{}'.format(num_of_layers-1):
            break
        i += 1
    model = Model(inputs=inception.layers[0].input,
                  outputs=inception.layers[i].output,
                  name='inception')
    for layer in model.layers:
        layer.trainable = False
    model.compile('adadelta', 'mse')
    return model


# dense block
def get_dense_block(prev_layer, count, name, id):
    '''
    Returns a Keras Layer with Dense block transformation

    Given the count of dense branches, this function returns a Dense block.
    A Dense block essentially allows you to add more feature maps while
    retaining the original layers. Inspired from the brilliant paper by
    Huang et al published at arXiv. https://arxiv.org/abs/1608.06993

    Parameters
    ----------
    prev_layer: Layer
        the reference to previous layer
    count: int
        the number of forward connections in the Dense block.
    name: str
        the name of the model defined in the calling function
    id: int
        the number of this block in namespace of model in calling function

    Returns
    -------
    Layer
        a Keras Layer with the configured Dense block transformation
    '''
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


# reduce block
def get_reduce_block(prev_layer, reduce_filters, reduce, name, id):
    '''
    Returns a Keras Layer with Reduce block transformation

    Given the `reduce_filters` count, this block reduces the number of filters
    in 2 steps. A 1x1 convolution to reduce the filters. Followed by either
    a size preserving 3x3 convolution or size reducing 3x3 convolution based
    on the `reduce` argument. Inspired from the weight saving convolutions
    suggested in the development of Inception-V3, published by Szegeddy et al
    in arXiv. https://arxiv.org/abs/1512.00567

    Parameters
    ----------
    prev_layer: Layer
        the reference to previous layer
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
    Layer
        a Keras Layer with the configured Reduce block transformation
    '''
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


# scale block
def get_scale_block(prev_layer, name, id):
    '''
    Returns a Keras Layer with Tranposed Conv2D block transformation

    This block performs a Tranposed Convolution or Deconvolution operation,
    leading to doubling the X-Y dimension of the batch. Popularised by the
    papers, by Dumoulin et al in https://arxiv.org/abs/1603.07285v1 and Zeiler
    et al in https://www.matthewzeiler.com/mattzeiler/deconvolutionalnetworks.pdf

    Parameters
    ----------
    prev_layer: Layer
        the reference to previous layer
    name: str
        the name of the model defined in the calling function
    id: int
        the number of this block in namespace of model in calling function

    Returns
    -------
    Layer
        a Keras Layer with the configured Transposed Conv2D block transformation
    '''
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


# encoder
def get_encoder(prev_layer, num_of_layers, input_size, name):
    '''
    Returns a Keras Layer with Encoder transformation

    Given the `num_of_layers` count, this function extracts those number of
    layers from the Transfer Learning model, Inception-V3 here, and builds
    a Feature Encoder using those layers and multiple Reduce and Dense blocks.

    Parameters
    ----------
    prev_layer: Layer
        the reference to previous layer
    num_of_layers: int
        number of layers you want to select from the InceptionV3 model
    input_size: int
        the dimension of square image fed as input to the Model
    name: str
        the name of the model defined in the calling function

    Returns
    -------
    Layer
        a Keras Layer with the configured Encoder transformation
    '''
    name = '{}_encode'.format(name)
    inception = get_inception(num_of_layers, input_size)(prev_layer)
    reduce_1 = get_reduce_block(inception, 128, True, name, 1)
    dense_1 = get_dense_block(reduce_1, 8, name, 1)
    reduce_2 = get_reduce_block(dense_1, 128, True, name, 2)
    dense_2 = get_dense_block(reduce_2, 8, name, 2)
    reduce_3 = get_reduce_block(dense_2, 128, False, name, 3)
    dense_3 = get_dense_block(reduce_3, 8, name, 3)
    reduce_4 = get_reduce_block(dense_3, 128, False, name, 4)
    dense_4 = get_dense_block(reduce_4, 8, name, 4)
    return dense_4


# decoder
def get_decoder(prev_layer, name):
    '''
    Returns a Keras Layer with Decoder transformation

    This functions builds a Decoder/Generator using multiple
    Reduce and Scale blocks.

    Parameters
    ----------
    prev_layer: Layer
        the reference to previous layer
    name: str
        the name of the model defined in the calling function

    Returns
    -------
    Layer
        a Keras Layer with the configured Decoder transformation
    '''
    name = '{}_decode'.format(name)
    reduce_1 = get_reduce_block(prev_layer, K.int_shape(prev_layer)[-1] // 2,
                                False, name, 1)
    scale_1 = get_scale_block(reduce_1, name, 1)
    reduce_2 = get_reduce_block(scale_1, K.int_shape(scale_1)[-1] // 2,
                                False, name, 2)
    scale_2 = get_scale_block(reduce_2, name, 2)
    reduce_3 = get_reduce_block(scale_2, K.int_shape(scale_2)[-1] // 2,
                                False, name, 3)
    scale_3 = get_scale_block(reduce_3, name, 3)
    reduce_4 = get_reduce_block(scale_3, K.int_shape(scale_3)[-1] // 2,
                                False, name, 4)
    scale_4 = get_scale_block(reduce_4, name, 4)
    reduce_5 = get_reduce_block(scale_4, K.int_shape(scale_4)[-1] // 2,
                                False, name, 5)
    scale_5 = get_scale_block(reduce_5, name, 5)
    return scale_5


# AutoEncoder
def get_autoencoder(input_size, name):
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
    input_layer = Input(batch_shape=(None, input_size, input_size, 3),
                        name='input')
    encoder = get_encoder(input_layer, 3, input_size, name)
    decoder = get_decoder(encoder, name)
    output_layer = Conv2D(filters=1, kernel_size=3, strides=1,
                          padding='same', activation='sigmoid',
                          kernel_initializer='he_uniform',
                          bias_initializer='he_uniform',
                          name='output')(decoder)
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

    model = get_autoencoder(256, name)
    path = os.path.join('docs', 'inception')
    if not os.path.exists(path):
        os.mkdir(path)
    plot_model(model, to_file=os.path.join(path, 'autoencoder.png'),
               show_shapes=True, show_layer_names=True, rankdir='TB')
    plot_model(model.layers[1], to_file=os.path.join(path, 'inceptionv3.png'),
               show_shapes=True, show_layer_names=True, rankdir='TB')
