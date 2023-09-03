from tensorflow.keras import layers
import tensorflow.keras as tfk
from typing import List, Tuple

class ConvPoolBlock(layers.Layer):
    '''
    A layer that performs a convolution, batch normalization, max pooling, and dropout. \n
    CONV -> BN -> RELU -> MAXPOOL -> DROPOUT
    '''

    def __init__(self, conv_filters: int, conv_kernel: Tuple[int, int] = (3,3), conv_stride: Tuple[int, int] = (1,1), 
                 conv_padding: str = 'same', activation: str = 'relu', pool: bool = True, max_pool_kernel: Tuple[int, int] = (2,2),
                 pool_stride: Tuple[int, int] = None, pool_padding: str = 'same', batch_norm: bool = True , dropout: bool = True, dropout_rate: float = 0.2):
        '''
        Creates a layer that performs a convolution, batch normalization, max pooling, and dropout.
        :param conv_filters int: number of filters in the convolutional layer 
        :param conv_kernel Tuple[int, int]: kernel size of the convolutional layer. default (3,3)
        :param conv_stride Tuple[int, int]: stride of the convolutional layer. default (1,1)
        :param conv_padding str: padding of the convolutional layer. default 'same'
        :param activation str: activation function of the convolutional layer. default 'relu'
        :param pool bool: whether to use max pooling or not. default True
        :param max_pool_kernel Tuple[int, int]: kernel size of the max pooling layer. default (2,2)
        :param pool_stride Tuple[int, int]: stride of the max pooling layer. default None
        :param pool_padding str: padding of the max pooling layer. default 'same'
        :param batch_norm bool: whether to use batch normalization or not. default True
        :param dropout bool: whether to use dropout or not. default True
        :param dropout_rate float: dropout rate. default 0.2
        '''
        super(ConvPoolBlock, self).__init__()
        self.conv = tfk.layers.Conv2D(conv_filters, conv_kernel, strides = conv_stride, padding = conv_padding, activation = activation)
        self.bn = tfk.layers.BatchNormalization()
        self.max_pool = tfk.layers.MaxPool2D(max_pool_kernel, strides = pool_stride, padding = pool_padding)
        self.dropout = tfk.layers.Dropout(rate = dropout_rate)
        self.has_batch_norm = batch_norm
        self.has_pool = pool
        self.has_dropout = dropout

    def call(self, x, training = False):
        '''Performs a convolution, batch normalization, max pooling, and dropout.'''
        x = self.conv(x, training = training)
        if self.has_batch_norm:
            x = self.bn(x, training = training)
        if self.has_pool:
            x = self.max_pool(x, training = training)
        if self.has_dropout and training:
            x = self.dropout(x)
        return x

class FCBlock(layers.Layer):
    '''
    A layer with 2 fully connected layers and a dropout layer between them. \n
    Dense -> Dropout -> Dense
    '''
    def __init__(self, out_name: str, first_layer: bool = True, first_layer_units: int = 64, output_units: int = 19, dropout: bool = True,
                 dropout_rate: float = 0.5, hidden_activation: str = 'relu', output_activation: str = 'softmax'):
        '''
        Creates a layer with 2 fully connected layers and a dropout layer between them.
        :param out_name str: name of the output layer
        :param first_layer bool: whether this is the first layer or not. default True
        :param first_layer_units int: number of units in the first layer. default 64
        :param output_units int: number of units in the output layer. default 19
        :param dropout bool: whether to use dropout or not. default True
        :param dropout_rate float: dropout rate. default 0.5
        :param hidden_activation str: activation function of the hidden layer. default 'relu'
        :param output_activation str: activation function of the output layer. default 'softmax'
        '''
        super(FCBlock, self).__init__()
        self.dense1 = tfk.layers.Dense(first_layer_units, activation = hidden_activation) if first_layer else None
        self.dropout = tfk.layers.Dropout(dropout_rate) if dropout else None
        self.dense2 = tfk.layers.Dense(output_units, activation = output_activation)
        self._name = out_name
        self.first_layer = first_layer
        self.has_dropout = dropout

    def call(self, x, training = False):
        '''Performs a fully connected layer, a dropout layer, and another fully connected layer.'''
        if self.first_layer:
            x = self.dense1(x, training = training)
        if training and self.has_dropout:
            x = self.dropout(x)
        x = self.dense2(x, training = training)
        return x


class ResNetBlock(tfk.layers.Layer):
    '''
    A layer for representing a single block of ResNet50. \n
    It contains a shortcut connection that is added to the output of the main path. \n
    CONV 1*1 -> BN -> RELU -> CONV 3*3 -> BN -> RELU -> CONV 1*1 -> BN -> ADD -> RELU
    '''
    def __init__(self, kernel_size: Tuple[int, int], filters: List[int], reduce: bool, strides: Tuple[int, int] = (1, 1)):
        '''
        Creates a layer for representing a single block of ResNet50.
        :param kernel_size Tuple[int, int]: kernel size of the convolutional layers
        :param filters List[int]: number of filters in the convolutional layers
        :param reduce bool: whether to reduce the size of the input or not
        :param strides Tuple[int, int]: stride of the convolutional layers default (1,1)
        '''
        super(ResNetBlock, self).__init__()
        self.reduce = reduce
        if len(filters) != 3:
            raise ValueError('filters must be a list containing 3 integer numbers.')
        f1, f2, f3 = filters
        # shortcut path
        self.shortcut_conv = tfk.layers.Conv2D(f3, (1, 1), strides = strides, padding = 'valid') if reduce else None
        self.shortcut_bn = tfk.layers.BatchNormalization(axis = 3) if reduce else None
        self.shortcut_activation = tfk.layers.Activation('relu') if reduce else None
        # main path 1 * 1 - batch norm - relu
        self.main_conv1 = tfk.layers.Conv2D(f1, (1, 1), strides = strides, padding = 'valid')
        self.main_bn1 = tfk.layers.BatchNormalization(axis = 3)
        self.main_activation1 = tfk.layers.Activation('relu')
        # main path 3 * 3 - batch norm - relu
        self.main_conv2 = tfk.layers.Conv2D(f2, kernel_size, strides = (1, 1), padding = 'same')
        self.main_bn2 = tfk.layers.BatchNormalization(axis = 3)
        self.main_activation2 = tfk.layers.Activation('relu')
        # main path 1 * 1 - batch norm - relu
        self.main_conv3 = tfk.layers.Conv2D(f3, (1, 1), strides = (1, 1), padding = 'valid')
        self.main_bn3 = tfk.layers.BatchNormalization(axis = 3)
  
    def call(self, inputs, training = False):
        '''
        Performs a forward pass through the layer.
        '''
        shortcut_path = inputs
        # create shortcut filter if needed
        if self.reduce:
            shortcut_path = self.shortcut_conv(shortcut_path, training = training)
            shortcut_path = self.shortcut_bn(shortcut_path, training = training)
            shortcut_path = self.shortcut_activation(shortcut_path, training = training)
        # main path 1 * 1 - batch norm - relu
        main_path = self.main_conv1(inputs, training = training)
        main_path = self.main_bn1(main_path, training = training)
        main_path = self.main_activation1(main_path, training = training)
        # main path 3 * 3 - batch norm - relu
        main_path = self.main_conv2(main_path, training = training)
        main_path = self.main_bn2(main_path, training = training)
        main_path = self.main_activation2(main_path, training = training)
        # main path 1 * 1 - batch norm
        main_path = self.main_conv3(main_path, training = training)
        main_path = self.main_bn3(main_path, training = training)
        # add shortcut path to main path
        result = tfk.layers.Add()([main_path, shortcut_path])
        result = tfk.layers.Activation('relu')(result, training = training)
        return result

class BatchReluConv(tfk.layers.Layer):
    '''
    A layer containg a batch normalization, a relu activation, and a convolutional layer. \n
    It is used in dense block in DenseNet. \n
    BN -> RELU -> CONV
    '''
    def __init__(self, kernels: int, conv_kernel: Tuple[int, int]):
        super(BatchReluConv, self).__init__()
        self.bn = tfk.layers.BatchNormalization(axis = 3)
        self.relu = tfk.layers.Activation('relu')
        self.conv = tfk.layers.Conv2D(kernels, conv_kernel, strides = (1, 1), padding = 'same')

    def call(self, x, training = False):
        x = self.bn(x, training = training)
        x = self.relu(x, training = training)
        x = self.conv(x, training = training)
        return x

class Transition(tfk.layers.Layer):
    '''
    A layer for representing a transition block in DenseNet(DFCR Version). \n
    It is used to reduce the size of the input. \n
    BN -> ReLU -> CONV 1*1 -> AVGPOOL 2*2 \n
    '''
    def __init__(self, kernels: int, conv_kernel: Tuple[int, int] = (1, 1), conv_strides: Tuple[int, int] = (1, 1),
                 pool_kernel: Tuple[int, int] = (2, 2), pool_strides: Tuple[int, int] = (2, 2)):
        '''
        Creates a layer for representing a transition block in DenseNet.
        :param kernels int: number of kernels in the convolutional layer
        :param conv_kernel Tuple[int, int]: kernel size of the convolutional layer
        :param conv_strides Tuple[int, int]: stride of the convolutional layer
        :param pool_kernel Tuple[int, int]: kernel size of the average pooling layer
        :param pool_strides Tuple[int, int]: stride of the average pooling layer
        '''
        super(Transition, self).__init__()
        self.bn = tfk.layers.BatchNormalization(axis = 3)
        self.relu = tfk.layers.Activation('relu')
        self.conv = tfk.layers.Conv2D(kernels, conv_kernel, strides = conv_strides, padding = 'valid')
        self.pool = tfk.layers.MaxPool2D(pool_kernel, strides = pool_strides, padding = 'valid')
    
    def call(self, x, training = False):
        '''
        Performs a forward pass through the layer.
        '''
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv(x, training = training)
        x = self.pool(x, training = training)
        return x

class DFCRDenseBlock(tfk.layers.Layer):
    ''''
    Represents a dense block in DenseNet(DFCR Version).
    Every feature map in this block is connected to every other feature map in the same block.
    Each feature map is concatenated with its previous feature maps.
    between each two feature map we have:
    BN -> RELU -> CONV 1*1 -> BN -> RELU -> CONV 3*3
    '''
    def __init__(self, l: int, k: int):
        '''
        Creates a dense block in DenseNet(DFCR Version).
        :param l int: number of layers in the block
        :param k int: growth rate of the block. its the depth of the output feature maps.
        '''
        super(DFCRDenseBlock, self).__init__()
        self.l = l
        self.k = k
        # create layers
        self.brc_list = [(BatchReluConv(4 * k, (1, 1)), BatchReluConv(k, (3, 3))) for _ in range(l - 1)]

    def call(self, x, training = False):
        '''
        Performs a forward pass through the layer.
        '''
        last_layer = x
        last_concat = x
        # connect layers and concatenate them
        for i in range(self.l - 1):
            new_layer = self.brc_list[i][0](last_layer, training = training)
            new_layer = self.brc_list[i][1](new_layer, training = training)
            last_layer = new_layer
            last_concat = tfk.layers.Concatenate()([last_concat, new_layer])
        return last_concat
