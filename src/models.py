from layers import ConvPoolBlock, FCBlock, ResNetBlock, Transition, DFCRDenseBlock
import tensorflow.keras as tfk
from typing import Tuple, List

class SimpleModelBuilder:
    '''
    A simple model with 4 convolutional layers and 5 branches with 2 fully connected layers each. \n
    8 CONV 3*3 -> 16 CONV 3*3 -> 32 CONV 3*3 -> 64 CONV 3*3 \n
    A dropout layer is added after each convolutional layer. \n
    5 branches: \n
    128 Dense -> 19 Dense \n
    128 Dense -> 19 Dense \n
    128 Dense -> 19 Dense \n
    128 Dense -> 19 Dense \n
    128 Dense -> 19 Dense \n
    '''
    def __init__(self) -> None:
        '''
        Creates a simple model with 4 convolutional layers and 5 branches with 2 fully connected layers each.
        '''
        # create the convolutional layers
        self.conv_pool1 = ConvPoolBlock(conv_filters = 8, dropout=True, dropout_rate = 0.1)
        self.conv_pool2 = ConvPoolBlock(conv_filters = 16, dropout=True, dropout_rate = 0.1)
        self.conv_pool3 = ConvPoolBlock(conv_filters = 32, dropout=True, dropout_rate = 0.1)
        self.conv_pool4 = ConvPoolBlock(conv_filters = 64, dropout=True, dropout_rate = 0.1)
        # create the flatten layer
        self.flatten_layer = tfk.layers.Flatten()
        # create the branches
        self.branches = [FCBlock(f'out{i}', first_layer_units=128) for i in range(5)]
        self.outputs = []
    
    def get_model(self, input_shape: Tuple[int, int, int] = (50, 200, 1)) -> tfk.Model:
        '''
        Creates a simple model with 4 convolutional layers and 5 branches with 2 fully connected layers each.
        :param input_shape Tuple[int, int, int]: shape of the input. default (50, 200, 1)
        :return tfk.Model: the model
        '''
        inputs = tfk.Input(shape=input_shape)  
        x = self.conv_pool1(inputs)
        x = self.conv_pool2(x)
        x = self.conv_pool3(x)
        x = self.conv_pool4(x)
        x = self.flatten_layer(x)
        for branch in self.branches:
            self.outputs.append(branch(x))
        return tfk.Model(inputs = inputs, outputs = self.outputs, name = 'model')


class ResNetBuilder:
    '''
    A class to represent ResNet50 model. \n
    It contains 16 ResNet blocks. \n
    more details on : https://iq.opengenus.org/resnet50-architecture/
    '''
    def __init__(self):
        '''Creates a ResNet50 model.'''
        # layers we need:
        # before ResNet blocks
        self.conv7_7 = tfk.layers.Conv2D(64, (7,7), strides = (2, 2))
        self.bn1 = tfk.layers.BatchNormalization(axis = 3)
        self.activation1 = tfk.layers.Activation('relu')
        self.max_pool = tfk.layers.MaxPool2D(pool_size = (3,3), strides = (2, 2))
        # first blocks
        self.res_block11 = ResNetBlock((3,3), [64, 64, 256], True, strides = (1, 1))
        self.res_block12 = ResNetBlock((3,3), [64, 64, 256], False)
        self.res_block13 = ResNetBlock((3,3), [64, 64, 256], False)
        # second blocks
        self.res_block21 = ResNetBlock((3,3), [128, 128, 512], True, strides = (2, 2))
        self.res_block22 = ResNetBlock((3,3), [128, 128, 512], False)
        self.res_block23 = ResNetBlock((3,3), [128, 128, 512], False)
        self.res_block24 = ResNetBlock((3,3), [128, 128, 512], False)
        # third blocks
        self.res_block31 = ResNetBlock((3,3), [256, 256, 1024], True, strides = (2, 2))
        self.res_block32 = ResNetBlock((3,3), [256, 256, 1024], False)
        self.res_block33 = ResNetBlock((3,3), [256, 256, 1024], False)
        self.res_block34 = ResNetBlock((3,3), [256, 256, 1024], False)
        self.res_block35 = ResNetBlock((3,3), [256, 256, 1024], False)
        self.res_block36 = ResNetBlock((3,3), [256, 256, 1024], False)
        # fourth blocks
        self.res_block41 = ResNetBlock((3,3), [512, 512, 2048], True, strides = (2, 2))
        self.res_block42 = ResNetBlock((3,3), [512, 512, 2048], False)
        self.res_block43 = ResNetBlock((3,3), [512, 512, 2048], False)
        # averege pool
        self.avg_pool = tfk.layers.AveragePooling2D((1, 1))
        # flatten
        self.flatten = tfk.layers.Flatten()
        self.branches = [FCBlock(f'out{i}', first_layer_units = 128) for i in range(5)]
    
    def get_model(self, input_shape: Tuple[int, int, int] = (50, 200, 1)) -> tfk.Model:
        '''
        Creates a ResNet50 model. \n
        :param input_shape Tuple[int, int, int]: shape of the input. default (50, 200, 1) \n
        :return tfk.Model: the model
        '''
        inputs = tfk.Input(shape=input_shape)  
        # before ResNet blocks
        x = self.conv7_7(inputs)
        x = self.bn1(x)
        x = self.activation1(x)
        x = self.max_pool(x)
        # first blocks
        x = self.res_block11(x)
        x = self.res_block12(x)
        x = self.res_block13(x)
        # second blocks
        x = self.res_block21(x)
        x = self.res_block22(x)
        x = self.res_block23(x)
        x = self.res_block24(x)
        # third blocks
        x = self.res_block31(x)
        x = self.res_block32(x)
        x = self.res_block33(x)
        x = self.res_block34(x)
        x = self.res_block35(x)
        x = self.res_block36(x)
        # fourth blocks
        x = self.res_block41(x)
        x = self.res_block42(x)
        x = self.res_block43(x)
        # avg pooling
        x = self.avg_pool(x)
        # flatten
        x = self.flatten(x)
        # output branches
        outputs = []
        for branch in self.branches:
            outputs.append(branch(x))
        return tfk.Model(inputs = inputs, outputs = outputs, name = 'ResNetModel')

class DFCRBuilder:
    '''
    A class to represent DFCR model. \n
    It consist of 4 dense blocks and 5 branches with 2 fully connected layers each. \n
    more details on DenseNet: https://arxiv.org/pdf/1608.06993.pdf
    more details on DFCR: https://www.aimspress.com/fileOther/PDF/MBE/mbe-16-05-292.pdf
    '''
    def __init__(self, k: int = 32, theta: float = 0.5, block_sizes: List[int] = [6, 6, 24, 16]):
        '''
        Creates a DFCR model. \n
        :param k int: growth rate. default 32 \n
        :param theta float: compression factor. default 0.5 \n
        :param block_sizes List[int]: number of layers in each dense block. default [6, 6, 24, 16]
        '''
        # initial 7*7 conv with strides (2, 2) and 3*3 max pool with strides (2, 2)
        self.init_pad1 = tfk.layers.ZeroPadding2D(padding = (3, 3))
        self.init_conv = tfk.layers.Conv2D(2 * k, (7, 7), strides = (2, 2), padding = 'valid')
        self.init_pad2 = tfk.layers.ZeroPadding2D(padding = (1, 1))
        self.init_pool = tfk.layers.MaxPooling2D(pool_size = (3, 3), strides = (2, 2), padding = 'valid')
        current_channels = 2 * k
        # dense1 + 1 * 1 conv + 2*2 average pool with strides (2, 2)
        self.dense1 = DFCRDenseBlock(block_sizes[0], k)
        current_channels = current_channels + (block_sizes[0] - 1) * k
        self.transition1 = Transition(int(current_channels * theta))
        current_channels = int(current_channels * theta)
        # dense2 + 1 * 1 conv + 2*2 average pool with strides (2, 2)
        self.dense2 = DFCRDenseBlock(block_sizes[1], k)
        current_channels = current_channels + (block_sizes[1] - 1) * k
        self.transition2 = Transition(int(current_channels * theta))
        current_channels = int(current_channels * theta)
        # dense3 + 1 * 1 conv + 2*2 average pool with strides (2, 2)
        self.dense3 = DFCRDenseBlock(block_sizes[2], k)
        current_channels = current_channels + (block_sizes[2] - 1) * k
        self.transition3 = Transition(int(current_channels * theta))
        current_channels = int(current_channels * theta)
        # dense4
        self.dense4 = DFCRDenseBlock(block_sizes[3], k)
        # global average pooling
        self.global_pool = tfk.layers.GlobalAveragePooling2D()
        # output branches
        self.branches = [FCBlock(f'out{i}', first_layer = True, dropout = True, first_layer_units = 128) for i in range(5)]

    def get_model(self, input_shape = (64, 256, 1)):
        '''
        Creates a DFCR model. \n
        :param input_shape Tuple[int, int, int]: shape of the input. default (64, 256, 1) \n
        :return tfk.Model: the model
        '''
        inputs = tfk.Input(input_shape)  
        x = self.init_pad1(inputs)
        x = self.init_conv(x)
        x = self.init_pad2(x)
        x = self.init_pool(x)
        x = self.dense1(x)
        x = self.transition1(x)
        x = self.dense2(x)
        x = self.transition2(x)
        x = self.dense3(x)
        x = self.transition3(x)
        x = self.dense4(x)
        x = self.global_pool(x)
        # output branches
        outputs = []
        for branch in self.branches:
          outputs.append(branch(x))
        return tfk.Model(inputs = inputs, outputs = outputs, name = 'DFCRModel')
    