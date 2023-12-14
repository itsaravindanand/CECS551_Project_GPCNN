import tensorflow
import tensorflow.keras.layers
import gpc.util
import math

class CopyChannels(tensorflow.keras.layers.Layer):
    """
    This layer copies channels from channel_start the number of channels given in channel_count.
    """

    def __init__(self,
                 channel_start=0,
                 channel_count=1,
                 **kwargs):
        self.channel_start = channel_start
        self.channel_count = channel_count
        super(CopyChannels, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[2], self.channel_count)

    def call(self, x):
        return x[:, :, :, self.channel_start:(self.channel_start + self.channel_count)]

    def get_config(self):
        config = {
            'channel_start': self.channel_start,
            'channel_count': self.channel_count
        }
        base_config = super(CopyChannels, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Negate(tensorflow.keras.layers.Layer):
    """
    This layer negates (multiplies by -1) the input tensor.
    """

    def __init__(self, **kwargs):
        super(Negate, self).__init__(**kwargs)
        self.trainable = False

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[2], input_shape[3])

    def call(self, x):
        return -x

    def get_config(self):
        # this is here to make the warning to disappear.
        base_config = super(Negate, self).get_config()
        return dict(list(base_config.items()))


class ConcatNegation(tensorflow.keras.layers.Layer):
    """
    This layer concatenates to the input its negation.
    """

    def __init__(self, **kwargs):
        super(ConcatNegation, self).__init__(**kwargs)
        self.trainable = False

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[2], input_shape[3] * 2)

    def call(self, x):
        # return np.concatenate((x, -x), axis=3)
        return tensorflow.keras.layers.Concatenate(axis=3)([x, -x])

    def get_config(self):
        # this is here to make the warning to disappear.
        base_config = super(ConcatNegation, self).get_config()
        return dict(list(base_config.items()))


class InterleaveChannels(tensorflow.keras.layers.Layer):
    """
    This layer interleaves channels stepping according to the number passed as parameter.
    """

    def __init__(self,
                 step_size=2,
                 **kwargs):
        if step_size < 2:
            self.step_size = 1
        else:
            self.step_size = step_size
        super(InterleaveChannels, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[2], input_shape[3])

    def call(self, x):
        return tensorflow.keras.layers.Concatenate(axis=3)(
            [x[:, :, :, shift_pos::self.step_size] for shift_pos in range(self.step_size)]
        )
        # for self.step_size == 2, we would have:
        #  return keras.layers.Concatenate(axis=3)([
        #    x[:, :, :, 0::self.step_size],
        #    x[:, :, :, 1::self.step_size]
        #    ])

    def get_config(self):
        config = {
            'step_size': self.step_size
        }
        base_config = super(InterleaveChannels, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class SumIntoHalfChannels(tensorflow.keras.layers.Layer):
    """
    This layer divides channels into 2 half's and then sums resulting in half of the input channels.
    """

    def __init__(self, **kwargs):
        super(SumIntoHalfChannels, self).__init__(**kwargs)
        self.trainable = False

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[2], input_shape[3] // 2)

    def call(self, x):
        outputchannels = x.shape[3] // 2
        return tensorflow.math.add(
            x=x[:, :, :, 0:outputchannels],
            y=x[:, :, :, outputchannels:outputchannels * 2]
        )

    def get_config(self):
        # this is here to make the warning to disappear.
        base_config = super(SumIntoHalfChannels, self).get_config()
        return dict(list(base_config.items()))


def GlobalAverageMaxPooling2D(previous_layer, name=None):
    """
    Adds both global Average and Max poolings. This layers is known to speed up training.
    """
    if name is None: name = 'global_pool'
    return tensorflow.keras.layers.Concatenate(axis=1)([
        tensorflow.keras.layers.GlobalAveragePooling2D(name=name + '_avg')(previous_layer),
        tensorflow.keras.layers.GlobalMaxPooling2D(name=name + '_max')(previous_layer)
    ])


def FitChannelCountTo(last_tensor, next_channel_count, has_interleaving=False, channel_axis=3):
    prev_layer_channel_count = tensorflow.keras.backend.int_shape(last_tensor)[channel_axis]
    full_copies = next_channel_count // prev_layer_channel_count
    extra_channels = next_channel_count % prev_layer_channel_count
    output_copies = []
    for copy_cnt in range(full_copies):
        if copy_cnt == 0:
            output_copies.append(last_tensor)
        else:
            if has_interleaving:
                output_copies.append(
                    InterleaveChannels(step_size=((copy_cnt + 1) % prev_layer_channel_count))(last_tensor))
            else:
                output_copies.append(last_tensor)
    if (extra_channels > 0):
        if has_interleaving:
            extra_tensor = InterleaveChannels(step_size=((full_copies + 1) % prev_layer_channel_count))(last_tensor)
        else:
            extra_tensor = last_tensor
        output_copies.append(CopyChannels(0, extra_channels)(extra_tensor))
    last_tensor = tensorflow.keras.layers.Concatenate(axis=channel_axis)(output_copies)
    return last_tensor


def EnforceEvenChannelCount(last_tensor, channel_axis=3):
    prev_layer_channel_count = tensorflow.keras.backend.int_shape(last_tensor)[channel_axis]
    if (prev_layer_channel_count % 2 > 0):
        last_tensor = FitChannelCountTo(
            last_tensor,
            next_channel_count=prev_layer_channel_count + 1,
            channel_axis=channel_axis)
    return last_tensor


def BinaryConvLayers(last_tensor, name, shape=(3, 3), conv_count=1, has_batch_norm=True, has_interleaving=False,
                     activation='relu', channel_axis=3):
    last_tensor = EnforceEvenChannelCount(last_tensor)
    prev_layer_channel_count = tensorflow.keras.backend.int_shape(last_tensor)[channel_axis]
    for conv_cnt in range(conv_count):
        input_tensor = last_tensor
        if has_interleaving:
            last_tensor_interleaved = InterleaveChannels(step_size=2, name=name + "_i_" + str(conv_cnt))(last_tensor)
        else:
            last_tensor_interleaved = last_tensor
        x1 = tensorflow.keras.layers.Conv2D(prev_layer_channel_count // 2, shape, padding='same', activation=None,
                                            name=name + "_a_" + str(conv_cnt), groups=prev_layer_channel_count // 2)(
            last_tensor)
        x2 = tensorflow.keras.layers.Conv2D(prev_layer_channel_count // 2, shape, padding='same', activation=None,
                                            name=name + "_b_" + str(conv_cnt), groups=prev_layer_channel_count // 2)(
            last_tensor_interleaved)
        last_tensor = tensorflow.keras.layers.Concatenate(axis=channel_axis, name=name + "_conc_" + str(conv_cnt))(
            [x1, x2])
        if has_batch_norm: last_tensor = tensorflow.keras.layers.BatchNormalization(axis=channel_axis,
                                                                                    name=name + "_batch_" + str(
                                                                                        conv_cnt))(last_tensor)
        if activation is not None: last_tensor = tensorflow.keras.layers.Activation(activation=activation,
                                                                                    name=name + "_act_" + str(
                                                                                        conv_cnt))(last_tensor)
        from_highway = tensorflow.keras.layers.DepthwiseConv2D(1,  # kernel_size
                                                               strides=1,
                                                               padding='valid',
                                                               use_bias=False,
                                                               name=name + '_depth_' + str(conv_cnt))(input_tensor)
        last_tensor = tensorflow.keras.layers.add([from_highway, last_tensor], name=name + '_add' + str(conv_cnt))
        if has_batch_norm: last_tensor = tensorflow.keras.layers.BatchNormalization(axis=channel_axis)(last_tensor)
    return last_tensor


def BinaryPointwiseConvLayers(last_tensor, name, conv_count=1, has_batch_norm=True, has_interleaving=False,
                              activation='relu', channel_axis=3):
    return BinaryConvLayers(last_tensor, name, shape=(1, 1), conv_count=conv_count, has_batch_norm=has_batch_norm,
                            has_interleaving=has_interleaving, activation=activation, channel_axis=channel_axis)


def BinaryCompressionLayer(last_tensor, name, has_batch_norm=True, activation='relu', channel_axis=3):
    last_tensor = EnforceEvenChannelCount(last_tensor)
    prev_layer_channel_count = tensorflow.keras.backend.int_shape(last_tensor)[channel_axis]
    last_tensor = tensorflow.keras.layers.Conv2D(prev_layer_channel_count // 2, (1, 1), padding='same', activation=None,
                                                 name=name + "_conv", groups=prev_layer_channel_count // 2)(last_tensor)
    if has_batch_norm: last_tensor = tensorflow.keras.layers.BatchNormalization(axis=channel_axis,
                                                                                name=name + "_batch")(last_tensor)
    if activation is not None: last_tensor = tensorflow.keras.layers.Activation(activation=activation,
                                                                                name=name + "_act")(last_tensor)
    return last_tensor


def BinaryCompression(last_tensor, name, target_channel_count, has_batch_norm=True, activation='relu', channel_axis=3):
    prev_layer_channel_count = tensorflow.keras.backend.int_shape(last_tensor)[channel_axis]
    cnt = 0
    while (prev_layer_channel_count >= target_channel_count * 2):
        last_tensor = BinaryCompressionLayer(last_tensor, name=name + '_' + str(cnt), has_batch_norm=has_batch_norm,
                                             activation=activation, channel_axis=channel_axis)
        prev_layer_channel_count = tensorflow.keras.backend.int_shape(last_tensor)[channel_axis]
        cnt = cnt + 1
    if prev_layer_channel_count > target_channel_count:
        last_tensor = FitChannelCountTo(last_tensor, next_channel_count=target_channel_count * 2,
                                        channel_axis=channel_axis)
        last_tensor = BinaryCompressionLayer(last_tensor, name=name + '_' + str(cnt), has_batch_norm=has_batch_norm,
                                             activation=activation, channel_axis=channel_axis)
    return last_tensor


def GetChannelAxis():
    if tensorflow.keras.backend.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = 3
    return channel_axis


def conv2d_bn(x,
              filters,
              num_row,
              num_col,
              padding='same',
              strides=(1, 1),
              name=None,
              use_bias=False,
              activation='relu',
              has_batch_norm=True,
              has_batch_scale=False,
              groups=0
              ):
    """Utility function to apply convolution, batch norm and activation function.

    # Arguments
        x: input tensor.
        filters: filters in `Conv2D`.
        num_row: height of the convolution kernel.
        num_col: width of the convolution kernel.
        padding: padding mode in `Conv2D`.
        strides: strides in `Conv2D`.
        name: name of the ops; will become `name + '_conv'`
            for the convolution and `name + '_bn'` for the
            batch norm layer.
        use_bias: True means that bias will be added,
        activation: activation function. None means no activation function.
        has_batch_norm: True means that batch normalization is added.
        has_batch_scale: True means that scaling is added to batch norm.
        groups: number of groups in the convolution

    # Returns
        Output tensor after applying `Conv2D` and `BatchNormalization`.
    """
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None
    if tensorflow.keras.backend.image_data_format() == 'channels_first':
        bn_axis = 1
    else:
        bn_axis = 3

    # groups parameter isn't available in older tensorflow implementations
    if (groups > 1):
        x = tensorflow.keras.layers.Conv2D(
            filters, (num_row, num_col),
            strides=strides,
            padding=padding,
            use_bias=use_bias,
            groups=groups,
            name=conv_name)(x)
    else:
        x = tensorflow.keras.layers.Conv2D(
            filters, (num_row, num_col),
            strides=strides,
            padding=padding,
            use_bias=use_bias,
            name=conv_name)(x)

    if (has_batch_norm): x = tensorflow.keras.layers.BatchNormalization(axis=bn_axis, scale=has_batch_scale,
                                                                        name=bn_name)(x)
    if activation is not None: x = tensorflow.keras.layers.Activation(activation=activation, name=name)(x)
    return x


def HardSigmoid(x):
    return tensorflow.keras.layers.ReLU(6.0)(x + 3.0) * (1.0 / 6.0)


def HardSwish(x):
    return tensorflow.keras.layers.Multiply()([tensorflow.keras.layers.Activation(HardSigmoid)(x), x])

def gpcConv2DType2(last_tensor,  filters=32,  channel_axis=3,  name=None, activation=None, has_batch_norm=True, has_batch_scale=True, use_bias=True, kernel_size=1, stride_size=1, padding='same', min_channels_per_group=16):    
    output_tensor = last_tensor
    prev_layer_channel_count = tensorflow.keras.backend.int_shape(last_tensor)[channel_axis]
    output_channel_count = filters
    max_acceptable_divisor = (prev_layer_channel_count//min_channels_per_group)
    group_count = gpc.util.get_max_acceptable_common_divisor(prev_layer_channel_count, output_channel_count, max_acceptable = max_acceptable_divisor)
    if group_count is None: group_count=1
    output_group_size = output_channel_count // group_count    
    if (group_count>1):        
        output_tensor = conv2d_bn(output_tensor, output_channel_count, kernel_size, kernel_size, name=name, activation=activation, has_batch_norm=has_batch_norm, has_batch_scale=has_batch_scale, groups=group_count, use_bias=use_bias, strides=(stride_size, stride_size), padding=padding)
        compression_tensor = output_tensor
        if output_group_size > 1:
            output_tensor = InterleaveChannels(output_group_size, name=name+'_group_interleaved')(output_tensor)
        if (prev_layer_channel_count >= output_channel_count):            
            output_tensor = conv2d_bn(output_tensor, output_channel_count, 1, 1, name=name+'_group_interconnect', activation=activation, has_batch_norm=has_batch_norm, has_batch_scale=has_batch_scale, groups=group_count, use_bias=use_bias)
            output_tensor = tensorflow.keras.layers.add([output_tensor, compression_tensor], name=name+'_inter_group_add')
    else:        
        output_tensor = conv2d_bn(output_tensor, output_channel_count, kernel_size, kernel_size, name=name, activation=activation, has_batch_norm=has_batch_norm, has_batch_scale=has_batch_scale, use_bias=use_bias)
    return output_tensor


def gpcConv2D(last_tensor, filters=32, channel_axis=3, name=None, activation=None, has_batch_norm=True, has_batch_scale=True, use_bias=True, kernel_size=1, stride_size=1, padding='same', gpcType=0):
    prev_layer_channel_count = tensorflow.keras.backend.int_shape(last_tensor)[channel_axis]
    if gpcType == 1:
        #16 Channels
        return gpcConv2DType2(last_tensor, filters=filters, channel_axis=channel_axis, name=name, activation=activation, has_batch_norm=has_batch_norm, has_batch_scale=has_batch_scale, use_bias=use_bias, kernel_size=kernel_size, stride_size=stride_size, padding=padding, min_channels_per_group=16)
    elif gpcType == 2:
        #32 Channels
        return gpcConv2DType2(last_tensor, filters=filters, channel_axis=channel_axis, name=name, activation=activation, has_batch_norm=has_batch_norm, has_batch_scale=has_batch_scale, use_bias=use_bias, kernel_size=kernel_size, stride_size=stride_size, padding=padding, min_channels_per_group=32)
    elif gpcType == 3:
        #64 Channels
        return gpcConv2DType2(last_tensor, filters=filters, channel_axis=channel_axis, name=name, activation=activation, has_batch_norm=has_batch_norm, has_batch_scale=has_batch_scale, use_bias=use_bias, kernel_size=kernel_size, stride_size=stride_size, padding=padding, min_channels_per_group=64)
    elif gpcType == 4:
        #128 Channels
        return gpcConv2DType2(last_tensor, filters=filters, channel_axis=channel_axis, name=name, activation=activation, has_batch_norm=has_batch_norm, has_batch_scale=has_batch_scale, use_bias=use_bias, kernel_size=kernel_size, stride_size=stride_size, padding=padding, min_channels_per_group=128)


def gpcPointwiseConv2D(last_tensor, filters=32, channel_axis=3, name=None, activation=None, has_batch_norm=True, has_batch_scale=True, use_bias=True, gpcType=0):
    return gpcConv2D(last_tensor, filters=filters, channel_axis=channel_axis, name=name, activation=activation, has_batch_norm=has_batch_norm, has_batch_scale=has_batch_scale, use_bias=use_bias, kernel_size=1, stride_size=1, padding='same', gpcType=gpcType)

def GetClasses():
    return {
        'CopyChannels': CopyChannels,
        'Negate': Negate,
        'ConcatNegation': ConcatNegation,
        'InterleaveChannels': InterleaveChannels,
        'SumIntoHalfChannels': SumIntoHalfChannels,
        'HardSigmoid': HardSigmoid,
        'HardSwish': HardSwish,
        'hard_sigmoid': HardSigmoid,
        'hard_swish': HardSwish
    }
