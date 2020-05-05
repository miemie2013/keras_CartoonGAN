import keras
import tensorflow as tf
from keras import backend as K
import keras.layers as layers
from keras.engine.topology import Layer

def gram(x):
    shape_x = tf.shape(x)
    b = shape_x[0]
    c = shape_x[3]
    x = tf.reshape(x, [b, -1, c])
    x_t = tf.transpose(x, [0, 2, 1])
    # 遍历这一批里的每个样本，单独计算Gram矩阵
    def _process_sample(args):
        _x_t, _x = args
        _r = tf.matmul(_x_t, _x)
        _gramm = K.stack([_r], -1)
        return _gramm
    gramm = K.map_fn(_process_sample, [x_t, x], dtype=K.floatx())
    gramm = tf.reshape(gramm, [b, c, c])
    hwc = tf.size(x) // b
    return gramm / tf.cast(hwc, tf.float32)

def generator_adversarial_loss(y_true, y_pred):
    '''
    只是二值交叉熵损失
    '''
    return K.mean(K.binary_crossentropy(y_true, y_pred, from_logits=True), axis=-1)

def style_loss(y_true, y_pred):
    '''
    先gram()处理再mae损失
    y_true是动漫图片imgs_B经过vggl的输出imgs_B_style。 y_pred是真实世界图片imgs_A经过G和vggl的输出feature_fake_B
    标记来自领域B，神经网络的预测输出来自领域A。抽取的图片看似八竿子打不着（非成对训练风格转换），如何度量两个领域的风格损失？

    假设y_true.shape = (2, 2, 1, 3),   y_pred.shape = (2, 2, 1, 3)
    即批大小为2，特征图的h=2, w=1, c=3, 即每张图都只有两个像素。先压平行和列，使之变成
    y_true.shape = (2, 2, 3),   y_pred.shape = (2, 2, 3)
    即批大小为2，每张特征图的2个像素, c=3。
    设 A=y_true[0], 则A.shape = (2, 3)
    不妨设  A = |3 2 5|
               |1 8 0|
      则  A^T = |3 1|
                |2 8|
                |5 0|
    计算 B = A^T*A = |10 14 15|
                     |14 68 10|
                     |15 10 25|
        第i行第j列表示 通道i上全部像素的值 和 通道j上相同位置像素的值 进行相乘再求和。
        由于有3个通道，所以结果有3^2=9个
        而且B是对称的矩阵。
        a = [[[[3, 2, 5]], [[1, 8, 0]]], [[[31, 25, 53]], [[12, 83, 20]]]]
        train_ys = np.array(a)
    '''
    y_true = gram(y_true)
    y_pred = gram(y_pred)
    return K.mean(K.abs(y_pred - y_true), axis=-1)

class InstanceNormalization(Layer):
    """InstanceNormalization，输出图片的格式是(m, h, w, c)
    """
    def __init__(self, epsilon=1e-9, **kwargs):
        super(InstanceNormalization, self).__init__(**kwargs)
        self.epsilon = epsilon

    def build(self, input_shape):
        super(InstanceNormalization, self).build(input_shape)
        shape = (input_shape[-1], )
        self.gamma = self.add_weight(shape=shape,
                                     initializer='ones',
                                     name='gamma')
        self.beta = self.add_weight(shape=shape,
                                    initializer='zeros',
                                    name='beta')

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, x):
        m, h, w, c = K.shape(x)[0], K.shape(x)[1], K.shape(x)[2], K.shape(x)[3]
        x_reshape = K.reshape(x, (m, h*w, c))
        mean = K.mean(x_reshape, axis=1, keepdims=True)
        t = K.square(x_reshape - mean)
        variance = K.mean(t, axis=1, keepdims=True)
        std = K.sqrt(variance + self.epsilon)
        outputs = (x_reshape - mean) / std
        outputs = outputs*self.gamma + self.beta
        outputs = K.reshape(outputs, (m, h, w, c))
        return outputs

class ReflectionPadding2D(Layer):
    def __init__(self, padding=(1, 1), **kwargs):
        super(ReflectionPadding2D, self).__init__(**kwargs)
        padding = tuple(padding)
        self.padding = ((0, 0), padding, padding, (0, 0))
        self.input_spec = [layers.InputSpec(ndim=4)]

    def compute_output_shape(self, input_shape):
        if input_shape[1] is not None:
            aa = (input_shape[0], input_shape[1]+self.padding[1][0]++self.padding[1][1], input_shape[2]+self.padding[2][0]++self.padding[2][1], input_shape[3])
            return aa
        return input_shape

    def call(self, x):
        return tf.pad(x, self.padding, "REFLECT")

def conv2d_unit(x, filters, kernel_size, strides=1, padding='same', name=None):
    x = layers.Conv2D(filters, kernel_size,
               padding=padding,
               strides=strides,
               use_bias=True,
               activation='linear',
               name=name,
               kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.01))(x)
    return x

def get_generator(initial_filters=64):
    # inputs = layers.Input(shape=(256, 256, 3), name='generator_input')
    inputs = layers.Input(shape=(None, None, 3), name='generator_input')

    i64 = initial_filters
    i128 = i64 * 2
    i256 = i64 * 4

    x = ReflectionPadding2D((3, 3))(inputs)
    x = conv2d_unit(x, i64, 7, padding='valid', name='conv01_1')
    x = InstanceNormalization(name='in01_1')(x)
    x = layers.ReLU()(x)


    '''
    pytorch的模型移植到keras。 
    kernel_size=3, stride=2, padding=1  -->  前面要加 ZeroPadding2D(1)，且kernel_size=3, strides=2, padding='valid'
    kernel_size=3, stride=1, padding=1  -->  不用加 ZeroPadding2D(1)，  且kernel_size=3, strides=1, padding='same'
    '''
    x = layers.ZeroPadding2D(1)(x)
    x = conv2d_unit(x, i128, 3, strides=2, padding='valid', name='conv02_1')
    x = conv2d_unit(x, i128, 3, strides=1, padding='same', name='conv02_2')
    x = InstanceNormalization(name='in02_1')(x)
    x = layers.ReLU()(x)

    x = layers.ZeroPadding2D(1)(x)
    x = conv2d_unit(x, i256, 3, strides=2, padding='valid', name='conv03_1')
    x = conv2d_unit(x, i256, 3, strides=1, padding='same', name='conv03_2')
    x = InstanceNormalization(name='in03_1')(x)
    x = layers.ReLU()(x)

    # 残差部分
    for i in range(8):
        shortcut = x
        x = ReflectionPadding2D((1, 1))(x)
        x = conv2d_unit(x, i256, 3, padding='valid', name='conv%.2d_1' % (4+i))
        x = InstanceNormalization(name='in%.2d_1' % (4+i))(x)
        x = layers.ReLU()(x)
        x = ReflectionPadding2D((1, 1))(x)
        x = conv2d_unit(x, i256, 3, padding='valid', name='conv%.2d_2' % (4+i))
        x = InstanceNormalization(name='in%.2d_2' % (4+i))(x)
        x = layers.add([x, shortcut])

    # 上采样
    x = layers.Conv2DTranspose(i128, 3, strides=2, padding='same', output_padding=1, name='deconv01_1')(x)
    x = conv2d_unit(x, i128, 3, strides=1, padding='same', name='deconv01_2')
    x = InstanceNormalization(name='in12_1')(x)
    x = layers.ReLU()(x)

    x = layers.Conv2DTranspose(i64, 3, strides=2, padding='same', output_padding=1, name='deconv02_1')(x)
    x = conv2d_unit(x, i64, 3, strides=1, padding='same', name='deconv02_2')
    x = InstanceNormalization(name='in13_1')(x)
    x = layers.ReLU()(x)

    x = ReflectionPadding2D((3, 3))(x)
    x = conv2d_unit(x, 3, 7, padding='valid', name='deconv03_1')
    x = layers.Activation('tanh', name='generator_output')(x)

    model = keras.models.Model(inputs=inputs, outputs=x, name='generator')
    return model


def get_discriminator(initial_filters=32):
    # inputs = layers.Input(shape=(256, 256, 3), name='discriminator_input')
    inputs = layers.Input(shape=(None, None, 3), name='discriminator_input')

    i32 = initial_filters
    i64 = i32 * 2
    i128 = i32 * 4
    i256 = i32 * 8

    x = conv2d_unit(inputs, i32, 3, strides=1, padding='same')
    x = keras.layers.advanced_activations.LeakyReLU(alpha=0.2)(x)


    # 2倍下采样
    x = layers.ZeroPadding2D(1)(x)
    x = conv2d_unit(x, i64, 3, strides=2, padding='valid')
    x = keras.layers.advanced_activations.LeakyReLU(alpha=0.2)(x)
    x = conv2d_unit(x, i128, 3, strides=1, padding='same')
    x = InstanceNormalization()(x)
    x = keras.layers.advanced_activations.LeakyReLU(alpha=0.2)(x)


    # 4倍下采样
    x = layers.ZeroPadding2D(1)(x)
    x = conv2d_unit(x, i128, 3, strides=2, padding='valid')
    x = keras.layers.advanced_activations.LeakyReLU(alpha=0.2)(x)
    x = conv2d_unit(x, i256, 3, strides=1, padding='same')
    x = InstanceNormalization()(x)
    x = keras.layers.advanced_activations.LeakyReLU(alpha=0.2)(x)
    x = conv2d_unit(x, i256, 3, strides=1, padding='same')
    x = InstanceNormalization()(x)
    x = keras.layers.advanced_activations.LeakyReLU(alpha=0.2)(x)

    # 最后的卷积层。tf版本里，最后的激活是线性激活，损失函数是mse
    x = conv2d_unit(x, 1, 3, strides=1, padding='same', name='discriminator_output')
    # x = layers.Activation('sigmoid', name='discriminator_output')(x)

    model = keras.models.Model(inputs=inputs, outputs=x, name='discriminator')
    return model

def get_vggl():
    img_input = layers.Input(shape=(None, None, 3))

    # Block 1
    x = layers.Conv2D(64, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block1_conv1')(img_input)
    x = layers.Conv2D(64, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block1_conv2')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = layers.Conv2D(128, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block2_conv1')(x)
    x = layers.Conv2D(128, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block2_conv2')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv1')(x)
    x = layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv2')(x)
    x = layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv3')(x)
    x = layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv4')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv1')(x)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv2')(x)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv3')(x)
    x = layers.Conv2D(512, (3, 3),
                      activation='linear',
                      padding='same',
                      name='block4_conv4')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)
    # block4_conv4的relu激活变成线性激活

    vggl = keras.models.Model(inputs=img_input, outputs=x, name='vggl')
    return vggl

