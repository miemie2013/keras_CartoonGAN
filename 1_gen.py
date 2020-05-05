
import os

import cv2
import numpy as np
import tensorflow as tf
from models import get_generator

# 显存分配，不使用gpu
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 1.0
set_session(tf.Session(config=config))


test_img_path = os.listdir('test_img/')
if not os.path.exists('keras_output/'): os.mkdir('keras_output/')

# CartoonGan输入图片的颜色格式是bgr，像素值被转换到(-1, 1)之间。输出的图片也是bgr格式，像素值在(-1, 1)之间。
# 不能直接 G = keras.models.load_model('Shinkai.h5', ...)  这样做的话 fake_B = G.predict(imgs_A)   fake_B的大小会变成248x248(比原图大小256x256小)，不知道怎么回事。
G = get_generator()
G.load_weights('Shinkai.h5', by_name=True)
# G.load_weights('Hayao.h5', by_name=True)
# G.load_weights('Hosoda.h5', by_name=True)
# G.load_weights('Paprika.h5', by_name=True)


for p in test_img_path:
    ss = p.split('.')
    houzhui = ss[-1]
    l = len(houzhui)
    img_name = p[:len(p)-l-1]

    image_data = []
    image_bgr = cv2.imread('test_img/'+p)
    image_data.append(image_bgr)
    image_data = np.array(image_data)
    image_data = image_data / 127.5 - 1.0
    fake_B = G.predict(image_data)

    # [-1, 1] --> [0, 255]
    fake_B = (fake_B + 1.0) * 127.5
    fake_B = fake_B.astype(np.uint8)
    fake_B = fake_B[0]
    cv2.imwrite('keras_output/%s_Shinkai.jpg' % img_name, fake_B)
    # cv2.imwrite('keras_output/%s.jpg' % img_name, fake_B)

    gray = cv2.cvtColor(fake_B, cv2.COLOR_BGR2GRAY)  # gray是numpy数组，shape是(50, 100)，已经转换成灰度。
    gray = gray.reshape((fake_B.shape[0], fake_B.shape[1], 1))  # gray是numpy数组，shape是(50, 100, 1)
    # cv2.imwrite('keras_output/%s_gray.jpg' % img_name, gray)


