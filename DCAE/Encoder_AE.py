import tensorflow as tf
import pickle
from tensorflow import keras
from keras import layers
from tensorflow.python.framework import ops
import numpy as np
# Global Settings
batch_size=2 #batch_size=2
num_epochs=1000
learning_rate=1e-5 #learning_rate=1e-52e-4
img_size=1024
img_channels=1
def leaky_relu(x, alpha=0.2):
    return tf.maximum(x, alpha * x)

faults=['A',  'B-1','B-2','D-1','D-2','E-1','E-2',
        'J-2','K-1']
# faults=['A']
load_root = 'E:\\论文2\\谌洪梦\\代码\\SHM\\Data_Chopper\\2_S2L2P0_train_test\\S2L2P0_40+291_'
for fault in faults:
    # Build graph
    ops.reset_default_graph()
    # Re-Build model(970)
    inputs_ = layers.Input(shape=(img_size, img_channels), name="image_input")
    layers = tf.keras.layers
    conv1 = layers.Conv1D(filters=64, kernel_size=5, padding='same', activation=leaky_relu)(inputs_)
    maxpool1 = layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(conv1)
    conv2 = layers.Conv1D(filters=32, kernel_size=5, padding='same', activation=leaky_relu)(maxpool1)
    maxpool2 = layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(conv2)
    conv3 = layers.Conv1D(filters=16, kernel_size=5, padding='same', activation=leaky_relu)(maxpool2)
    maxpool3 = layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(conv3)
    conv4 = layers.Conv1D(filters=8, kernel_size=5, padding='same', activation=leaky_relu)(maxpool3)
    maxpool4 = layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(conv4)
    conv5 = layers.Conv1D(filters=4, kernel_size=5, padding='same', activation=leaky_relu)(maxpool4)
    maxpool5 = layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(conv5)
    re = tf.reshape(maxpool5, [-1, 128])

    latent = layers.Dense(units=128,activation=None)(re)

    x1 = layers.Dense(units=128, activation=tf.nn.relu)(re)
    x = tf.reshape(x1, [-1, 32, 4])
    x = layers.UpSampling1D(2)(x)
    x = layers.Conv1D(filters=8, kernel_size=5, padding='same', activation=leaky_relu)(x)
    x = layers.UpSampling1D(2)(x)
    x = layers.Conv1D(filters=16, kernel_size=5, padding='same', activation=leaky_relu)(x)
    x = layers.UpSampling1D(2)(x)
    x = layers.Conv1D(filters=32, kernel_size=5, padding='same', activation=leaky_relu)(x)
    x = layers.UpSampling1D(2)(x)
    x = layers.Conv1D(filters=64, kernel_size=5, padding='same', activation=leaky_relu)(x)
    x = layers.UpSampling1D(2)(x)
    rx = layers.Conv1D(filters=1, kernel_size=5, padding='same', activation=leaky_relu)(x)
    dcae=keras.Model(inputs_, rx)
    opt = keras.optimizers.Adam(learning_rate=learning_rate, epsilon=1e-8)

    # load saved weights from path
    dir = ('E:\\论文2\\谌洪梦1\\代码\\SHM\\auto-encoder\\Results\\dcae_model\\model(970)\\model_last_19999.ckpt')
    print('Load weights from ', dir)
    dcae.load_weights(dir)
    new_enout=tf.keras.models.Model(inputs=inputs_,outputs=latent)

    print('load success!!! for ',fault)

    #Encode for data train and test
    file_name = load_root + fault + '.pkl'
    x_train,x_test = pickle.load(open(file_name, 'rb'))
    x_train,x_test = tf.reshape(x_train, shape=[-1, 1024, 1]),tf.reshape(x_test, shape=[-1, 1024, 1])
    extracted_features_train = new_enout.predict(x_train)
    extracted_features_test = new_enout.predict(x_test)
    print(extracted_features_train.shape,extracted_features_test.shape)

    # save latents
    with open('E:\\论文2\\谌洪梦1\\代码\\SHM\\auto-encoder\\Results\\dcae_model\\Encoded\\S2L2P0_enc40+291_'+fault+'.pkl', 'wb') as f:
        pickle.dump([extracted_features_train,extracted_features_test], f, pickle.HIGHEST_PROTOCOL)

    #取出dense 层的权重进行保存
    new_enout.summary()
    w,b=new_enout.get_layer('dense').get_weights()
    # 现在保存权重
    with open('E:\\论文2\\谌洪梦1\\代码\\SHM\\auto-encoder\\Results\\dcae_model\\Encoded\\Weights\\S2L2P0_enc40+291_'+fault+'_weights.pkl', 'wb') as f:
        pickle.dump([w,b], f, pickle.HIGHEST_PROTOCOL)




