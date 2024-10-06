
import tensorflow as tf
import pickle
from tensorflow import keras
from keras import layers
from tensorflow.python.framework import ops
import numpy as np

def leaky_relu(x, alpha=0.2):
    return tf.maximum(x, alpha * x)


# Global Settings
batch_size=2 #batch_size=2
num_epochs=1000
learning_rate=1e-5 #learning_rate=1e-52e-4
img_size=1024
img_channels=1

faults=['B-1','B-2','D-1','D-2','E-1','E-2',
        'J-2','K-1']
# faults=['A']
load_root = 'E:\\论文2\\谌洪梦1\\代码1\\SHM\\Data_Chopper\\3_data_Encoded\\10-PZQ\\S10_S2L2P0_enc_'
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

    latent = layers.Dense(units=128)(re)

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

    # load saved dense layer weights from path
    print('Load weights from ', 'E:\\论文2\\谌洪梦1\\代码\\SHM\\auto-encoder\\Results\\dcae_model\\Encoded\\Weights\\S2L2P0_enc_'+fault+'_weights.pkl')
    with open('E:\\论文2\\谌洪梦1\\代码\\SHM\\auto-encoder\\Results\\dcae_model\\Encoded\\Weights\\S2L2P0_enc_'+fault+'_weights.pkl', 'rb') as f:
        w,b = pickle.load(f)
    re_dense=tf.keras.models.Model(inputs=re,outputs=latent)
    # re_dense.summary()
    re_dense.get_layer('dense').set_weights([w,b])
    print('desne layer load completed!!!!')

    dir = 'E:\\论文2\\谌洪梦1\\代码\\SHM\\auto-encoder\\Results\\dcae_model\\model\\model_last_19999.ckpt'
    print('Load weights from ', dir)
    dcae.load_weights(dir)
    new_deout=tf.keras.models.Model(inputs=re,outputs=rx)

    print('AE model load completed!!!!')

    print('laod success!!! for ',fault)
    print('--------------------------------')

    #Decode for data train and test
    file_name = load_root + fault + '.pkl'
    x_train = pickle.load(open(file_name, 'rb'))
    rex_train = np.reshape(new_deout.predict(x_train),(-1,1024))
    print(rex_train.shape)

    with open('E:\\论文2\\谌洪梦1\\代码1\\SHM\\Data_Chopper\\4_Decoded\\S10\\S10_S2L2P0_re_'+fault+'.pkl', 'wb') as f:
        pickle.dump(rex_train, f, pickle.HIGHEST_PROTOCOL)

