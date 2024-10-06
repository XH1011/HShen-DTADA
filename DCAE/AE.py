import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import os
import pickle

def leaky_relu(x, alpha=0.2):
    return tf.maximum(x, alpha * x)
def train_on_step(images_batch):
    image_batch = tf.reshape(images_batch, shape=[-1, 1024, 1])
    loss=train_loss(image_batch)#(3072,1) (3072,)
    return loss

def train_loss(image_batch):
    with tf.GradientTape() as tape:
        # model
        recon_image = dcae(image_batch, training=True)
        # print('reree',recon_image.shape,image_batch.shape)
        # loss
        diff = recon_image - image_batch
        recon_loss = tf.reduce_mean(tf.reduce_sum(diff ** 2, axis=1))
        # print('loss:',image_batch.shape,images_batch.shape,recon_image.shape)
        total_loss = recon_loss
    gradients = tape.gradient(total_loss, dcae.trainable_weights)
    opt.apply_gradients(zip(gradients, dcae.trainable_weights))

    return total_loss.numpy()
# Global Settings
batch_size=32 #batch_size=2
num_epochs=40000
learning_rate=2e-4 #learning_rate=2e-4
img_size=1024
img_channels=1

file_name='./data_chopper/divide/S2L2P0_train_test_n/S2L2P0_A.pkl'
with open(file_name, 'rb') as f:
    data_train,data_test = pickle.load(f)
data_train=tf.cast(data_train,dtype=tf.float32)
train_ds=tf.data.Dataset.from_tensor_slices(data_train).shuffle(10000).batch(batch_size)

#inputs
inputs_=layers.Input(shape=(img_size, img_channels), name="image_input")
# 2，神经网络
layers=tf.keras.layers
# ### Encoder
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
print('latent vector after encoder', re.shape)

x = layers.Dense(units=128, activation=leaky_relu)(re)

x = tf.reshape(x, [-1, 32, 4])
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
print(rx.shape,inputs_.shape)
print('Built Encoder../')

# #Build model
dcae=keras.Model(inputs_, rx)

# # Opimizer and loss function
opt=keras.optimizers.Adam(learning_rate=learning_rate,epsilon=1e-8)
print('Network Summary-->')
dcae.summary()

# # load saved weights from path
# dir = './Results/dcae_model/model/model_6000.ckpt'
# print('Load weights from ', dir)
# dcae.load_weights(dir)


# --------------->>>Training Phase<<<---------------------------
# Run
loss_list=[]
# total_batch=350
total_batch = int(len(data_train) / batch_size)
for epoch in range(num_epochs):
    ave_cost=0
    for images_batch in train_ds:
        loss=train_on_step(images_batch)
        ave_cost+=loss/total_batch
    print("Epoch:", (epoch + 1), "cost =", "{:.3f}".format(ave_cost))
    loss_list.append(ave_cost)

    # Save the model weights every 100 epoches
    if epoch%1000==0:
      save_dir='./Results/dcae_model/model/'
      os.makedirs(save_dir, exist_ok=True)
      dcae.save_weights(save_dir+'model_'+str(epoch)+'.ckpt')

    # save loss
    loss_curve = np.array(loss_list)
    save_loss = './Results/dcae_model/loss/'
    os.makedirs(save_loss, exist_ok=True)
    np.savetxt(save_loss + 'dcae_loss.txt', loss_curve)

# Save the model weights in the last step
save_dir='./Results/dcae_model/model/'
dcae.save_weights(save_dir+'model_last_'+str(epoch)+'.ckpt')
print('Optimization Finished')



