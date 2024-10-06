#self package
from BuildModel2 import *
from network_utils import *
from Unet_utils import *
from utils import *

import pickle


#self difined
def train_on_step(image_batch):
    image_re=tf.reshape(image_batch,shape=[-1,128,1])
    diffusion_loss=train_loss(image_re)
    return diffusion_loss

def train_loss(image_batch):
    t = tf.random.uniform(minval=0, maxval=timesteps, shape=(image_batch.shape[0],), dtype=tf.int64)
    with tf.GradientTape() as tape:
        noise=tf.random.normal(shape=tf.shape(image_batch),dtype=image_batch.dtype)
        image_noise=gdf_util.q_sample(image_batch,t,noise)
        pred_noise = ddpm([image_noise, t], training=True)
        difusion_loss = mseloss(noise, pred_noise)
    gradients = tape.gradient(difusion_loss,
                              ddpm.trainable_weights)
    opt.apply_gradients(zip(gradients, ddpm.trainable_weights))
    return difusion_loss.numpy()

def generate_images(num_images=679):
    # 1. Randomly sample noise (starting point for reverse process)
    samples = tf.random.normal(
        shape=(num_images, img_size, img_channels), dtype=tf.float32
    )
    # 2. Sample from the model(970) iteratively
    for t in reversed(range(0, timesteps)):
        tt = tf.cast(tf.fill(num_images, t), dtype=tf.int64)
        pred_noise = ddpm.predict([samples, tt], verbose=0, batch_size=num_images)

        samples = gdf_util.p_sample(
            pred_noise, samples, tt, clip_denoised=True
        )

    # 3. Return generated samples
    return samples
def generate_images_test(x,num_images):
    # 1. Randomly sample noise (starting point for reverse process)
    # samples = tf.random.normal(
    #     shape=(num_images, img_size, img_channels), dtype=tf.float32
    # )
    samples = tf.cast(tf.reshape(x,shape=[num_images,img_size,img_channels]),dtype=tf.float32)
    print(samples.shape)
    # 2. Sample from the model(970) iteratively
    for t in reversed(range(0, timesteps)):
        # print(t)
        tt = tf.cast(tf.fill(num_images, t), dtype=tf.int64)
        pred_noise = ddpm.predict([samples, tt], verbose=0, batch_size=num_images)

        samples = gdf_util.p_sample(
            pred_noise, samples, tt, clip_denoised=True
        )

    # 3. Return generated samples
    return samples

# data input
batch_size=32
num_epochs=10000
timesteps = 1000
norm_groups=8
learning_rate=2e-4
img_size=128
img_channels=1
first_conv_channels=16
channel_multipier=[4,2,1,1/2]
widths=[first_conv_channels* mult for mult in channel_multipier]
has_attention=[False,False,False,False]
num_res_blocks = 2

velocity='std'

type='E-2'

path = 'E:\\论文2\\谌洪梦1\\代码1\\SHM\\Data_Chopper\\3_data_Encoded\\S15\\S15_S2L2P0_enc_'+type+'_train.pkl'
with open(path, 'rb') as f:
    data_train = pickle.load(f)
data_train=data_train.astype(np.float32)
train_ds=tf.data.Dataset.from_tensor_slices(data_train).shuffle(50000).batch(batch_size)

# Build model(970)
image_input = layers.Input(shape=( img_size, img_channels), name="generator_input")
time_input = keras.Input(shape=(), dtype=tf.int64, name="time_input")  # *(None,)
ddpm_x = build_model(
    input=image_input,
    time_input=time_input,
    widths=widths,
    has_attention=has_attention,
    first_conv_channels=first_conv_channels,
    num_res_blocks=num_res_blocks,
    norm_groups=norm_groups,
    activation_fn=keras.activations.swish,
)

ddpm=keras.Model([image_input,time_input],ddpm_x)

gdf_util = GaussianDiffusion(timesteps=timesteps)
opt=keras.optimizers.Adam(learning_rate=learning_rate)
mseloss = keras.losses.MeanSquaredError()
print('Network Summary-->')
ddpm.summary()

# Training
loss_list=[]
for epoch in range(num_epochs):
    for images_batch in train_ds:
        difusion_loss=train_on_step(images_batch)
    loss_list.append(difusion_loss)

    if epoch%500==0:
      # Save the model(970) weights
      save_dir='./data_Encoded/data_Encoded_main/model(970)'+type+'/'
      os.makedirs(save_dir, exist_ok=True)
      ddpm.save_weights(save_dir+'model_'+str(epoch)+'.ckpt')
    print('epoch',epoch,'difusion loss:',difusion_loss)

# Save the trained model(970) weights (last step)
save_dir = './data_Encoded/data_Encoded_main/model(970)'+type+'/'
ddpm.save_weights(save_dir + 'model_last_' + str(epoch) + '.ckpt')
np.savetxt('E:\\论文2\\ddpm_shm\\Results\\ddpm_loss30_'+type+'.txt',np.array(loss_list))

# Load the model(970) weights
dir='./data_Encoded/data_Encoded_main/model(970)'+type+'/model_last_9999.ckpt'
print('Load weights from ',dir)
ddpm.load_weights(dir)
print('load weights successfully!!!')
# ddpm.summary()

# Generate the data from trained model(970)
print('start generate')
generated_samples=np.squeeze(generate_images(num_images=679))
generated_samples_test=np.squeeze(generate_images(num_images=291))

# Save the generated data to the directory
with open('E:\\论文2\\谌洪梦1\\代码1\\SHM\\Data_Chopper\\5_Generated\\S15\\Gen15_S2L2P0_enc_'+type+'_train.pkl','wb') as f:
    pickle.dump([generated_samples,generated_samples_test],f,protocol=4)

