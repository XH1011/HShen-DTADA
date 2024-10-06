import numpy as np
import tensorflow as tf


def freq_Analysis(x):
    x-=np.mean(x)
    x=do_fft_norm(x)
    return x


def do_fft_norm(x):
    xx0_fft=np.abs(np.fft.fft(x))*2/len(x)
    xx0_fft=xx0_fft[:len(x)]
    return xx0_fft


# def TFData_preprocessing(x,y,batch_size,conditional=True):
#   if conditional:
#       x=tf.data.Dataset.from_tensor_slices((x,y))
#       x=x.shuffle(10000).batch(batch_size)
#   else:
#       x=tf.data.Dataset.from_tensor_slices(x)
#       x=x.shuffle(10000).batch(batch_size)
#
#   return x

def TFData_preprocessing(x, y, batch_size, conditional=True, drop_last=False):
    if conditional:
        x = tf.data.Dataset.from_tensor_slices((x, y))
        x = x.shuffle(10000).batch(batch_size, drop_remainder=drop_last)
    else:
        x = tf.data.Dataset.from_tensor_slices(x)
        x = x.shuffle(10000).batch(batch_size, drop_remainder=drop_last)

    return x
