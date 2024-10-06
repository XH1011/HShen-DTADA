import tensorflow as tf
from keras import layers, models
import pickle as pickle
import matplotlib.pyplot as plt
from pylab import xticks
from keras.initializers import RandomNormal
_leaky_relu = lambda net: tf.nn.leaky_relu(net, alpha=0.3)
import numpy as np
from sklearn.preprocessing import normalize,scale

#self
from util import TFData_preprocessing
from model3 import DANN
from zca_util import ZCA
mode='gen'
faults=['B-1','B-2','D-2','E-1','E-2','J-2','K-1']
datasets_gen_T=[]
datasets_real_T=[]
datasets_gen_t=[]
datasets_real_t=[]
for fault in faults:
    # load target data
    file_name='E:\\论文2\\谌洪梦1\\代码1\\SHM\\Data_Chopper\\5_Generated\\S15\\Gen15_S2L2P0_enc_'+fault+'_train.pkl'
    enc_train_gen,enc_test_gen = pickle.load(open(file_name, 'rb'))
    datasets_gen_T.extend(enc_train_gen)
    datasets_gen_t.extend(enc_test_gen)

    # load source data
    file_name='E:\\论文2\\谌洪梦1\\代码1\\SHM\\Data_Chopper\\3_data_Encoded\\S15\\S15_S2L2P0_enc_'+fault+'.pkl'
    enc_train_real,enc_test_real = pickle.load(open(file_name, 'rb'))
    datasets_real_T.extend(enc_train_real)
    datasets_real_t.extend(enc_test_real)

x_gen=np.array(datasets_gen_T) #target train
x_real=np.array(datasets_real_T) #source train

x_gen_t=np.array(datasets_gen_t) #target test
x_real_t=np.array(datasets_real_t) #source test


source_train=x_gen
source_test=x_gen_t
target_train=x_real
target_test=x_real_t

# 为source_train生成标签
num_T_source = 679
source_train_label = tf.one_hot(np.array([0]*num_T_source+[1]*num_T_source+[2]*num_T_source+[3]*num_T_source+[4]*num_T_source+[5]*num_T_source+[6]*num_T_source), depth=7)

# 为target_train生成标签
num_T_target = 10
target_train_label = tf.one_hot(np.array([0]*num_T_target+[1]*num_T_target+[2]*num_T_target+[3]*num_T_target+[4]*num_T_target+[5]*num_T_target+[6]*num_T_target), depth=7)

# 测试标签保持不变
num_t = 291
test_label = np.array([0]*num_t+[1]*num_t+[2]*num_t+[3]*num_t+[4]*num_t+[5]*num_t+[6]*num_t)


# 数据预处理
source_train_db = TFData_preprocessing(source_train, source_train_label, batch_size=15, drop_last=True)
source_test_db = TFData_preprocessing(source_test, test_label, batch_size=15, drop_last=True)

target_train_db = TFData_preprocessing(target_train, target_train_label, batch_size=15, drop_last=True)
target_test_db = TFData_preprocessing(target_test, test_label, batch_size=15, drop_last=True)


# num_T=10
# train_label=tf.one_hot(np.array([0]*num_T+[1]*num_T+[2]*num_T+[3]*num_T+
#                                 [4]*num_T+[5]*num_T+[6]*num_T+[7]*num_T),depth=8)
# num_t=291
# test_label=np.array([0]*num_t+[1]*num_t+[2]*num_t+[3]*num_t+
#                     [4]*num_t+[5]*num_t+[6]*num_t+[7]*num_t)
# #
# source_train_db = TFData_preprocessing(source_train, train_label, batch_size=32)
# source_test_db = TFData_preprocessing(source_test, test_label, batch_size=32)
#
# target_train_db = TFData_preprocessing(target_train, train_label, batch_size=32)
# target_test_db = TFData_preprocessing(target_test, test_label, batch_size=32)

dann=DANN(input_shape=(128,),num_classes=7)

#pre-train
# dann.train_source_only(source_train_db, source_test_db, target_test_db, epochs=1000)
dann.train(source_train_db, target_train_db,source_test_db,target_test_db, epochs=100)
