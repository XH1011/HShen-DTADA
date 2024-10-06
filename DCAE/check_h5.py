import h5py

f=h5py.File('./Results/dcae_model/Encoded/dense_layer_weights.h5','r')
for k in f.keys():

    print(k)

model.get