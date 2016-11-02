import os
import logging
import numpy as np
import pandas as pd
import h5py
import scipy.io as sio 

# attention : this data hasn't been normalized!!!

# ###
# DATA_ROOT = 'data'
# join = os.path.join
# TRAIN = join(DATA_ROOT, 'train.csv')
# train_file = join(DATA_ROOT, 'mnist_train.h5')
# test_file = join(DATA_ROOT, 'mnist_test.h5')

# # logger
# logger = logging.getLogger()
# logger.setLevel(logging.DEBUG)
# sh = logging.StreamHandler()
# sh.setLevel(logging.DEBUG)
# formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
# sh.setFormatter(formatter)
# logger.addHandler(sh)

# # load data from train.csv
# logger.info('Load data from %s', TRAIN)
# df = pd.read_csv(TRAIN)
# data = df.values

# logger.info('Get %d Rows in dataset', len(data))

# # random shuffle
# np.random.shuffle(data)

# # all dataset
# labels = data[:, 0]
# images = data[:, 1:]
###

mwidth = 166
mhight = 33

train_file = './data/music/wtrainint.h5'
test_file = './data/music/wtestint.h5'

trainmat = sio.loadmat('./data/music/trmint.mat')
testmat = sio.loadmat('./data/music/temint.mat')
trdata = trainmat['traindouble']
print trdata.shape
trlabel = trainmat['trainl']
print trlabel.shape
tedata = testmat['testdouble']
telabel = testmat['testl']

Ntr = trdata.shape[0]
Nte = tedata.shape[0]

meant = (np.sum(trdata,0) + np.sum(tedata,0))/(Ntr + Nte)

trdata = trdata - np.tile(meant,[Ntr,1])
tedata = tedata - np.tile(meant,[Nte,1])

print trdata

data = np.concatenate((trdata, tedata))

print data.shape

label = np.concatenate((trlabel,telabel))

trainlabel = np.zeros(((Ntr+Nte)*4/5,1),dtype = np.uint8)
traindata = np.zeros(((Ntr+Nte)*4/5,mhight*mwidth),dtype = np.double)

testlabel = np.zeros(((Ntr+Nte)*1/5,1),dtype = np.uint8)
testdata = np.zeros(((Ntr+Nte)*1/5,mhight*mwidth),dtype = np.double)
for i in range(0,10,1):
	traindata[i*1200:(i+1)*1200] = data[i*1500:i*1500+1200]
	trainlabel[i*1200:(i+1)*1200] = label[i*1500:i*1500+1200]

	testdata[i*300:(i+1)*300] = data[i*1500+1200:(i+1)*1500]
	testlabel[i*300:(i+1)*300] = label[i*1500+1200:(i+1)*1500]


print trainlabel
print traindata[1]


pridetrain = np.column_stack((traindata,trainlabel))
np.random.shuffle(pridetrain)
traindata = pridetrain[:,0:(pridetrain.shape[1]-1)]
trainlabel = pridetrain[:,(pridetrain.shape[1]-1)]



images_train = trdata.reshape(traindata.shape[0],1,mhight,mwidth)
images_test = tedata.reshape(testdata.shape[0],1,mhight,mwidth)
print images_test.shape
print images_train.shape

labels_train = trlabel.reshape(trainlabel.shape[0])
labels_test = telabel.reshape(testlabel.shape[0])
print labels_test.shape
print labels_train.shape




# write to hdf5
if os.path.exists(train_file):
    os.remove(train_file)
if os.path.exists(test_file):
    os.remove(test_file)

#logger.info('Write train dataset to %s', train_file)
with h5py.File(train_file, 'w') as f:
    f['label'] = labels_train.astype(np.double)
    f['data'] = images_train.astype(np.double)

#logger.info('Write test dataset to %s', test_file)
with h5py.File(test_file, 'w') as f:
    f['label'] = labels_test.astype(np.double)
    f['data'] = images_test.astype(np.double)

print "done"
#logger.info('Done')