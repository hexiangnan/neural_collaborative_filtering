import numpy as np

import theano
import theano.tensor as T
import keras
from keras import backend as K
from keras import initializations
from keras.models import Sequential, Model
from keras.layers.core import Dense, Lambda, Activation
from keras.layers import Embedding, Input, Dense, merge, Reshape, Merge, Flatten, Dropout
from keras.optimizers import Adagrad, Adam, SGD, RMSprop
from evaluate import evaluate_model
from Dataset import Dataset
from time import time
import sys
import NeuMF

model_file = "Pretrain/ml-1m_NeuMF_64_neg_4_hr_0.7301_ndcg_0.4465.h5"
dataset_name = "ml-1m"
mf_dim = 64
layers = [512,256,128,64]

reg_layers = [0,0,0,0]
reg_mf = 0

# Loading data
t1 = time()
dataset = Dataset("Data/"+dataset_name)
train, testRatings, testNegatives = dataset.trainMatrix, dataset.testRatings, dataset.testNegatives
num_users, num_items = train.shape
print("Load data done [%.1f s]. #user=%d, #item=%d, #train=%d, #test=%d" 
      %(time()-t1, num_users, num_items, train.nnz, len(testRatings)))

# Get model
model = NeuMF.get_model(num_users, num_items, mf_dim, layers, reg_layers, reg_mf, False)
model.load_weights(model_file)

# Evaluate performance
print("K\tHR\tNDCG")
for topK in xrange(1, 10):
    (hits, ndcgs) = evaluate_model(model, testRatings, testNegatives, topK, 1)
    hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
    print("%d\t%.4f\t%.4f" %(topK, hr, ndcg))