'''
Created on Aug 9, 2016
Logistic loss for learning MLP model.

@author: he8819197
'''
import numpy as np

import theano
import theano.tensor as T
import keras
from keras import backend as K
from keras import initializations
from keras.regularizers import l2, activity_l2
from keras.models import Sequential, Graph, Model
from keras.layers.core import Dense, Lambda, Activation
from keras.layers import Embedding, Input, Dense, merge, Reshape, Merge, Flatten, Dropout
from keras.constraints import maxnorm
from keras.optimizers import Adagrad, Adam, SGD, RMSprop
from evaluate import evaluate_model
from Dataset import Dataset
from time import time
import sys

def init_normal(shape, name=None):
    return initializations.normal(shape, scale=0.01, name=name)

def get_model(num_users, num_items, layers = [20,10], reg_layers=[0,0]):
    assert len(layers) == len(reg_layers)
    num_layer = len(layers) #Number of layers in the MLP
    # Input variables
    user_input = Input(shape=(1,), dtype='int32', name = 'user_input')
    item_input = Input(shape=(1,), dtype='int32', name = 'item_input')

    MLP_Embedding_User = Embedding(input_dim = num_users, output_dim = layers[0]/2, name = 'user_embedding',
                                  init = init_normal, W_regularizer = l2(reg_layers[0]), input_length=1)
    MLP_Embedding_Item = Embedding(input_dim = num_items, output_dim = layers[0]/2, name = 'item_embedding',
                                  init = init_normal, W_regularizer = l2(reg_layers[0]), input_length=1)   
    
    # Crucial to flatten an embedding vector!
    user_latent = Flatten()(MLP_Embedding_User(user_input))
    item_latent = Flatten()(MLP_Embedding_Item(item_input))
    
    # The 0-th layer is the concatenation of embedding layers
    vector = merge([user_latent, item_latent], mode = 'concat')
    
    # MLP layers
    for idx in xrange(1, num_layer):
        layer = Dense(layers[idx], W_regularizer= l2(reg_layers[idx]), activation='relu', name = 'layer%d' %idx)
        vector = layer(vector)
        
    # Final prediction layer
    prediction = Dense(1, activation='sigmoid', init='lecun_uniform', name = 'prediction')(vector)
    
    model = Model(input=[user_input, item_input], 
                  output=prediction)
    
    return model

def get_train_instances(train, num_negatives, weight_negatives, user_weights):
    user_input, item_input, labels, weights = [],[],[],[]
    num_users = train.shape[0]
    for (u, i) in train.keys():
        # positive instance
        user_input.append(u)
        item_input.append(i)
        labels.append(1)
        weights.append(user_weights[u])
        # negative instances
        for t in xrange(num_negatives):
            j = np.random.randint(num_items)
            while train.has_key((u, j)):
                j = np.random.randint(num_items)
            user_input.append(u)
            item_input.append(j)
            labels.append(0)
            weights.append(weight_negatives * user_weights[u])
    return user_input, item_input, labels, weights

if __name__ == '__main__':
    dataset_name = "ml-1m"
    layers = eval("[16,8]")
    reg_layers = eval("[0,0]")
    num_negatives = 1   #number of negatives per positive instance
    weight_negatives = 1
    learner = "Adam"
    learning_rate = 0.001
    epochs = 100
    batch_size = 256
    verbose = 1
    
    if (len(sys.argv) > 3):
        dataset_name = sys.argv[1]
        layers = eval(sys.argv[2])
        reg_layers = eval(sys.argv[3])
        num_negatives = int(sys.argv[4])
        weight_negatives = float(sys.argv[5])
        learner = sys.argv[6]
        learning_rate = float(sys.argv[7])
        epochs = int(sys.argv[8])
        batch_size = int(sys.argv[9])
        verbose = int(sys.argv[10])
    
    topK = 10
    evaluation_threads = 1#mp.cpu_count()
    print("MLP-logistic(%s) Settings: layers=%s, reg_layers=%s, num_neg=%d, weight_neg=%.2f, learning_rate=%.1e, epochs=%d, batch_size=%d, verbose=%d"
          %(learner, layers, reg_layers, num_negatives, weight_negatives, learning_rate, epochs, batch_size, verbose))
    
    # Loading data
    t1 = time()
    dataset = Dataset("Data/"+dataset_name)
    train, testRatings, testNegatives = dataset.trainMatrix, dataset.testRatings, dataset.testNegatives
    num_users, num_items = train.shape
    total_weight_per_user = train.nnz / float(num_users)
    train_csr, user_weights = train.tocsr(), []
    for u in xrange(num_users):
        #user_weights.append(total_weight_per_user / float(train_csr.getrow(u).nnz))
        user_weights.append(1)
    print("Load data done [%.1f s]. #user=%d, #item=%d, #train=%d, #test=%d" 
          %(time()-t1, num_users, num_items, train.nnz, len(testRatings)))
    
    # Build model
    model = get_model(num_users, num_items, layers, reg_layers)
    if learner.lower() == "adagrad": 
        model.compile(optimizer=Adagrad(lr=learning_rate), loss='binary_crossentropy')
    elif learner.lower() == "rmsprop":
        model.compile(optimizer=RMSprop(lr=learning_rate), loss='binary_crossentropy')
    elif learner.lower() == "adam":
        model.compile(optimizer=Adam(lr=learning_rate), loss='binary_crossentropy')
    else:
        model.compile(optimizer=SGD(lr=learning_rate), loss='binary_crossentropy')    
    
    # Check Init performance
    (hits, ndcgs) = evaluate_model(model, testRatings, testNegatives, topK, evaluation_threads)
    hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
    print('Init: HR = %.4f, NDCG = %.4f' %(hr, ndcg))
    
    # Train model
    loss_pre = sys.float_info.max
    best_hr, best_ndcg = 0, 0
    for epoch in xrange(epochs):
        t1 = time()
        # Generate training instances
        user_input, item_input, labels, weights = get_train_instances(train, num_negatives, weight_negatives, user_weights)
    
        # Training        
        hist = model.fit([np.array(user_input), np.array(item_input)], #input
                         np.array(labels), # labels 
                         batch_size=batch_size, nb_epoch=1, verbose=0, shuffle=True)
        t2 = time()

        # Evaluation
        if epoch %verbose == 0:
            (hits, ndcgs) = evaluate_model(model, testRatings, testNegatives, topK, evaluation_threads)
            hr, ndcg, loss = np.array(hits).mean(), np.array(ndcgs).mean(), hist.history['loss'][0]
            print('Iteration %d [%.1f s]: HR = %.4f, NDCG = %.4f, loss = %.4f [%.1f s]' 
                  % (epoch,  t2-t1, hr, ndcg, loss , time()-t2))
            if hr > best_hr:
                best_hr = hr
                if hr > 0.6:
                    model.save_weights('Pretrain/%s_MLP_%d_neg_%d_hr_%.4f_ndcg_%.4f.h5' %(dataset_name, layers[-1], num_negatives, hr, ndcg), overwrite=True)
            if ndcg > best_ndcg:
                best_ndcg = ndcg

    print("End. best HR = %.4f, best NDCG = %.4f" %(best_hr, best_ndcg))