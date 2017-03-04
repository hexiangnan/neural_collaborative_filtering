'''
Created on Aug 9, 2016

@author: he8819197
'''
import numpy as np

import theano
import theano.tensor as T
import keras
from keras import backend as K
from keras import initializations
from keras.regularizers import l1, l2, l1l2
from keras.models import Sequential, Model
from keras.layers.core import Dense, Lambda, Activation
from keras.layers import Embedding, Input, Dense, merge, Reshape, Merge, Flatten, Dropout
from keras.optimizers import Adagrad, Adam, SGD, RMSprop
from evaluate import evaluate_model
from Dataset import Dataset
from time import time
import sys
import GMFlogistic, MLPlogistic

def init_normal(shape, name=None):
    return initializations.normal(shape, scale=0.01, name=name)

def get_model(num_users, num_items, mf_dim=10, layers=[10], reg_layers=[0], reg_mf=0, enable_dropout=False):
    assert len(layers) == len(reg_layers)
    num_layer = len(layers) #Number of layers in the MLP
    # Input variables
    user_input = Input(shape=(1,), dtype='int32', name = 'user_input')
    item_input = Input(shape=(1,), dtype='int32', name = 'item_input')
    
    # Embedding layer
    MF_Embedding_User = Embedding(input_dim = num_users, output_dim = mf_dim, name = 'mf_embedding_user',
                                  init = init_normal, W_regularizer = l2(reg_mf), input_length=1)
    MF_Embedding_Item = Embedding(input_dim = num_items, output_dim = mf_dim, name = 'mf_embedding_item',
                                  init = init_normal, W_regularizer = l2(reg_mf), input_length=1)   

    MLP_Embedding_User = Embedding(input_dim = num_users, output_dim = layers[0]/2, name = "mlp_embedding_user",
                                  init = init_normal, W_regularizer = l2(reg_layers[0]), input_length=1)
    MLP_Embedding_Item = Embedding(input_dim = num_items, output_dim = layers[0]/2, name = 'mlp_embedding_item',
                                  init = init_normal, W_regularizer = l2(reg_layers[0]), input_length=1)   
    
    # MF part
    mf_user_latent = Flatten()(MF_Embedding_User(user_input))
    mf_item_latent = Flatten()(MF_Embedding_Item(item_input))
    mf_vector = merge([mf_user_latent, mf_item_latent], mode = 'mul') # element-wise multiply

    # MLP part 
    mlp_user_latent = Flatten()(MLP_Embedding_User(user_input))
    mlp_item_latent = Flatten()(MLP_Embedding_Item(item_input))
    mlp_vector = merge([mlp_user_latent, mlp_item_latent], mode = 'concat')
    for idx in xrange(1, num_layer):
        layer = Dense(layers[idx], W_regularizer= l2(reg_layers[idx]), activation='relu', name="layer%d" %idx)
        mlp_vector = layer(mlp_vector)

    # Concatenate MF and MLP parts
    #mf_vector = Lambda(lambda x: x * alpha)(mf_vector)
    #mlp_vector = Lambda(lambda x : x * (1-alpha))(mlp_vector)
    predict_vector = merge([mf_vector, mlp_vector], mode = 'concat')
    if enable_dropout:
        predict_vector = Dropout(0.5)(predict_vector)
    
    # Final prediction layer
    prediction = Dense(1, activation='sigmoid', init='lecun_uniform', name = "prediction")(predict_vector)
    
    model = Model(input=[user_input, item_input], 
                  output=prediction)
    
    return model

def load_pretrain_model(model, gmf_model, mlp_model, num_layers):
    # MF embeddings
    gmf_user_embeddings = gmf_model.get_layer('user_embedding').get_weights()
    gmf_item_embeddings = gmf_model.get_layer('item_embedding').get_weights()
    model.get_layer('mf_embedding_user').set_weights(gmf_user_embeddings)
    model.get_layer('mf_embedding_item').set_weights(gmf_item_embeddings)
    
    # MLP embeddings
    mlp_user_embeddings = mlp_model.get_layer('user_embedding').get_weights()
    mlp_item_embeddings = mlp_model.get_layer('item_embedding').get_weights()
    model.get_layer('mlp_embedding_user').set_weights(mlp_user_embeddings)
    model.get_layer('mlp_embedding_item').set_weights(mlp_item_embeddings)
    
    # MLP layers
    for i in xrange(1, num_layers):
        mlp_layer_weights = mlp_model.get_layer('layer%d' %i).get_weights()
        model.get_layer('layer%d' %i).set_weights(mlp_layer_weights)
        
    # Prediction weights
    gmf_prediction = gmf_model.get_layer('prediction').get_weights()
    mlp_prediction = mlp_model.get_layer('prediction').get_weights()
    new_weights = np.concatenate((gmf_prediction[0], mlp_prediction[0]), axis=0)
    new_b = gmf_prediction[1] + mlp_prediction[1]
    model.get_layer('prediction').set_weights([0.5*new_weights, 0.5*new_b])    
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
    mf_dim = 8    #embedding size
    layers = eval("[16,8]")
    reg_layers = eval("[0,0]")
    reg_mf = 0
    num_negatives = 4   #number of negatives per positive instance
    weight_negatives = 1.0
    learner = "Adam"
    learning_rate = 0.001
    num_epochs = 100
    batch_size = 256
    verbose = 1
    enable_dropout = False
    mf_pretrain = ''
    mlp_pretrain = ''
    
    if (len(sys.argv) > 3):
        dataset_name = sys.argv[1]
        mf_dim = int(sys.argv[2])
        layers = eval(sys.argv[3])
        reg_layers = eval(sys.argv[4])
        reg_mf = float(sys.argv[5])
        num_negatives = int(sys.argv[6])
        weight_negatives = float(sys.argv[7])
        learner = sys.argv[8]
        learning_rate = float(sys.argv[9])
        num_epochs = int(sys.argv[10])
        batch_size = int(sys.argv[11])
        verbose = int(sys.argv[12])
        if (sys.argv[13] == 'true' or sys.argv[13] == 'True'):
            enable_dropout = True
        if len(sys.argv) > 14:
            mf_pretrain = sys.argv[14]
            mlp_pretrain = sys.argv[15]
            
    topK = 10
    evaluation_threads = 1#mp.cpu_count()
    print("NeuMF(%s) Dropout %s: mf_dim=%d, layers=%s, regs=%s, reg_mf=%.1e, num_negatives=%d, weight_negatives=%.2f, learning_rate=%.1e, num_epochs=%d, batch_size=%d, verbose=%d"
          %(learner, enable_dropout, mf_dim, layers, reg_layers, reg_mf, num_negatives, weight_negatives, learning_rate, num_epochs, batch_size, verbose))
        
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
    model = get_model(num_users, num_items, mf_dim, layers, reg_layers, reg_mf, enable_dropout)
    if learner.lower() == "adagrad": 
        model.compile(optimizer=Adagrad(lr=learning_rate), loss='binary_crossentropy')
    elif learner.lower() == "rmsprop":
        model.compile(optimizer=RMSprop(lr=learning_rate), loss='binary_crossentropy')
    elif learner.lower() == "adam":
        model.compile(optimizer=Adam(lr=learning_rate), loss='binary_crossentropy')
    else:
        model.compile(optimizer=SGD(lr=learning_rate), loss='binary_crossentropy')
    
    # Load pretrain model
    if mf_pretrain != '' and mlp_pretrain != '':
        gmf_model = GMFlogistic.get_model(num_users,num_items,mf_dim)
        gmf_model.load_weights(mf_pretrain)
        mlp_model = MLPlogistic.get_model(num_users,num_items, layers, reg_layers)
        mlp_model.load_weights(mlp_pretrain)
        model = load_pretrain_model(model, gmf_model, mlp_model, len(layers))
        print("Load pretrained GMF (%s) and MLP (%s) models done. " %(mf_pretrain, mlp_pretrain))
        
    # Init performance
    (hits, ndcgs) = evaluate_model(model, testRatings, testNegatives, topK, evaluation_threads)
    hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
    print('Init: HR = %.4f, NDCG = %.4f' % (hr, ndcg))
    if hr > 0.6:
        model.save_weights('Pretrain/%s_NeuMF_%d_neg_%d_hr_%.4f_ndcg_%.4f.h5' %(dataset_name, layers[-1], num_negatives, hr, ndcg), overwrite=True)

    # Training model
    best_hr, best_ndcg  = hr, ndcg
    for epoch in xrange(num_epochs):
        t1 = time()
        # Generate training instances
        user_input, item_input, labels, weights = get_train_instances(train, num_negatives, weight_negatives, user_weights)
        
        # Training
        hist = model.fit([np.array(user_input), np.array(item_input)], #input
                         np.array(labels), # labels 
                         sample_weight=np.array(weights), # weight of samples
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
                    model.save_weights('Pretrain/%s_NeuMF_%d_neg_%d_hr_%.4f_ndcg_%.4f.h5' %(dataset_name, layers[-1], num_negatives, hr, ndcg), overwrite=True)
            if ndcg > best_ndcg:
                best_ndcg = ndcg

    print("End. best HR = %.4f, best NDCG = %.4f" %(best_hr, best_ndcg))
