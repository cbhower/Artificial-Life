# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 08:53:42 2018

@author: Christian

Predict wind values using and RNN. Search hyperparameter space using GA to find best RNN.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split as split
import os

from keras import metrics
from keras.layers import LSTM, Dense, SimpleRNN, Dropout
from keras.models import Sequential
from deap import base, creator, tools, algorithms
from scipy.stats import bernoulli
from bitstring import BitArray

from matplotlib import pyplot as plt

###
# Read in Data
os.chdir("C:/Users/Christian/windData")
#data = pd.read_csv('train.csv')
#data = np.reshape(np.array(data['wp1']),(len(data['wp1']),1))



iv = np.array(range(0, 20000))
dv = np.array(np.sin((3*iv-20)/6))
data = np.reshape(np.array(dv),(len(iv),1))


# Use first 17,257 points as training/validation and rest of the 1500 points as test set.
train_data = data[0:17257]
test_data = data[17257:]
dropout_dict = {0: .1 , 1: .2, 2: .3, 3: .4}

logbook = []

# Define Custom Functions 
def prepare_dataset(data, window_size):
    X, Y = np.empty((0,window_size)), np.empty((0))
    for i in range(len(data)-window_size-1):
        X = np.vstack([X,data[i:(i + window_size),0]])
        Y = np.append(Y,data[i + window_size,0])   
    X = np.reshape(X,(len(X),window_size,1))
    Y = np.reshape(Y,(len(Y),1))
    return X, Y

def train_evaluate(ga_individual_solution):
    # Decode GA solution to integer for window_size and num_units
    window_size = BitArray(ga_individual_solution[0:4]).uint
    num_units = BitArray(ga_individual_solution[4:8]).uint
    dropout_key  = BitArray(ga_individual_solution[8:]).uint
    dropout_rate = dropout_dict[dropout_key]
    print('\nWindow Size: ', window_size, ', Num of Units: ', num_units, ', Dropout Rate: ', dropout_rate)
    print(ga_individual_solution[0:4],ga_individual_solution[4:8],ga_individual_solution[8:] )
    print(BitArray(ga_individual_solution[0:4]))
    #print(list('{BitArray(ga_individual_solution[0:4])}'.format(6)))
    
    # Set minumum sizes for window and units to 1
    if window_size == 0:
        ga_individual_solution[0:4] = [0,0,0,1]
        window_size = 1
    if num_units == 0:
        ga_individual_solution[4:8] = [0,0,0,1]
        num_units = 1
    print('\nWindow Size: ', window_size, ', Num of Units: ', num_units, ', Dropout Rate: ', dropout_rate)
    print(ga_individual_solution[0:4],ga_individual_solution[4:8],ga_individual_solution[8:] )
    
    # Segment the train_data based on new window_size; split into train and validation (80/20)
    X,Y = prepare_dataset(train_data,window_size)
    X_train, X_val, y_train, y_val = split(X, Y, test_size = 0.20, random_state = 1120)
    
    # Train LSTM model and predict on validation set
    model = Sequential()
    model.add(LSTM(num_units, input_shape=(window_size,1)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer='rmsprop',loss='mse', metrics = ['mae'])
    history = model.fit(X_train, y_train, epochs=4, batch_size=10,shuffle=True)
    y_pred = model.predict(X_val)
    
    plt.plot(history.history['loss'], label='train')
    
    # Calculate the RMSE score as fitness score for GA
    mse = mean_squared_error(y_val, y_pred)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    adjusted_rmse = .1*((num_units/16)+((np.square(num_units)*window_size)/(np.power(16,3)))) + rmse
    print('Validation RMSE: ', rmse,'\nValidation MSE: ', mse, 'adjusted: ', adjusted_rmse)
    print
    return adjusted_rmse, 

def trait_swap(ind1,ind2):
    trait1_swap = bernoulli.rvs(.5)
    trait2_swap = bernoulli.rvs(.5)
    trait3_swap = bernoulli.rvs(.5)
    #print('/nparent1: ' ,ind1[:], 'parent2: ', ind2[:] )
    if trait1_swap == 1:
        ind1[:4], ind2[:4], = ind2[:4], ind1[:4]
    if trait2_swap == 1:
        ind1[4:8], ind2[4:8], = ind2[4:8], ind1[4:8]
    if trait3_swap == 1:
        ind1[8:], ind2[8:], = ind2[8:], ind1[8:]
    #print('/nchild1: ' ,ind1[:], 'child2: ', ind2[:] )    
    return ind1, ind2

'''
def step_mutate(individual, indpb):
    for i in range(len(individual)):
        if random.random() < indpb:
            window_size = BitArray(individual[i][0:4]).uint
            num_units = BitArray(individual[i][4:8]).uint
            dropout_key  = BitArray(individual[i][8:]).uint
            
            
            
            list(map(int,R.bin))
    return individual,
'''

# Genetic Algorithm
population_size = 8
num_generations = 8
gene_length = 10

# As we are trying to minimize the RMSE score, that's why using -1.0. 
# In case, when you want to maximize accuracy for instance, use 1.0
creator.create('FitnessMin', base.Fitness, weights = (-1.0,))
creator.create('Individual', list , fitness = creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register('binary', bernoulli.rvs, 0.3)
toolbox.register('individual', tools.initRepeat, creator.Individual, toolbox.binary, 
n = gene_length)
toolbox.register('population', tools.initRepeat, list , toolbox.individual)

toolbox.register('mate', tools.cxOrdered)
toolbox.register('mutate', tools.mutFlipBit, indpb = 0.2)
toolbox.register('select', tools.selTournament, tournsize = 2)
toolbox.register('evaluate', train_evaluate)

population = toolbox.population(n = population_size)
algorithms.eaSimple(population, toolbox, cxpb = 0.3, mutpb = 0.1, 
                    ngen = num_generations, verbose = True)    

best_individuals = tools.selBest(population, k = 6)
best_window_size = None
best_num_units = None
best_dropout_key = None


for bi in best_individuals:
    best_window_size = BitArray(bi[0:4]).uint
    best_num_units = BitArray(bi[4:8]).uint
    best_dropout_key  = BitArray(bi[8:]).uint 
    print('\nChamp!: ','Window Size: ', best_window_size, ', Num of Units: ', best_num_units,
          'Best Dropout: ', dropout_dict[best_dropout_key])
    print(bi[0:4],bi[4:8],bi[8:])



X_train,y_train = prepare_dataset(train_data,best_window_size)
X_test, y_test = prepare_dataset(test_data,best_window_size)

model = Sequential()
model.add(LSTM(best_num_units, input_shape=(best_window_size,1)))
model.add(Dropout(dropout_dict[best_dropout_key]))
model.add(Dense(1, activation='linear'))
model.compile(optimizer='rmsprop',loss='mean_squared_error')
model.fit(X_train, y_train, epochs=4, batch_size=10,shuffle=True)
y_pred = model.predict(X_test) 


rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print('Test RMSE: ', rmse)


history = model.fit(X_train, y_train, epochs=2, batch_size=10, verbose=2, shuffle=False)
# plot history

print(history.history)
plt.plot(history.history['loss'], label='train')
plt.plot(['loss'], label='test')
plt.legend()
plt.show()



#############BitArray experiments


list(map(int,R.bin))
