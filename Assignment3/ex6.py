#!/usr/bin/env python
# coding: utf-8

# In[1]:


import random
import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import minmax_scale
import matplotlib.pyplot as plt
from network import NeuralNetwork
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


def plots(loss_fun: str, config: str):
    plt.clf()
    plt.figure(figsize=(7.5,4))
    plt.plot(x_epochs, y_train, label='Training')
    plt.plot(x_epochs, y_eval, label='Validation')

    plt.xlabel('Epochs')
    plt.ylabel(loss_fun)

    plt.title("Plot of '{}' over training and validation data".format(loss_fun))

    plt.legend()
#     plt.savefig('fig1.png', dpi = 300)
    plt.savefig(config + '.png')


# In[34]:


df = pd.read_excel('HW3train.xlsx')
x0 = minmax_scale(df['X_0'].tolist())
x1 = minmax_scale(df['X_1'].tolist())
y = df['y'].tolist()
training_sets = []
for i in range(len(x0)):
    training_sets.append([[x0[i],x1[i]],[y[i]]])

df = pd.read_excel('HW3validate.xlsx')
x0 = minmax_scale(df['X_0'].tolist())
x1 = minmax_scale(df['X_1'].tolist())
y = df['y'].tolist()
validation_sets = []
for i in range(len(x0)):
    validation_sets.append([[x0[i],x1[i]],[y[i]]])

params = [[0.001, 0.01, 0.1, 0.2, 0.5], "cross", "leaky", "leaky", "sigmoid"]
stds = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
accuracy = []
# loop over all learning rates
for i in range(len(params[0])):
    for j in range(len(stds)):
        y_train = []
        y_eval = []
        x_epochs = []
        
        nn = NeuralNetwork(2, 10, 10, 1, learning_rate=params[0][i], std = stds[j], loss_function=params[1], hidden_layer_1_activation=params[2], hidden_layer_2_activation=params[3], output_layer_activation=params[4])
        prev_error = 2
        error = 1
        iteration = 0
        epochs = 0
        BATCH_SIZE = 32

        while epochs < 35:
            nn.train(training_sets[BATCH_SIZE*iteration:BATCH_SIZE+BATCH_SIZE*iteration])
            y_train.append(nn.calculate_total_error(training_sets))
            y_eval.append(nn.calculate_total_error(validation_sets))
            x_epochs.append(epochs)
            if nn.calculate_total_error(training_sets) < 0.15:
                break
            iteration += 1
            if iteration % (math.ceil(len(training_sets) / BATCH_SIZE)) == 0:
                iteration = 0
                epochs += 1
        accurate = nn.count_correct(validation_sets)*100 / 81
        accuracy.append(accurate)
        plots("Cross-Entropy", "{}_{}_{}".format(params[0][i], stds[j], accurate))
        nn.undo()


# In[24]:


from os import listdir
from os.path import isfile, join

mypath = "./ex6/"

files = [f for f in listdir(mypath) if isfile(join(mypath, f))]

learn = [0.001, 0.01, 0.1, 0.2, 0.5]
sigma = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
percent = np.zeros(shape = (len(learn), len(sigma)))


float(files[0].split("_")[2].replace(".png", ""))
for file in files:
    name = file.split("_")
    i = learn.index(float(name[0]))
    j = sigma.index(float(name[1]))
    percent[i] [j] = float(name[2].replace(".png", ""))
    
print(learn)
print(sigma)
print(percent)


# In[35]:


import seaborn as sns

sns.heatmap(percent, xticklabels=sigma, yticklabels=learn)


# In[7]:


import os 
from os import listdir
from os.path import isfile, join

mypath = "./ex6/"

files = [f for f in listdir(mypath) if isfile(join(mypath, f))]

lowest = 100
f = 0
for file in files:
    if len(file) < 11:
        continue
    temp = float(file.split("_")[2].replace(".png", ""))
    if temp < lowest:
        lowest = temp
        f = file

print(lowest)
print(f)

