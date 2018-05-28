'''
Created on 23 May, 2018

@author: shobi
https://github.com/lferry007/LargeVis

-fea: specify whether the input file is high-dimensional feature vectors (1) or networks (0). Default is 1.
-threads: Number of threads. Default is 8.
-outdim: The lower dimensionality LargesVis learns for visualization (usually 2 or 3). Default is 2.
-samples: Number of edge samples for graph layout (in millions). Default is set to data size / 100 (million).
-prop: Number of times for neighbor propagations in the state of K-NNG construction, usually less than 3. Default is 3.
-alpha: Initial learning rate. Default is 1.0.
-trees: Number of random-projection trees used for constructing K-NNG. 50 is sufficient for most cases unless you are dealing with very large datasets (e.g. data size over 5 million), and less trees are suitable for smaller datasets. Default is set according to the data size.
-neg: Number of negative samples used for negative sampling. Default is 5.
-neigh: Number of neighbors (K) in K-NNG, which is usually set as three times of perplexity. Default is 150.
-gamma: The weights assigned to negative edges. Default is 7.
-perp: The perplexity used for deciding edge weights in K-NNG. Default is 50.

'''

import time
import numpy as np
import LargeVis
print('start time is:', time.ctime())
time_start = time.time()

outdim=2
threads=8
samples = -1
prop =-1
alpha=-1
trees = -1
neg = -1
neigh = -1
gamma = -1
perp =30
fea = 1
input = np.random.rand(50,10)

def format_LVinput(input_array):
    num_samples = input_array.shape[0]
    num_dim = input_array.shape[1]
    ll= input_array.tolist()
    '''
    ll = [] #list of lists
    for row in input_array:
        ll.append(list(row))
    if num_dim != len(ll[0]): print('error: the num dimensions do not match')
    if num_samples != len(ll): print('error: the num of samples do not match')
    print(len(ll),len(ll[0]))
    '''
    return ll
def format_LVoutput(output_list):
    output_array = np.array(output_list)
    return output_array

LVinput = format_LVinput(input)
LargeVis.loaddata(LVinput)
Y = LargeVis.run(outdim, threads, samples, prop, alpha, trees, neg, neigh, gamma, perp)
Y_array = format_LVoutput(Y)
print(Y_array.shape)
print(time.ctime())
