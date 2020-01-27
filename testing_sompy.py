import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from neupy import algorithms, environment
import numpy as np
from matplotlib import pyplot as plt
from sompy.sompy import SOMFactory
from mpl_toolkits.mplot3d import Axes3D
import somoclu
'''
data = np.array([[0.1961, 0.9806, 2],
    [-0.1961, 0.9806,-3],
    [-0.5812, -0.8137,2],
    [-0.8137, -0.5812,1],    [-0.8137, -0.5812,-1],  [-0.8137, -0.2,-1],[-0.8137, -0.2,-2],[-0.8137, 0.2,-1],[0.8137, -0.2,-1],[0.8137, -0.2,-1.56],[8.3, -0.2,-1]
    ])
print('data size', data.shape)

sm = SOMFactory().build(data, normalization = 'var', initialization='random', mapsize = (3,3))
sm.train(n_job=1, verbose=False, train_rough_len=2, train_finetune_len=5)

print(sm.find_bmu(data)[0,:])
print(sm.codebook.matrix)
bmu = np.empty((data.shape[0], 2))
#print(sm.predict_probability(data))
prod =np.dot(sm.codebook.matrix, data.T)
prod *= -2
print(prod.T)
bmu[:, 0] = np.argmin(prod, axis=0)
bmu[:, 1] = np.min(prod, axis=0)
print('bmu',bmu)
xy = sm.bmu_ind_to_xy(bmu[:,0])
print(xy[:,0:2])

sofm = algorithms.SOFM( n_inputs = 3,step = 0.1,
learning_radius = 0, features_grid=(2, 2))
sofm.train(data, epochs=100)
print(sofm.predict_raw(data))
print(sofm.predict(data))
print(sofm.weight.shape, sofm.weight)
print(np.dot(sofm.predict_raw(data),sofm.weight.T))
'''
c1 = np.random.rand(50, 3)/5
c2 = (0.6, 0.1, 0.05) + np.random.rand(50, 3)/5
c3 = (0.4, 0.1, 0.7) + np.random.rand(50, 3)/5
data = np.float32(np.concatenate((c1, c2, c3)))
print(data[1,:])
colors = ["red"] * 50
colors.extend(["green"] * 50)
colors.extend(["blue"] * 50)
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=colors)
labels = range(150)
som = somoclu.Somoclu(n_columns=4, n_rows=3, maptype="toroid",compactsupport=False, initialization='pca')
som.train(data)


surface_state = som.get_surface_state(data)
print('surface_state', surface_state.shape, surface_state[1,:])
bmus = som.get_bmus(surface_state,)

codebook = som.codebook
print('original codebook', codebook.shape, codebook)
codebook_extract = codebook.reshape(codebook.shape[0] * codebook.shape[1], codebook.shape[2])
print(codebook_extract)
print('codebook extracted', codebook_extract.shape)
#print('codebook',codebook.shape, codebook[0,0,:])
#print(som.bmus)
print('bmus', bmus.shape, bmus[1,:])
x = bmus[1,0]
y = bmus[1,1]
print('xy',x,y)
datapoint1 = data[1,:]
print(datapoint1, 'datapoint')
print(codebook[x,y,:])
index = 4*y+x
print(datapoint1, codebook_extract[index,:])