# PARC
PARC, “phenotyping by accelerated refined community-partitioning” - is a fast, automated, combinatorial  graph-based clustering approach that integrates hierarchical graph construction (HNSW) and data-driven graph-pruning with the new Leiden community-detection algorithm.
## Install Dependencies
### Install leidenalg, igraph and hnswlib
```
pip install leidenalg igraph 
```
```
git clone https://github.com/nmslib/hnswlib
apt-get install -y python-setuptools python-pip
pip install pybind11 numpy setuptools
cd python_bindings // 'python bindings' is a folder within the cloned repository
python3 setup.py install
```

  
## Example Usage - IRIS dataset 

```
import PARC
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sklearn import datasets

// load sample IRIS data
iris = datasets.load_iris()
X = iris.data // data
y=iris.target // labels

plt.scatter(X[:,0],X[:,1], c = y) // colored by 'ground truth'

parc = PARC(X,y) // instantiate PARC
parc.run_clustering() // run the clustering

# View scatterplot colored by PARC labels
plt.show()
plt.scatter(X[:, 0], X[:, 1], c=p1.labels)
plt.show()

```
## Example Usage: 10X PBMC (Zheng et al., 2017)

```
import PARC
## load data (50 PCs of filtered gene matrix pre-processed as per Zheng et al. 2017)
X = open() // load file
y = open() // load annotations

parc = PARC(X,y) // instantiate PARC
parc.run_clustering() // run the clustering
parc_labels = parc.labels 
```
![](Images/10X_PBMC_PARC_andGround.png) tsne plot of annotations and PARC clustering

## References to dependencies 
- Leiden (pip install leidenalg) (V.A. Traag, 2019 doi.org/10.1038/s41598-019-41695-z)
- hsnwlib Malkov, Yu A., and D. A. Yashunin. "Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small   World graphs." TPAMI, preprint: https://arxiv.org/abs/1603.09320
- igraph (igraph.org/python/)
