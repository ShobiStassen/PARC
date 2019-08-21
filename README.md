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

  
## Example Usage - IRIS and Digits dataset from sklearn

```
import PARC as parc
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sklearn import datasets

// load sample IRIS data
iris = datasets.load_iris()
X = iris.data // data
y=iris.target // labels

plt.scatter(X[:,0],X[:,1], c = y) // colored by 'ground truth'

Parc1 = parc.PARC(X,y) // instantiate PARC
Parc1.run_PARC() // run the clustering
parc_labels = Parc1.labels

# View scatterplot colored by PARC labels
plt.show()
plt.scatter(X[:, 0], X[:, 1], c=parc_labels)
plt.show()

// load sample digits data
digits = datasets.load_digits()
X = digits.data
y = digits.target
Parc2 = parc.PARC(X,y, jac_std_global='median') // 'median' is default pruning level
Parc2.run_PARC()
parc_labels = Parc2.labels

```
## Example Usage: 10X PBMC (Zheng et al., 2017)
[pre-processed datafile](https://drive.google.com/file/d/1H4gOZ09haP_VPCwsYxZt4vf3hJ1GZj3b/view?usp=sharing)

[annotations](Datasets/annotations_zhang.txt)

```
import PARC
import csv

## load data (50 PCs of filtered gene matrix pre-processed as per Zheng et al. 2017)

X = csv.reader(open("/home/shobi/Thesis/Rcode/pca50_pbmc68k.txt", 'rt'),delimiter = ",")
X = list(X)

y = [] // annotations
with open('/annotations_zhang.txt', 'rt') as f: 
    for line in f: y.append(line.strip().replace('\"', ''))

parc1 = parc.PARC(X,y) // instantiate PARC
parc1.run_PARC() // run the clustering
parc_labels = parc1.labels 
```
![](Images/10X_PBMC_PARC_andGround.png) tsne plot of annotations and PARC clustering

## Example Usage with Scanpy: 10X PBMC (Zheng et al., 2017)

[raw datafile](https://github.com/10XGenomics/single-cell-3prime-paper/tree/master/pbmc68k_analysis)

```
import scanpy.api as sc
//load data
path = './data/zheng17_filtered_matrices_mex/hg19/'
adata = sc.read(path + 'matrix.mtx', cache=True).T  # transpose the data
adata.var_names = pd.read_csv(path + 'genes.tsv', header=None, sep='\t')[1]
adata.obs_names = pd.read_csv(path + 'barcodes.tsv', header=None)[0]

// annotations as per correlation with pure samples
adata.obs['annotations'] = pd.read_csv('./data/zheng17_annotations.txt', header=None)[0].values
annotations_list = annotationsXXXXXX

//pre-process as per Zheng et al., and take first 50 PCs for analysis
sc.pp.recipe_zheng17(adata)
sc.tl.pca(adata, n_comps=50)
parc1 = parc.PARC(adata2.obsm['X_pca'], annotations_list)
parc_labels = parc1.labels
adata2.obs["PARC"] = pd.Categorical(parc_labels)

//visualize
sc.pl.umap(adata, color='annotations')
sc.pl.umap(adata, color='PARC')
```

## References to dependencies 
- Leiden (pip install leidenalg) (V.A. Traag, 2019 doi.org/10.1038/s41598-019-41695-z)
- hsnwlib Malkov, Yu A., and D. A. Yashunin. "Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small   World graphs." TPAMI, preprint: https://arxiv.org/abs/1603.09320
- igraph (igraph.org/python/)
