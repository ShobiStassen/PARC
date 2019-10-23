# PARC
PARC, “phenotyping by accelerated refined community-partitioning” - is a fast, automated, combinatorial  graph-based clustering approach that integrates hierarchical graph construction (HNSW) and data-driven graph-pruning with the new Leiden community-detection algorithm.

## Getting Started
### install using pip
```
conda create --name ParcEnv pip 
pip install parc // tested on linux
```
### install by cloning repository and running setup.py (ensure dependencies are installed)
```
git clone https://github.com/ShobiStassen/PARC.git 
python3 setup.py install // cd into the directory of the cloned PARC folder containing setup.py and issue this command
```

### install dependencies separately if needed (linux)
```
pip install python-igraph, leidenalg, hnswlib
pip install parc
```

### Windows installation

install python-igraph and leidenalg using binaries before calling "pip install parc"
python-igraph: download the python36 Windows Binaries by [Gohlke](http://www.lfd.uci.edu/~gohlke/pythonlibs) 
leidenalg: depends on python-igraph. download [windows binary](https://pypi.org/project/leidenalg/#files) available for python3.6 only

```
conda create --name parcEnv python=3.6 pip
pip install python_igraph-0.7.1.post6-cp36-cp36m-win_amd64.whl 
pip install leidenalg-0.7.0-cp36-cp36m-win_amd64.whl
pip install hnswlib
pip install parc
```
 
## Example Usage 1. (small test sets) - IRIS and Digits dataset from sklearn

```
import parc
import matplotlib.pyplot as plt
from sklearn import datasets

// load sample IRIS data
//data (n_obs x k_dim, 150x4)
iris = datasets.load_iris()
X = iris.data
y=iris.target

plt.scatter(X[:,0],X[:,1], c = y) // colored by 'ground truth'
plt.show()

Parc1 = parc.PARC(X,true_label=y) // instantiate PARC
//Parc1 = parc.PARC(X) // when no 'true labels' are available
Parc1.run_PARC() // run the clustering
parc_labels = Parc1.labels

# View scatterplot colored by PARC labels

plt.scatter(X[:, 0], X[:, 1], c=parc_labels)
plt.show()

// load sample digits data
digits = datasets.load_digits()
X = digits.data // (n_obs x k_dim, 1797x64) 
y = digits.target
Parc2 = parc.PARC(X,true_label=y, jac_std_global='median') // 'median' is default pruning level
Parc2.run_PARC()
parc_labels = Parc2.labels

```
## Example Usage 2. (mid-scale scRNA-seq): 10X PBMC (Zheng et al., 2017)
[pre-processed datafile](https://drive.google.com/file/d/1H4gOZ09haP_VPCwsYxZt4vf3hJ1GZj3b/view?usp=sharing)

[annotations](Datasets/annotations_zhang.txt)

```
import parc
import csv

## load data (50 PCs of filtered gene matrix pre-processed as per Zheng et al. 2017)

X = csv.reader(open("'./pca50_pbmc68k.txt", 'rt'),delimiter = ",")
X = np.array(list(X)) // (n_obs x k_dim, 68579 x 50)
X = X.astype("float")
// OR with pandas as: X = pd.read_csv("'./pca50_pbmc68k.txt").values.astype("float")

y = [] // annotations
with open('/annotations_zhang.txt', 'rt') as f: 
    for line in f: y.append(line.strip().replace('\"', ''))
// OR with pandas as: y =  list(pd.read_csv('./data/zheng17_annotations.txt', header=None)[0])   


parc1 = parc.PARC(X,true_label=y,jac_std_global=0.15) // instantiate PARC
parc1.run_PARC() // run the clustering
parc_labels = parc1.labels 
```
![](Images/10X_PBMC_PARC_andGround.png) tsne plot of annotations and PARC clustering

## Example Usage 3. 10X PBMC (Zheng et al., 2017) integrating Scanpy pipeline

[raw datafile](https://github.com/10XGenomics/single-cell-3prime-paper/tree/master/pbmc68k_analysis)

```
pip install scanpy
```

```
import scanpy.api as sc
import pandas as pd
//load data
path = './data/zheng17_filtered_matrices_mex/hg19/'
adata = sc.read(path + 'matrix.mtx', cache=True).T  # transpose the data
adata.var_names = pd.read_csv(path + 'genes.tsv', header=None, sep='\t')[1]
adata.obs_names = pd.read_csv(path + 'barcodes.tsv', header=None)[0]

// annotations as per correlation with pure samples
annotations = list(pd.read_csv('./data/zheng17_annotations.txt', header=None)[0])
adata.obs['annotations'] = pd.Categorical(annotations)

//pre-process as per Zheng et al., and take first 50 PCs for analysis
sc.pp.recipe_zheng17(adata)
sc.tl.pca(adata, n_comps=50)
parc1 = parc.PARC(adata2.obsm['X_pca'], true_label = annotations)
parc_labels = parc1.labels
adata2.obs["PARC"] = pd.Categorical(parc_labels)

//visualize
sc.pl.umap(adata, color='annotations')
sc.pl.umap(adata, color='PARC')
```
## Example Usage 4. Large-scale (70K subset and 1.1M cells) Lung Cancer cells (multi-ATOM imaging cytometry based features)

[normalized image-based feature matrix 70K cells](https://drive.google.com/open?id=1LeFjxGlaoaZN9sh0nuuMFBK0bvxPiaUz)

[Lung Cancer cells annotation 70K cells](https://drive.google.com/open?id=1iwXQkdwEwplhZ1v0jYWnu2CHziOt_D9C)

[1.1M cell features and annotations](https://data.mendeley.com/datasets/nnbfwjvmvw/draft?a=dae895d4-25cd-4bdf-b3e4-57dd31c11e37)

```
import parc
import pandas as pd

// load data: digital mix of 7 cell lines from 7 sets of pure samples (1.1M cells x 26 features)
X = pd.read_csv("'./LungData.txt").values.astype("float") 
y = list(pd.read_csv('./LungData_annotations.txt', header=None)[0]) // list of cell-type annotations

// run PARC
parc1 = parc.PARC(X, true_label=y)
parc_labels = parc1.labels

```
![](Images/70K_Lung_github_overview.png) tsne plot of annotations and PARC clustering, heatmap of features

### Parameters and attributes

| Input Parameter | Description |
| ---------- |----------|
| `data` | (numpy.ndarray) n_samples x n_features |
| `true_label` | (numpy.ndarray) (optional)|
| `dist_std_local` |  (optional, default = 2) local pruning threshold: the number of standard deviations above the mean minkowski distance between neighbors of a given node. the higher the parameter, the more edges are retained|
| `jac_std_global` |  (optional, default = 'median') global level  graph pruning. This threshold can also be set as the number of standard deviations below the network's mean-jaccard-weighted edges. 0.1-1 provide reasonable pruning. higher value means less pruning. e.g. a value of 0.15 means all edges that are above mean(edgeweight)-0.15*std(edge-weights) are retained. We find both 0.15 and 'median' to yield good results resulting in pruning away ~ 50-60% edges |
| `dist_std_local` |  (optional, default = 2) local pruning threshold: the number of standard deviations above the mean minkowski distance between neighbors of a given node. higher value means less pruning|
| `partition_seed` |  (optional, default = 10) The random seed to pass to Leiden|

| Attributes | Description |
| ---------- |----------|
| `labels` | (list) length n_samples of corresponding cluster labels |
| `f1_mean` | (list) f1 score (not weighted by population). For details see supplementary section of [paper](https://doi.org/10.1101/765628) |
| `stats_df` | (DataFrame) stores parameter values and performance metrics |


## References to dependencies 
- Leiden (pip install leidenalg) (V.A. Traag, 2019 doi.org/10.1038/s41598-019-41695-z)
- hsnwlib Malkov, Yu A., and D. A. Yashunin. "Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small   World graphs." TPAMI, preprint: https://arxiv.org/abs/1603.09320
- igraph (igraph.org/python/)

## Citing
If you find this code useful in your work, please consider citing this paper [PARC:ultrafast and accurate clustering of phenotypic data of millions of single cells](https://doi.org/10.1101/765628)
