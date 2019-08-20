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
## References to dependencies 
- Leiden (pip install leidenalg) (V.A. Traag, 2019 doi.org/10.1038/s41598-019-41695-z)
- hsnwlib Malkov, Yu A., and D. A. Yashunin. "Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small   World graphs." TPAMI, preprint: https://arxiv.org/abs/1603.09320
- igraph (igraph.org/python/)
  
## Example Usage
