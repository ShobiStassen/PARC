# PARC
PARC, “phenotyping by accelerated refined community-partitioning” - is a fast, automated, combinatorial  graph-based clustering approach that integrates hierarchical graph construction (HNSW) and data-driven graph-pruning with the new Leiden community-detection algorithm.
## Dependencies

- pip install leidenalg igraph 
- Leiden (pip install leidenalg) (V.A. Traag, 2019 doi.org/10.1038/s41598-019-41695-z)
- install hnswlib with the following commands: 
  git clone [hsnwlib URL]
  apt-get install -y python-setuptools python-pip
  pip install pybind11 numpy setuptools
  cd python_bindings // 'python bindings' is a folder within the cloned repository
  python3 setup.py install

References to dependencies
- igraph (igraph.org/python/)
- HNSW header only Python Bindings (arxiv.org/abs/1603.09320) (github.com/nmslib/hnswlib)

## Example Usage
