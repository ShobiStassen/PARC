# PARC
PARC, “phenotyping by accelerated refined community-partitioning” - is a fast, automated, combinatorial  graph-based clustering approach that integrates hierarchical graph construction (HNSW) and data-driven graph-pruning with the new Leiden community-detection algorithm.
## Dependencies
- ensure you have neccesary C++ compilers
- pip install leidenalg igraph 
- Leiden (pip install leidenalg) (V.A. Traag, 2019 doi.org/10.1038/s41598-019-41695-z)
- igraph (igraph.org/python/) (pip install igraph)
- HNSW header only Python Bindings (arxiv.org/abs/1603.09320) (github.com/nmslib/hnswlib)
