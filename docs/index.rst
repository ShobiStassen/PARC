PARC - phenotyping by accelerated refined community-partitioning
============================================================
**PARC** is a fast, automated, combinatorial  graph-based clustering approach that integrates hierarchical graph construction (HNSW) and data-driven graph-pruning with the new Leiden community-detection algorithm. `Stassen et al. (Bioinformatics, 2020) <https://academic.oup.com/bioinformatics/article/36/9/2778/5714737>`_ *PARC:ultrafast and accurate clustering of phenotypic data of millions of single cells.*

-:eight_spoked_asterisk:-
PARC forms the clustering basis for our new Trajectory Inference (TI) method `VIA <https://github.com/ShobiStassen/VIA>`_. VIA is a single-cell Trajectory Inference method that offers topology construction and visualization, pseudotimes, automated prediction of terminal cell fates and temporal gene dynamics along detected lineages. VIA can also be used to topologically visualize the graph-based connectivity of clusters found by PARC in a non-TI context. 

Getting Started 
----------------
**install using pip** ::
  
  conda create --name ParcEnv pip 
  pip install parc // tested on linux

           
**install by cloning repository and running setup.py** (ensure dependencies are installed)::

  git clone https://github.com/ShobiStassen/PARC.git 
  python3 setup.py install // cd into the directory of the cloned PARC folder containing setup.py and issue this command


**install dependencies separately if needed (linux)** 
If the pip install doesn't work, it usually suffices to first install all the requirements (using pip) and subsequently install parc (also using pip)::

  pip install igraph, leidenalg, hnswlib, umap-learn
  pip install parc


**Windows installation**::

  install igraph and leidenalg using binaries before calling "pip install parc"
  python-igraph: download the python36 Windows Binaries by [Gohlke](http://www.lfd.uci.edu/~gohlke/pythonlibs) 
  leidenalg: depends on python-igraph. download [windows binary](https://pypi.org/project/leidenalg/#files)

  conda create --name parcEnv python=3.6 pip
  pip install python_igraph-0.7.1.post6-cp36-cp36m-win_amd64.whl 
  pip install leidenalg-0.7.0-cp36-cp36m-win_amd64.whl
  pip install hnswlib
  pip install parc

Example Usage on Covid-19 scRNA-seq data
-----------------------------------------
Check out the *new* `Jupyter Notebook]<https://github.com/ShobiStassen/PARC/blob/master/Covid19_Parc.ipynb>`_ for how to pre-process and PARC cluster the new Covid-19 BALF dataset by `Liao et. al 2020 <https://www.nature.com/articles/s41591-020-0901-9>`_. 
We also show how to integrate UMAP with HNSW such that the embedding in UMAP is constructed using the HNSW graph built in PARC, enabling a very fast and memory efficient viusalization (particularly noticeable when n_cells > 1 Million) 

**PARC Cluster-level average gene expression**

.. raw:: html

  <img src="https://github.com/ShobiStassen/PARC/blob/master/Images/Covid_matrixplot.png?raw=true" width="220px" align="left" </a>
  <img src ="https://github.com/ShobiStassen/PARC/blob/master/Images/Covid_hnsw_umap.png" align="right" width ="580px" </a>

.. toctree::
   :caption: Main
   :maxdepth: 1
   :hidden:

   installation
