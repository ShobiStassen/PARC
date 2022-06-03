PARC - phenotyping by accelerated refined community-partitioning
==================================================================
**PARC** is a fast, automated, combinatorial  graph-based clustering approach that integrates hierarchical graph construction (HNSW) and data-driven graph-pruning with the new Leiden community-detection algorithm. `Stassen et al. (Bioinformatics, 2020) <https://academic.oup.com/bioinformatics/article/36/9/2778/5714737>`_ *PARC:ultrafast and accurate clustering of phenotypic data of millions of single cells.*

|:eight_spoked_asterisk:| **PARC** forms the clustering basis for our new Trajectory Inference (TI) method **VIA** available on `Readthedocs <https://pyvia.readthedocs.io/en/latest/>`_ and `Github <https://github.com/ShobiStassen/VIA>`_. **VIA** is a single-cell Trajectory Inference method that offers topology construction and visualization, pseudotimes, automated prediction of terminal cell fates and temporal gene dynamics along detected lineages. **VIA can also be used to topologically visualize the graph-based connectivity of clusters found by PARC in a non-TI context.**


Example Usage on Covid-19 scRNA-seq data
-----------------------------------------
Check out the `Jupyter Notebook <https://github.com/ShobiStassen/PARC/blob/master/Covid19_Parc.ipynb>`_ for how to pre-process and PARC cluster the new Covid-19 BALF dataset by `Liao et. al 2020 <https://www.nature.com/articles/s41591-020-0901-9>`_. 
We also show how to integrate UMAP with HNSW such that the embedding in UMAP is constructed using the HNSW graph built in PARC, enabling a very fast and memory efficient viusalization (particularly noticeable when number cells > 1 Million) 

**PARC Cluster-level average gene expression**

.. raw:: html

  <img src="https://github.com/ShobiStassen/PARC/blob/master/Images/Covid_matrixplot.png?raw=true" width="220px" align="center" </a>
  <img src ="https://github.com/ShobiStassen/PARC/blob/master/Images/Covid_hnsw_umap.png?raw=true" align="center" width ="600px" </a>


**Citing**
If you find this code useful in your work, please consider citing this paper `PARC:ultrafast and accurate clustering of phenotypic data of millions of single cells <https://academic.oup.com/bioinformatics/advance-article/doi/10.1093/bioinformatics/btaa042/5714737>`_


.. toctree::
   :caption: Main
   :maxdepth: 1
   :hidden:

   Installation
   Examples
   Parameter Usage
   Notebook-covid19

