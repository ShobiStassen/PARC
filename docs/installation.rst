Installation
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
