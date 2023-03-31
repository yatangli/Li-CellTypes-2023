# Li-CellTypes-2023
This repository accompanies our article:

**Li, YT and Meister, M (2023). Functional Cell Types in the Mouse Superior Colliculus. ELife.**

  
## Contents of the repo
This repo contains the data and code needed to reproduce the figures in the paper.

* `/manuscript.pdf`: Preprint version of the article on [biorxiv] (https://www.biorxiv.org/content/10.1101/2022.04.01.486789v2). All figure numbers refer to this version.
* `/data/`: raw data on which the analysis operates
* `/code/cell_types_main.py`: the main code for generating figures. Running it generates all figures. Figures can be saved to `/figures/` by setting save_fig = True.
* `/code/cell_types_utils.py`: functions that are used in `/code/cell_types_main.py`
* `/code/Notebook.ipynb`: Jupyter notebook version of `/code/cell_types_main.py`


## How to find code for a specific figure panel
* Search for "Figure nX" in `/code/cell_types_main.py` or in the Jupyter notebook, where "n" is 1,2,3..., "X" is A,B,C,...
