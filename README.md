# Li-CellTypes-2023
This repository is for our article:

**Li, YT and Meister, M. (2023). Functional Cell Types in the Mouse Superior Colliculus. ELife.**
  
## Contents of the repo
This repo contains the data and code needed to reproduce the figures in the paper.
* `/data/`: store data
* `/code/cell_types_main.py`: the main code for generatting figures. Running it generates all figures. Figures can be saved to `/figures/` by setting save_fig = True.
* `/code/cell_types_utils.py`: functions that are used in `/code/cell_types_main.py`


## How to find code for a specific figure panel
* Search "Figure nX" in `/code/cell_types_main.py`, where "n" is 1,2,3..., "X" is A,B,C,...
