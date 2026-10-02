#!/bin/bash
echo "inner HOME=$HOME"
echo "inner shell=$0 PS1=[${PS1:-<unset>}]"
source ~/.bashrc > /gpfs/home/mmahaut/projects/parrots/logs/diag_bashrc_src.log 2>&1
echo "bashrc_exit=$?"
type conda
conda activate parr
echo "activate_exit=$?"
which python
python -c "import numpy; print('numpy ok', numpy.__version__)"
