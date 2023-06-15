#!/bin/sh
#BSUB -J Trans
#BSUB -o CTNR_%J.out
#BSUB -e CTNR_%J.err
#BSUB -q gpua100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 1
#BSUB -R "rusage[mem=16G]"
#BSUB -R "span[hosts=1]"
#BSUB -W 72:00
#BSUB -N
#BSUB -B
# end of BSUB options

# module load scipy/VERSION
module load python3/3.9.11

# load CUDA (for GPU support)
module load cuda/11.7


# activate the virtual environment
# NOTE: needs to have been built with the same SciPy version above!
source venv_1/bin/activate

python3 TrainCTNRBert/FitCTNR.py
