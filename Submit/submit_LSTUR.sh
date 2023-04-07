#!/bin/sh
#BSUB -J LSTURini
#BSUB -o LSTURini_%J.out
#BSUB -e LSTURini_%J.err
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 1
#BSUB -R "rusage[mem=8G]"
#BSUB -R "span[hosts=1]"
#BSUB -W 24:00
#BSUB -N
# end of BSUB options

# module load scipy/VERSION
module load scipy/1.7.3-python-3.9.11

# load CUDA (for GPU support)
module load cuda/11.7


# activate the virtual environment
# NOTE: needs to have been built with the same SciPy version above!
source venv_1/bin/activate

python3 LSTURiniHPC.py
