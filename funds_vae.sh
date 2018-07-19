#!/bin/bash
#----------------------------------------------------
#SBATCH -J funds_vae           # Job name
#SBATCH -o funds_vae.o%j       # Name of stdout output file
#SBATCH -e funds_vae.e%j       # Name of stderr error file
#SBATCH -p development		   # Queue (partition) name
#SBATCH -N 1              		# Total # of nodes (must be 1 for serial)
#SBATCH -n 1              	 	# Total # of mpi tasks (should be 1 for serial)
#SBATCH -t 2:00:00        		# Run time (hh:mm:ss)
#SBATCH --mail-user=poulos@berkeley.edu
#SBATCH --mail-type=all    		# Send email at begin and end of job

python code/train_vae.py 0 5000 'west-revpc' 'treated-gans'
python code/train_vae.py 1 5000 'west-revpc' 'control'

python code/train_vae.py 0 5000 'south-revpc' 'treated-gans'
python code/train_vae.py 1 5000 'south-revpc' 'control'

python code/train_vae.py 0 5000 'west-exppc' 'treated-gans'
python code/train_vae.py 1 5000 'west-exppc' 'control'

python code/train_vae.py 0 5000 'south-exppc' 'treated-gans'
python code/train_vae.py 1 5000 'south-exppc' 'control'

python code/train_vae.py 0 5000 'west-educpc' 'treated-gans'
python code/train_vae.py 1 5000 'west-educpc' 'control'

python code/train_vae.py 0 5000 'south-educpc' 'treated-gans'
python code/train_vae.py 1 5000 'south-educpc' 'control'