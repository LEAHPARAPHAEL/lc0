#!/bin/bash
#SBATCH --job-name=compile
#SBATCH --partition=compil          
#SBATCH --account=kwf@v100             
#SBATCH --time=00:30:00               # Augmenté pour être sûr
#SBATCH --nodes=1                     
#SBATCH --ntasks-per-node=1          
#SBATCH --cpus-per-task=8             # 8 cœurs pour Ninja !
#SBATCH --output=compile.out
#SBATCH --error=compile.err

module purge
module load ninja/1.12.1
module load cuda/12.8.0 
module load gcc/13.3.0
# On charge Python car Meson en a besoin pour tourner
module load python/3.12.7

# LA LIGNE MAGIQUE : On dit à Linux de chercher Meson dans vos fichiers personnels d'abord
export PATH="$HOME/.local/bin:$PATH"

cd $WORK/lc0

echo "Begin compiling !"

./build.sh release 

echo "Compilation over !"