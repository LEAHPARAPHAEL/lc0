#!/bin/bash
#SBATCH --job-name=compile
#SBATCH --partition=compil          
#SBATCH --account=kwf@v100             
#SBATCH --time=00:30:00            
#SBATCH --nodes=1                     
#SBATCH --ntasks-per-node=1          
#SBATCH --cpus-per-task=8            
#SBATCH --output=compile.out
#SBATCH --error=compile.err

module purge
module load ninja/1.12.1
module load cuda/12.8.0 
module load meson/0.55.1
module load gcc/13.3.0

cd $WORK/lc0

echo "Begin compiling !"

./build.sh

echo "Compîlation over !"