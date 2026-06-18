#!/bin/bash
#SBATCH --job-name=lc0-backend
#SBATCH --account=kwf@v100
#SBATCH --time=00:03:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH -C v100-32g
#SBATCH --output=logs/default_job.out
#SBATCH --error=logs/default_job.err

module purge
module load ninja/1.12.1
module load cuda/12.8.0 
module load gcc/13.3.0
# 1. Capture model name from the first argument passed to sbatch
MODEL_NAME=$1

if [ -z "$MODEL_NAME" ]; then
    echo "Error: No model name provided!"
    echo "Usage: sbatch backendbench.sh <model_name>"
    exit 1
fi

NETWORKS_DIR="/lustre/fsn1/projects/rech/kwf/uzr96yg/leela/networks/$MODEL_NAME"
LC0_EXEC="./build/release/lc0"
LOG_DIR="logs/$MODEL_NAME"

echo "Targeting Model Directory: $NETWORKS_DIR"

# 2. Use an inline Python snippet to find the latest step count file or fallback
SELECTED_WEIGHT=$(python3 -c "
import os, re, sys

folder = '$NETWORKS_DIR'
if not os.path.exists(folder):
    sys.exit(1)

files = [f for f in os.listdir(folder) if f.endswith('.pb.gz')]
if not files:
    sys.exit(2)

numeric_files = []
for f in files:
    match = re.search(r'(\d+)\.pb\.gz$', f)
    if match:
        numeric_files.append((int(match.group(1)), f))

if numeric_files:
    numeric_files.sort(key=lambda x: x[0], reverse=True)
    print(os.path.join(folder, numeric_files[0][1]))
else:
    files.sort()
    print(os.path.join(folder, files[0]))
" 2>/dev/null)

# 3. Handle errors based on Python script exit status
exit_status=$?
if [ $exit_status -eq 1 ]; then
    echo "Error: Directory '$NETWORKS_DIR' does not exist."
    exit 1
elif [ $exit_status -eq 2 ] || [ -z "$SELECTED_WEIGHT" ]; then
    echo "Error: No '.pb.gz' weight files found in $NETWORKS_DIR."
    exit 1
fi

# 4. Verify that the lc0 executable is built and present
if [ ! -f "$LC0_EXEC" ]; then
    echo "Error: lc0 binary not found at $LC0_EXEC."
    exit 1
fi

# 5. Create the dedicated log directory if it doesn't exist
echo "Preparing logging directory at: $LOG_DIR"
mkdir -p "$LOG_DIR"

echo "Selected Weights: $SELECTED_WEIGHT"
echo "Running backendbench (Outputs redirected to $LOG_DIR/)"

# 6. Run the backendbench and cleanly separate stdout and stderr streams
$LC0_EXEC backendbench --weights="$SELECTED_WEIGHT" > "$LOG_DIR/backendbench.out" 2> "$LOG_DIR/backendbench.err"

echo "Benchmark run completed. Check outputs in $LOG_DIR/"