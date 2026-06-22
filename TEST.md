srun --account=kwf@v100 --nodes=1 --ntasks-per-node=1 --cpus-per-task=10 --gres=gpu:1 -C v100-32g --time=00:30:00 --pty bash

module purge

module load cuda/12.8.0 gcc/13.3.0

$WORK/lc0/build/release/lc0 --weights="/lustre/fsn1/projects/rech/kwf/uzr96yg/leela/networks/mobTnet-L/mobTnet-L-100000.pb.gz"

position startpos moves e2e4 c7c6 d2d4 d7d5 b1c3 d5e4 c3e4

go nodes 100

quit

exit



rsync -avP jean-zay:/lustre/fsn1/projects/rech/kwf/uzr96yg/leela/networks/mobTnet-L/mobTnet-L-100000.pb.gz ~/leela/networks/mobTnet-L/