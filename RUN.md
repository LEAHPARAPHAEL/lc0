# Build the project
meson setup --reconfigure build 
ninja -C build 
./build.sh release 


position startpos moves e2e4 c7c6 d2d4 d7d5 b1c3 d5e4 c3e4

# Test
./build/release/lc0 --weights=/home/raph/leela/networks/Mx2-Tx8/Mx2-Tx8-150000.pb.gz 

./engines/eps-6 --weights=/home/raph/leela/networks/Mx2-Tx8/Mx2-Tx8-150000.pb.gz

./build/release/lc0 backendbench --weights=/home/raph/leela/networks/Mx2-Tx8/Mx2-Tx8-150000.pb.gz 

./build/release/lc0 --weights=/home/raph/leela/networks/Tx8/Tx8-150000.pb.gz 

./build/release/lc0 backendbench --weights=/home/raph/leela/networks/Tx8/Tx8-150000.pb.gz 

./build/release/lc0 --weights=/home/raph/leela/networks/Tx8-LNDW-deep/Tx8-LNDW-deep-150000.pb.gz 

./build/release/lc0 backendbench --weights=/home/raph/leela/networks/Tx8-LNDW-deep/Tx8-LNDW-deep-150000.pb.gz

./build/release/lc0 --weights=/home/raph/leela/networks/Mx4-Tx6/Mx4-Tx6-150000.pb.gz 

./build/release/lc0 backendbench --weights=/home/raph/leela/networks/Mx4-Tx6/Mx4-Tx6-150000.pb.gz 

./build/release/lc0 --weights=/home/raph/leela/networks/Mx6-Tx4/Mx6-Tx4-150000.pb.gz

./build/release/lc0 backendbench --weights=/home/raph/leela/networks/Mx6-Tx4/Mx6-Tx4-150000.pb.gz

./build/release/lc0 --weights=/home/raph/leela/networks/MMTMMTMMT-cuda/MMTMMTMMT-cuda-150000.pb.gz

./build/release/lc0 backendbench --weights=/home/raph/leela/networks/MMTMMTMMT-cuda/MMTMMTMMT-cuda-150000.pb.gz

./build/release/lc0 --weights=/home/raph/leela/networks/Mx6-Tx4-768/Mx6-Tx4-768-150000.pb.gz

./build/release/lc0 backendbench --weights=/home/raph/leela/networks/Mx6-Tx4-768/Mx6-Tx4-768-150000.pb.gz

./build/release/lc0 --weights=/home/raph/leela/networks/Tx8-pre/Tx8-pre-150000.pb.gz

./build/release/lc0 backendbench --weights=/home/raph/leela/networks/Tx8-pre/Tx8-pre-150000.pb.gz


# Comparison 5x5 VS 3x3 VS custom

./engines/custom backendbench --weights=/home/raph/leela/networks/MMTx3/MMTx3-100000.pb.gz

./engines/custom backendbench --weights=/home/raph/leela/networks/MMTx3/MMTx3-100000.pb.gz -o "custom_depthwise=false"

./engines/dense_3x3 backendbench --weights=/home/raph/leela/networks/MMTx3-3x3/MMTx3-3x3-100000.pb.gz -o "custom_depthwise=false"

./engines/dense_3x3 backendbench --weights=/home/raph/leela/networks/MMTx3-3x3/MMTx3-3x3-100000.pb.gz

./engines/dense_5x5 backendbench --weights=/home/raph/leela/networks/MMTx3-dense/MMTx3-dense-100000.pb.gz

./engines/dense_5x5 backendbench --weights=/home/raph/leela/networks/MMTx3-dense/MMTx3-dense-100000.pb.gz -o "custom_depthwise=false"

./engines/dense_5x5 backendbench --weights=/home/raph/leela/networks/MMTx3/MMTx3-100000.pb.gz








# CUDNN

wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb

# 2. Install the keyring
sudo dpkg -i cuda-keyring_1.1-1_all.deb

# 3. Update apt to read the new repository
sudo apt-get update

wget https://developer.download.nvidia.com/compute/cudnn/9.6.0/local_installers/cudnn-local-repo-ubuntu2404-9.6.0_1.0-1_amd64.deb
sudo dpkg -i cudnn-local-repo-ubuntu2404-9.6.0_1.0-1_amd64.deb
sudo cp /var/cudnn-local-repo-ubuntu2404-9.6.0/cudnn-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cudnn9-cuda-12

cd ~
git clone https://github.com/NVIDIA/cudnn-frontend.git

# 2. Copy ALL files inside the include directory (the critical fix)
sudo cp -r cudnn-frontend/include/* /usr/local/cuda/include/

# 3. Clean up the cloned repo
rm -rf cudnn-frontend

# 4. Navigate back to your LC0 folder and resume the build!
cd ~/lc0
ninja -C build/release

With op_nhcw = false
