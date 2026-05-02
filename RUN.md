# Build the project
meson setup --reconfigure build 
ninja -C build 
./build.sh release -Dcudnn=true 


rm -rf build/release
./build.sh release -Dcudnn=true


position startpos moves e2e4 c7c6 d2d4 d7d5 b1c3 d5e4 c3e4

# Test
./build/lc0 --weights=/home/raph/leela/networks/8x256x8h/8x256x8h-150000.pb.gz
./build/release/lc0 --weights=/home/raph/leela/networks/8x256x8h-crbk/8x256x8h-crbk-150000.pb.gz

# Backendbench
./build/lc0 backendbench --weights=/home/raph/leela/networks/8x256x8h/8x256x8h-150000.pb.gz
./build/lc0 backendbench --weights=/home/raph/leela/networks/8x256x8h-crbk/8x256x8h-crbk-150000.pb.gz
./build/lc0 backendbench --weights=/home/raph/leela/networks/8x384x12h/8x384x12h-150000.pb.gz
./build/release/lc0 backendbench --weights=/home/raph/leela/networks/8x384x12h-4res/8x384x12h-4res-10000.pb.gz
./build/release/lc0 backendbench --weights=/home/raph/leela/networks/8x384x12h-2res/8x384x12h-2res-50000.pb.gz
./build/release/lc0 backendbench --weights=/home/raph/leela/networks/8x384x12h-2rbk-256/8x384x12h-2rbk-256-70000.pb.gz

./build/release/lc0 --weights=/home/raph/leela/networks/8x384x12h-2rbk-256/8x384x12h-2rbk-256-150000.pb.gz

./build/release/lc0 backendbench --weights=/home/raph/leela/networks/8x384x12h-2res-256/8x384x12h-2res-256-118000.pb.gz

./build/release/lc0 --weights=/home/raph/leela/networks/8x384x12h-2res-256/8x384x12h-2res-256-118000.pb.gz

./build/release/lc0 --weights=/home/raph/leela/networks/8x384x12h-256/8x384x12h-256-118000.pb.gz

./build/release/lc0 backendbench --weights=/home/raph/leela/networks/8x384x12h-256/8x384x12h-256-118000.pb.gz


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