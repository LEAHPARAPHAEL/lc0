# Build the project
meson setup --reconfigure build 
ninja -C build 
./build.sh release -Dcudnn=true 


position startpos moves e2e4 c7c6 d2d4 d7d5 b1c3 d5e4 c3e4

# Test
./build/release/lc0 --weights=/home/raph/leela/networks/8x384x12h-2rbk-relu/8x384x12h-2rbk-relu-150000.pb.gz -o custom_depthwise=true,nhwc=true

./build/release/lc0 backendbench --weights=/home/raph/leela/networks/8x384x12h-2rbk-relu/8x384x12h-2rbk-relu-150000.pb.gz -o custom_depthwise=true,nhwc=true

./build/release/lc0 --weights=/home/raph/leela/networks/8x384x12h-2rbk-256/8x384x12h-2rbk-256-150000.pb.gz -o custom_depthwise=true,nhwc=true

./build/release/lc0 backendbench --weights=/home/raph/leela/networks/8x384x12h-2rbk-256/8x384x12h-2rbk-256-150000.pb.gz -o custom_depthwise=true,nhwc=true

./build/release/lc0 --weights=/home/raph/leela/networks/Mx2-Tx8-balanced/Mx2-Tx8-balanced-150000.pb.gz -o custom_depthwise=true,nhwc=true

./build/release/lc0 backendbench --weights=/home/raph/leela/networks/Mx2-Tx8-balanced/Mx2-Tx8-balanced-150000.pb.gz -o custom_depthwise=true,nhwc=true

./build/release/lc0 --weights=/home/raph/leela/networks/Tx8/Tx8-150000.pb.gz -o custom_depthwise=true,nhwc=true

./build/release/lc0 --weights=/home/raph/leela/networks/Tx8-eps-5/Tx8-eps-5-150000.pb.gz -o custom_depthwise=true,nhwc=true

./build/release/lc0 backendbench --weights=/home/raph/leela/networks/Tx8/Tx8-150000.pb.gz -o custom_depthwise=true,nhwc=true

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

[DEBUG] OUTPUT OF BLOCK 0 | First 5 floats: 1.2832 1.28418 1.23145 1.23145 1.23145
        Sum: 31049.5 | Var: 76770.2

[DEBUG] OUTPUT OF BLOCK 1 | First 5 floats: 1.14941 1.15039 1.09375 1.09375 1.09375
        Sum: 46306.1 | Var: 157883

[DEBUG] OUTPUT OF BLOCK 2 | First 5 floats: -0.0662842 0.0173645 -0.991211 -0.322266 0.406982
        Sum: 914.237 | Var: 7801.42

[DEBUG] OUTPUT OF BLOCK 3 | First 5 floats: -0.660645 0.76416 -1.84375 -0.0689697 0.699219
        Sum: 337.655 | Var: 62649.9

[DEBUG] OUTPUT OF BLOCK 4 | First 5 floats: 0.0151978 0.0489197 0.1203 0.1203 0.1203
        Sum: 5213.36 | Var: 7447.4

[DEBUG] OUTPUT OF BLOCK 5 | First 5 floats: -0.274902 -0.243408 -0.240479 -0.230835 -0.231079
        Sum: 3760.27 | Var: 9007.74

[DEBUG] OUTPUT OF BLOCK 6 | First 5 floats: 0.116211 0.178711 -0.370605 -0.23999 0.496582
        Sum: -1893.5 | Var: 27162.9

[DEBUG] OUTPUT OF BLOCK 7 | First 5 floats: -0.213745 -0.143433 -0.130127 0.174316 0.067627
        Sum: -1098.55 | Var: 13820.2

[DEBUG] FINAL BACKBONE OUTPUT | First 5 floats: -0.213745 -0.143433 -0.130127 0.174316 0.067627
        Sum: -1098.55 | Var: 13820.2






[DEBUG] OUTPUT OF BLOCK 0 | First 5 floats: -0.0509949 -0.0306702 -0.0456238 -0.0456238 -0.0456238
        Sum: 16519.6 | Var: 18550.4

[DEBUG] OUTPUT OF BLOCK 1 | First 5 floats: 0.125854 0.108154 0.101929 0.0873413 0.0871582
        Sum: 16779.8 | Var: 33223.3

[DEBUG] OUTPUT OF BLOCK 2 | First 5 floats: 0.130737 0.0768433 -0.276367 0.01091 0.187378
        Sum: 637.924 | Var: 6259.56

[DEBUG] OUTPUT OF BLOCK 3 | First 5 floats: -0.125244 0.760742 -1.18555 0.0974121 -0.0210266
        Sum: -91.9457 | Var: 66912.8

[DEBUG] OUTPUT OF BLOCK 4 | First 5 floats: 0.0151978 0.0489197 0.1203 0.1203 0.1203
        Sum: 5213.36 | Var: 7447.4

[DEBUG] OUTPUT OF BLOCK 5 | First 5 floats: -0.274902 -0.243408 -0.240479 -0.230835 -0.231079
        Sum: 3760.27 | Var: 9007.74

[DEBUG] OUTPUT OF BLOCK 6 | First 5 floats: 0.116211 0.178711 -0.370605 -0.23999 0.496582
        Sum: -1893.5 | Var: 27162.9

[DEBUG] OUTPUT OF BLOCK 7 | First 5 floats: -0.213745 -0.143433 -0.130127 0.174316 0.067627
        Sum: -1098.55 | Var: 13820.2

[DEBUG] FINAL BACKBONE OUTPUT | First 5 floats: -0.213745 -0.143433 -0.130127 0.174316 0.067627
        Sum: -1098.55 | Var: 13820.2