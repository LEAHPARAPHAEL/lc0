meson setup --reconfigure build 
ninja -C build

./lc0 --weights=/home/raph/leela/networks/8x256x8h/8x256x8h-150000.pb.gz
./lc0 --weights=/home/raph/leela/networks/8x256x8h-crbk/8x256x8h-crbk-150000.pb.gz

./build.sh release -Dcc_cuda=120
./build.sh release -Dcc_cuda=89