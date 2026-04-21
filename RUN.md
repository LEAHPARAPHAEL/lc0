meson setup --reconfigure build 
ninja -C build

./lc0 --weights=/home/raph/leela/networks/8x256-256_no_mask/8x256-256_no_mask-90000.pb.gz