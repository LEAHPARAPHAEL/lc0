# 8x384x12h-2rbk-relu

fastchess -event 'Custom VS CUDNN 60s' -engine name='CUDNN' args='-w /home/raph/leela/networks/8x256x8h-2rbk/8x256x8h-2rbk-150000.pb.gz' cmd=./build/release/lc0 -engine name='Custom NCHW' args='-w /home/raph/leela/networks/8x256x8h-2rbk/8x256x8h-2rbk-150000.pb.gz -o custom_depthwise=true' cmd=./build/release/lc0 -openings file='/home/raph/leela/books/book-6-ply-unbalanced.pgn' format=pgn order=sequential -each tc=60+0.1 -rounds 100 -concurrency 1 -pgnout file=/home/raph/leela/pgns/custom_depthwise-60s -config outname=/home/raph/leela/configs/custom_depthwise-60s.json

fastchess -event 'Custom VS CUDNN 1000 nodes' -engine name='CUDNN' args='-w /home/raph/leela/networks/8x384x12h-2rbk-relu/8x384x12h-2rbk-relu-150000.pb.gz' cmd=./build/release/lc0 -engine name='Custom NCHW' args='-w /home/raph/leela/networks/8x384x12h-2rbk-relu/8x384x12h-2rbk-relu-150000.pb.gz -o custom_depthwise=true' cmd=./build/release/lc0 -openings file='/home/raph/leela/books/book-6-ply-unbalanced.pgn' format=pgn order=sequential -each nodes=1000 -rounds 100 -concurrency 1 -pgnout file=/home/raph/leela/pgns/custom_depthwise-1000n -config outname=/home/raph/leela/configs/custom_depthwise-1000n.json

# Precision NHWC

fastchess -event 'NHWC' -engine name='cudnn' args='-w /home/raph/leela/networks/8x384x12h-2rbk-relu/8x384x12h-2rbk-relu-150000.pb.gz' cmd=./engines/fp32 -engine name='fp16' args='-w /home/raph/leela/networks/8x384x12h-2rbk-relu/8x384x12h-2rbk-relu-150000.pb.gz -o custom_depthwise=true,nhwc=true' cmd=./engines/fp16 -engine name='fp32' args='-w /home/raph/leela/networks/8x384x12h-2rbk-relu/8x384x12h-2rbk-relu-150000.pb.gz -o custom_depthwise=true,nhwc=true' cmd=./engines/fp32 -engine name='dense' args='-w /home/raph/leela/networks/8x384x12h-2rbk-relu/8x384x12h-2rbk-relu-150000.pb.gz -o custom_depthwise=true,nhwc=true' cmd=./engines/dense -openings file='/home/raph/leela/books/book-6-ply-unbalanced.pgn' format=pgn order=sequential -each nodes=1000 -rounds 100 -concurrency 1 -pgnout file=/home/raph/leela/pgns/precision_nhwc-1000n -config outname=/home/raph/leela/configs/precision_nhwc-1000n.json

fastchess -event 'NHWC' -engine name='fp16' args='-w /home/raph/leela/networks/8x384x12h-2rbk-256/8x384x12h-2rbk-256-150000.pb.gz -o custom_depthwise=true,nhwc=true' cmd=./engines/fp16 -engine name='fp32' args='-w /home/raph/leela/networks/8x384x12h-2rbk-256/8x384x12h-2rbk-256-150000.pb.gz -o custom_depthwise=true,nhwc=true' cmd=./engines/fp32 -openings file='/home/raph/leela/books/book-6-ply-unbalanced.pgn' format=pgn order=sequential -each tc=60+0.1 -rounds 100 -concurrency 1 -pgnout file=/home/raph/leela/pgns/precision_nhwc-60s -config outname=/home/raph/leela/configs/precision_nhwc-60s.json







fastchess -event 'NHWC' -engine name='standard' args='-w /home/raph/leela/networks/8x384x12h-256/8x384x12h-256-150000.pb.gz' cmd=./engines/fp32 -engine name='rbk' args='-w /home/raph/leela/networks/8x384x12h-2rbk-relu/8x384x12h-2rbk-relu-150000.pb.gz' cmd=./engines/fp16 -openings file='/home/raph/leela/books/book-6-ply-unbalanced.pgn' format=pgn order=sequential -each nodes=1000 -rounds 100 -concurrency 1 -pgnout file=/home/raph/leela/pgns/rbk_vs_new-1000n -config outname=/home/raph/leela/configs/rbk_vs_new-1000n.json

fastchess -event 'NHWC' -engine name='standard' args='-w /home/raph/leela/networks/8x384x12h-256/8x384x12h-256-150000.pb.gz' cmd=./build/release/lc0 -engine name='rbk' args='-w /home/raph/leela/networks/8x384x12h-2rbk-256/8x384x12h-2rbk-256-150000.pb.gz -o custom_depthwise=true,nhwc=true' cmd=./build/release/lc0 -openings file='/home/raph/leela/books/book-6-ply-unbalanced.pgn' format=pgn order=sequential -each tc=60+0.1 -rounds 100 -concurrency 1 -pgnout file=/home/raph/leela/pgns/rbk_vs_new-60s -config outname=/home/raph/leela/configs/rbk_vs_new-60s.json

fastchess -config file=/home/raph/leela/configs/rbk_vs_new-60s.json


fastchess -event 'NHWC' -engine name='standard' args='-w /home/raph/leela/networks/8x384x12h-256/8x384x12h-256-150000.pb.gz' cmd=./build/release/lc0 -engine name='res' args='-w /home/raph/leela/networks/8x384x12h-2res-256/8x384x12h-2res-256-150000.pb.gz' cmd=./build/release/lc0 -openings file='/home/raph/leela/books/book-6-ply-unbalanced.pgn' format=pgn order=sequential -each nodes=1000 -rounds 100 -concurrency 1 -pgnout file=/home/raph/leela/pgns/res_vs_new-1000n -config outname=/home/raph/leela/configs/res_vs_new-1000n.json

