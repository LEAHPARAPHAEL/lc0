# 8x384x12h-2rbk-relu

fastchess -event 'Custom VS CUDNN 60s' -engine name='CUDNN' args='-w /home/raph/leela/networks/8x256x8h-2rbk/8x256x8h-2rbk-150000.pb.gz' cmd=./build/release/lc0 -engine name='Custom NCHW' args='-w /home/raph/leela/networks/8x256x8h-2rbk/8x256x8h-2rbk-150000.pb.gz -o custom_depthwise=true' cmd=./build/release/lc0 -openings file='/home/raph/leela/books/book-6-ply-unbalanced.pgn' format=pgn order=sequential -each tc=60+0.1 -rounds 100 -concurrency 1 -pgnout file=/home/raph/leela/pgns/custom_depthwise-60s -config outname=/home/raph/leela/configs/custom_depthwise-60s.json

fastchess -event 'Custom VS CUDNN 1000 nodes' -engine name='CUDNN' args='-w /home/raph/leela/networks/8x384x12h-2rbk-relu/8x384x12h-2rbk-relu-150000.pb.gz' cmd=./build/release/lc0 -engine name='Custom NCHW' args='-w /home/raph/leela/networks/8x384x12h-2rbk-relu/8x384x12h-2rbk-relu-150000.pb.gz -o custom_depthwise=true' cmd=./build/release/lc0 -openings file='/home/raph/leela/books/book-6-ply-unbalanced.pgn' format=pgn order=sequential -each nodes=1000 -rounds 100 -concurrency 1 -pgnout file=/home/raph/leela/pgns/custom_depthwise-1000n -config outname=/home/raph/leela/configs/custom_depthwise-1000n.json

# Precision NHWC

fastchess -event 'NHWC' -engine name='cudnn' args='-w /home/raph/leela/networks/8x384x12h-2rbk-relu/8x384x12h-2rbk-relu-150000.pb.gz' cmd=./engines/fp32 -engine name='fp16' args='-w /home/raph/leela/networks/8x384x12h-2rbk-relu/8x384x12h-2rbk-relu-150000.pb.gz -o custom_depthwise=true,nhwc=true' cmd=./engines/fp16 -engine name='fp32' args='-w /home/raph/leela/networks/8x384x12h-2rbk-relu/8x384x12h-2rbk-relu-150000.pb.gz -o custom_depthwise=true,nhwc=true' cmd=./engines/fp32 -engine name='dense' args='-w /home/raph/leela/networks/8x384x12h-2rbk-relu/8x384x12h-2rbk-relu-150000.pb.gz -o custom_depthwise=true,nhwc=true' cmd=./engines/dense -openings file='/home/raph/leela/books/book-6-ply-unbalanced.pgn' format=pgn order=sequential -each nodes=1000 -rounds 100 -concurrency 1 -pgnout file=/home/raph/leela/pgns/precision_nhwc-1000n -config outname=/home/raph/leela/configs/precision_nhwc-1000n.json

fastchess -event 'NHWC' -engine name='fp16' args='-w /home/raph/leela/networks/8x384x12h-2rbk-256/8x384x12h-2rbk-256-150000.pb.gz -o custom_depthwise=true,nhwc=true' cmd=./engines/fp16 -engine name='cudnn' args='-w /home/raph/leela/networks/8x384x12h-2rbk-256/8x384x12h-2rbk-256-150000.pb.gz' cmd=./engines/fp32 -openings file='/home/raph/leela/books/book-6-ply-unbalanced.pgn' format=pgn order=sequential -each tc=60+0.1 -rounds 100 -concurrency 1 -pgnout file=/home/raph/leela/pgns/precision_nhwc-60s -config outname=/home/raph/leela/configs/precision_nhwc-60s.json







fastchess -event 'NHWC' -engine name='standard' args='-w /home/raph/leela/networks/8x384x12h-256/8x384x12h-256-150000.pb.gz' cmd=./engines/fp32 -engine name='rbk' args='-w /home/raph/leela/networks/8x384x12h-2rbk-relu/8x384x12h-2rbk-relu-150000.pb.gz' cmd=./engines/fp16 -openings file='/home/raph/leela/books/book-6-ply-unbalanced.pgn' format=pgn order=sequential -each nodes=1000 -rounds 100 -concurrency 1 -pgnout file=/home/raph/leela/pgns/rbk_vs_new-1000n -config outname=/home/raph/leela/configs/rbk_vs_new-1000n.json

fastchess -event 'NHWC' -engine name='standard' args='-w /home/raph/leela/networks/8x384x12h-256/8x384x12h-256-150000.pb.gz' cmd=./build/release/lc0 -engine name='rbk' args='-w /home/raph/leela/networks/8x384x12h-2rbk-256/8x384x12h-2rbk-256-150000.pb.gz -o custom_depthwise=true,nhwc=true' cmd=./build/release/lc0 -openings file='/home/raph/leela/books/book-6-ply-unbalanced.pgn' format=pgn order=sequential -each tc=60+0.1 -rounds 100 -concurrency 1 -pgnout file=/home/raph/leela/pgns/rbk_vs_new-60s -config outname=/home/raph/leela/configs/rbk_vs_new-60s.json

fastchess -config file=/home/raph/leela/configs/rbk_vs_new-60s.json


fastchess -event 'NHWC' -engine name='standard' args='-w /home/raph/leela/networks/8x384x12h-256/8x384x12h-256-150000.pb.gz' cmd=./build/release/lc0 -engine name='res' args='-w /home/raph/leela/networks/8x384x12h-2res-256/8x384x12h-2res-256-150000.pb.gz' cmd=./build/release/lc0 -openings file='/home/raph/leela/books/book-6-ply-unbalanced.pgn' format=pgn order=sequential -each nodes=1000 -rounds 100 -concurrency 1 -pgnout file=/home/raph/leela/pgns/res_vs_new-1000n -config outname=/home/raph/leela/configs/res_vs_new-1000n.json

# Mx2-Tx8 VS Tx8

fastchess -event 'NHWC' -engine name='Tx8' args='-w /home/raph/leela/networks/Tx8/Tx8-150000.pb.gz' cmd=./build/release/lc0 -engine name='Mx2-Tx8' args='-w /home/raph/leela/networks/Mx2-Tx8/Mx2-Tx8-150000.pb.gz' cmd=./build/release/lc0 -openings file='/home/raph/leela/books/book-6-ply-unbalanced.pgn' format=pgn order=sequential -each nodes=1000 -rounds 100 -concurrency 1 -pgnout file=/home/raph/leela/pgns/Tx8_vs_Mx2-Tx8-1000n -config outname=/home/raph/leela/configs/Tx8_vs_Mx2-Tx8-1000n.json

# Mx2-Tx8 VS Mx4-Tx6

fastchess -event 'NHWC' -engine name='Mx2-Tx8' args='-w /home/raph/leela/networks/Mx2-Tx8/Mx2-Tx8-150000.pb.gz' cmd=./build/release/lc0 -engine name='Mx4-Tx6' args='-w /home/raph/leela/networks/Mx4-Tx6/Mx4-Tx6-150000.pb.gz' cmd=./build/release/lc0 -openings file='/home/raph/leela/books/book-6-ply-unbalanced.pgn' format=pgn order=sequential -each tc=60+0.1 -rounds 100 -concurrency 1 -pgnout file=/home/raph/leela/pgns/Mx2-Tx8_vs_Mx4-Tx6-60s -config outname=/home/raph/leela/configs/Mx2-Tx8_vs_Mx4-Tx6-60s.json



fastchess -event 'Tx8_vs_Mx6-Tx4' -engine name='Mx6-Tx4' args='-w /home/raph/leela/networks/Mx6-Tx4/Mx6-Tx4-150000.pb.gz' cmd=./build/release/lc0 -engine name='Tx8' args='-w /home/raph/leela/networks/Tx8/Tx8-150000.pb.gz' cmd=./build/release/lc0 -openings file='/home/raph/leela/books/book-6-ply-unbalanced.pgn' format=pgn order=sequential -each tc=60+0.1 -rounds 100 -concurrency 1 -pgnout file=/home/raph/leela/pgns/Tx8_vs_Mx6-Tx4-60s -config outname=/home/raph/leela/configs/Tx8_vs_Mx6-Tx4-60s.json

fastchess -config file=/home/raph/leela/configs/Tx8_vs_Mx6-Tx4-60s.json

fastchess -event 'Tx8_vs_Mx6-Tx4' -engine name='Mx6-Tx4' args='-w /home/raph/leela/networks/Mx6-Tx4/Mx6-Tx4-150000.pb.gz' cmd=./build/release/lc0 -engine name='Tx8' args='-w /home/raph/leela/networks/Tx8/Tx8-150000.pb.gz' cmd=./build/release/lc0 -openings file='/home/raph/leela/books/book-6-ply-unbalanced.pgn' format=pgn order=sequential -each nodes=1000 -rounds 100 -concurrency 1 -pgnout file=/home/raph/leela/pgns/Tx8_vs_Mx6-Tx4-1000n -config outname=/home/raph/leela/configs/Tx8_vs_Mx6-Tx4-1000n.json

fastchess -config file=/home/raph/leela/configs/Tx8_vs_Mx6-Tx4-1000n.json

# All

fastchess -event 'M-VS-T' -engine name='Tx8' args='-w /home/raph/leela/networks/Tx8/Tx8-150000.pb.gz' -engine name='Mx2-Tx8' args='-w /home/raph/leela/networks/Mx2-Tx8/Mx2-Tx8-150000.pb.gz' -engine name='Mx4-Tx6' args='-w /home/raph/leela/networks/Mx4-Tx6/Mx4-Tx6-150000.pb.gz' -engine name='Mx6-Tx4' args='-w /home/raph/leela/networks/Mx6-Tx4/Mx6-Tx4-150000.pb.gz' -engine name='Mx8-Tx2' args='-w /home/raph/leela/networks/Mx8-Tx2/Mx8-Tx2-150000.pb.gz' -engine name='Mx10' args='-w /home/raph/leela/networks/Mx10/Mx10-150000.pb.gz' -openings file='/home/raph/leela/books/book-6-ply-unbalanced.pgn' format=pgn order=sequential -each cmd=./build/release/lc0 tc=60+0.1 -rounds 100 -concurrency 1 -pgnout file=/home/raph/leela/pgns/M-VS-T-60s -config outname=/home/raph/leela/configs/M-VS-T-60s.json

# Tx8 VS Mx2-Tx8
fastchess -event 'Tx8 VS Mx2-Tx8' -engine name='Tx8' args='-w /home/raph/leela/networks/Tx8/Tx8-150000.pb.gz' -engine name='Mx2-Tx8' args='-w /home/raph/leela/networks/Mx2-Tx8/Mx2-Tx8-150000.pb.gz' cmd=./build/release/lc0 -openings file='/home/raph/leela/books/book-6-ply-unbalanced.pgn' format=pgn order=sequential -each cmd=./build/release/lc0 tc=60+0.1 -rounds 50 -concurrency 1 -pgnout file=/home/raph/leela/pgns/Tx8_vs_Mx2-Tx8-60s -config outname=/home/raph/leela/configs/Tx8_vs_Mx2-Tx8-60s.json

# Tx8 VS Mx4-Tx6
fastchess -event 'Tx8 VS Mx4-Tx6' -engine name='Tx8' args='-w /home/raph/leela/networks/Tx8/Tx8-150000.pb.gz' -engine name='Mx4-Tx6' args='-w /home/raph/leela/networks/Mx4-Tx6/Mx4-Tx6-150000.pb.gz' cmd=./build/release/lc0 -openings file='/home/raph/leela/books/book-6-ply-unbalanced.pgn' format=pgn order=sequential -each cmd=./build/release/lc0 tc=60+0.1 -rounds 50 -concurrency 1 -pgnout file=/home/raph/leela/pgns/Tx8_vs_Mx4-Tx6-60s -config outname=/home/raph/leela/configs/Tx8_vs_Mx4-Tx6-60s.json

fastchess -event 'Tx8 VS Mx4-Tx6' -engine name='Tx8' args='-w /home/raph/leela/networks/Tx8/Tx8-150000.pb.gz' cmd=./build/release/lc0 -engine name='Mx4-Tx6' args='-w /home/raph/leela/networks/Mx4-Tx6/Mx4-Tx6-150000.pb.gz' cmd=./build/release/lc0 -openings file='/home/raph/leela/books/book-6-ply-unbalanced.pgn' format=pgn order=sequential -each nodes=1000 -rounds 50 -concurrency 1 -pgnout file=/home/raph/leela/pgns/Tx8_vs_Mx4-Tx6-1000n -config outname=/home/raph/leela/configs/Tx8_vs_Mx4-Tx6-1000n.json

# EPS

fastchess -event 'eps' -engine name='eps-3' args='-w /home/raph/leela/networks/Mx2-Tx8/Mx2-Tx8-150000.pb.gz' cmd=./engines/eps-3 -engine name='eps-6' args='-w /home/raph/leela/networks/Mx2-Tx8/Mx2-Tx8-150000.pb.gz' cmd=./engines/eps-6 -openings file='/home/raph/leela/books/book-6-ply-unbalanced.pgn' format=pgn order=sequential -each nodes=1000 -rounds 100 -concurrency 1 -pgnout file=/home/raph/leela/pgns/eps -config outname=/home/raph/leela/configs/eps.json

# Tx8 VS Tx8-pre VS MMTMMTMMT

fastchess -event 'Prenorm VS Postnorm' -engine name='Tx8' args='-w /home/raph/leela/networks/Tx8/Tx8-150000.pb.gz' -engine name='Tx8-pre' args='-w /home/raph/leela/networks/Tx8-pre/Tx8-pre-150000.pb.gz' -engine name='MMTMMTMMT' args='-w /home/raph/leela/networks/MMTMMTMMT-cuda/MMTMMTMMT-cuda-150000.pb.gz' -openings file='/home/raph/leela/books/book-6-ply-unbalanced.pgn' format=pgn order=sequential -each cmd=./build/release/lc0 tc=60+0.1 -rounds 50 -concurrency 1 -pgnout file=/home/raph/leela/pgns/pre_VS_post-60s -config outname=/home/raph/leela/configs/pre_VS_post-60s.json

fastchess -event 'Prenorm VS Postnorm' -engine name='Tx8' args='-w /home/raph/leela/networks/Tx8/Tx8-150000.pb.gz' -engine name='Tx8-pre' args='-w /home/raph/leela/networks/Tx8-pre/Tx8-pre-150000.pb.gz' -engine name='MMTMMTMMT' args='-w /home/raph/leela/networks/MMTMMTMMT-cuda/MMTMMTMMT-cuda-150000.pb.gz' -openings file='/home/raph/leela/books/book-6-ply-unbalanced.pgn' format=pgn order=sequential -each cmd=./build/release/lc0 nodes=1000 -rounds 50 -concurrency 1 -pgnout file=/home/raph/leela/pgns/pre_VS_post-1000n -config outname=/home/raph/leela/configs/pre_VS_post-1000n.json


fastchess -event 'Tx8 VS MMTMMTMMT' -engine name='Tx8' args='-w /home/raph/leela/networks/Tx8/Tx8-150000.pb.gz' -engine name='MMTMMTMMT' args='-w /home/raph/leela/networks/MMTMMTMMT-cuda/MMTMMTMMT-cuda-150000.pb.gz' -openings file='/home/raph/leela/books/book-6-ply-unbalanced.pgn' format=pgn order=sequential -each cmd=./build/release/lc0 tc=60+0.1 -rounds 200 -concurrency 1 -pgnout file=/home/raph/leela/pgns/Tx8_VS_MMTMMTMMT-60s -config outname=/home/raph/leela/configs/Tx8_VS_MMTMMTMMT-60s.json

fastchess -event 'Prenorm VS Postnorm' -engine name='Tx8' args='-w /home/raph/leela/networks/Tx8/Tx8-150000.pb.gz' -engine name='Tx8-pre' args='-w /home/raph/leela/networks/Tx8-pre/Tx8-pre-150000.pb.gz' -openings file='/home/raph/leela/books/book-6-ply-unbalanced.pgn' format=pgn order=sequential -each cmd=./build/release/lc0 tc=60+0.1 -rounds 200 -concurrency 1 -pgnout file=/home/raph/leela/pgns/pre_VS_post-60s -config outname=/home/raph/leela/configs/pre_vs_post-60s.json

fastchess -config file=/home/raph/leela/configs/Tx8_VS_MMTMMTMMT-60s.json

fastchess -config file=/home/raph/leela/configs/pre_vs_post-60s.json

# CUDNN VS Custom

fastchess -event 'CUDNN Benchmark' -engine name='custom' args='-w /home/raph/leela/networks/MMTMMTMMT-cuda/MMTMMTMMT-cuda-300000.pb.gz' -engine name='cudnn' args='-w /home/raph/leela/networks/MMTMMTMMT-cuda/MMTMMTMMT-cuda-300000.pb.gz -o custom_depthwise=false'  -openings file='/home/raph/leela/books/book-6-ply-unbalanced.pgn' format=pgn order=sequential -each cmd=./build/release/lc0 tc=60+0.1 -rounds 200 -concurrency 1 -pgnout file=/home/raph/leela/pgns/cudnn-60s -config outname=/home/raph/leela/configs/cudnn-60s.json

# Classical VS v3

fastchess -event 'V3 Benchmark' -engine name='custom' args='-w /home/raph/leela/networks/MMTMMTMMT-cuda/MMTMMTMMT-cuda-300000.pb.gz' cmd=./build/release/lc0 -engine name='v3' args='-w /home/raph/leela/networks/MMTMMTMMT-v3/MMTMMTMMT-v3-300000.pb.gz' cmd=./engines/v3 -openings file='/home/raph/leela/books/book-6-ply-unbalanced.pgn' format=pgn order=sequential -each tc=60+0.1 -rounds 200 -concurrency 1 -pgnout file=/home/raph/leela/pgns/v3-60s -config outname=/home/raph/leela/configs/v3-60s.json

# 5x5 VS dense

fastchess -event 'Sparse VS Dense' -engine name='sparse' args='-w /home/raph/leela/networks/MMTx3/MMTx3-100000.pb.gz' cmd=./engines/custom -engine name='dense' args='-w /home/raph/leela/networks/MMTx3-dense/MMTx3-dense-100000.pb.gz -o custom_depthwise=false' cmd=./engines/dense_5x5 -openings file='/home/raph/leela/books/book-6-ply-unbalanced.pgn' format=pgn order=sequential -each tc=60+0.1 -rounds 200 -concurrency 1 -pgnout file=/home/raph/leela/pgns/sparse_VS_dense -config outname=/home/raph/leela/configs/sparse_VS_dense.json

# 3x3 VS dense

fastchess -event 'Sparse VS 3x3' -engine name='sparse' args='-w /home/raph/leela/networks/MMTx3/MMTx3-100000.pb.gz' cmd=./engines/custom -engine name='3x3' args='-w /home/raph/leela/networks/MMTx3-3x3/MMTx3-3x3-100000.pb.gz' cmd=./engines/dense_3x3 -openings file='/home/raph/leela/books/book-6-ply-unbalanced.pgn' format=pgn order=sequential -each tc=60+0.1 -rounds 200 -concurrency 1 -pgnout file=/home/raph/leela/pgns/sparse_VS_3x3 -config outname=/home/raph/leela/configs/sparse_VS_3x3.json