# 8x256x8h VS 8x256x8h-crbk 60+1 :

fastchess -event '8x256x8h VS 8x256x8h-crbk balanced 60+1s' -engine name='8x256x8h' args='-w /home/raph/leela/networks/8x256x8h/8x256x8h-150000.pb.gz' cmd=./build/lc0 -engine name='8x256x8h-crbk' args='-w /home/raph/leela/networks/8x256x8h-crbk/8x256x8h-crbk-150000.pb.gz' cmd=./build/lc0 -openings file='/home/raph/leela/books/book-6-ply-balanced.pgn' format=pgn order=random -each tc=60+0.1 -rounds 100 -concurrency 1 -pgnout file=/home/raph/leela/pgns/crbk_vs_0-60s -config outname=/home/raph/leela/configs/crbk_vs_0-60s.json 

fastchess-windows-latest -config file=/home/raph/leela/configs/crbk_vs_0-60s.json 

# 8x256x8h VS 8x256x8h-crbk 10000 nodes :

fastchess -event '8x256x8h VS 8x256x8h-crbk balanced 60+1s' -engine name='8x256x8h' args='-w /home/raph/leela/networks/8x256x8h/8x256x8h-150000.pb.gz' cmd=./build/lc0 -engine name='8x256x8h-crbk' args='-w /home/raph/leela/networks/8x256x8h-crbk/8x256x8h-crbk-150000.pb.gz' cmd=./build/lc0 -openings file='/home/raph/leela/books/book-6-ply-balanced.pgn' format=pgn order=random -each nodes=10000 -rounds 100 -concurrency 1 -pgnout file=/home/raph/leela/pgns/crbk_vs_0-10000n -config outname=/home/raph/leela/configs/crbk_vs_0-10000n.json 

fastchess-windows-latest -config file=/home/raph/leela/configs/crbk_vs_0-10000n.json 

# BT4 :

--minibatch-size=16

fastchess -event 'BT4 VS 8x256x8h-crbk balanced 60+1s' -engine name='BT4' args='-w /home/raph/leela/networks/BT4-1024x15x32h-swa-6147500-policytune-332.pb.gz --minibatch-size=16' cmd=./build/lc0 -engine name='8x256x8h-crbk' args='-w /home/raph/leela/networks/8x256x8h-crbk/8x256x8h-crbk-150000.pb.gz' cmd=./build/lc0 -openings file='/home/raph/leela/books/book-6-ply-balanced.pgn' format=pgn order=random -each tc=60+0.1 -rounds 100 -concurrency 1 -pgnout file=/home/raph/leela/pgns/crbk_vs_BT4-60s -config outname=/home/raph/leela/configs/crbk_vs_BT4-60s.json 

fastchess-windows-latest -config file=/home/raph/leela/configs/crbk_vs_BT4-60s.json 

# 8x384x12h VS 8x384x12h-c2r2bk 60+1 :

fastchess -event '8x384x12h VS 8x384x12h-c2r2bk balanced 60+1s' -engine name='8x384x12h' args='-w /home/raph/leela/networks/8x384x12h/8x384x12h-110000.pb.gz' cmd=./build/lc0 -engine name='8x384x12h-c2r2bk' args='-w /home/raph/leela/networks/8x384x12h-c2r2bk/8x384x12h-c2r2bk-110000.pb.gz' cmd=./build/lc0 -openings file='/home/raph/leela/books/book-6-ply-unbalanced.pgn' format=pgn order=random -each tc=60+0.1 -rounds 100 -concurrency 1 -pgnout file=/home/raph/leela/pgns/8x12-crbk-60s -config outname=/home/raph/leela/configs/8x12-crbk-60s.json 

fastchess-windows-latest -config file=/home/raph/leela/configs/8x12-crbk-60s.json 

# 8x384x12h-256 VS 8x384x12h-2rbk-256 60+1 :

fastchess -event '8x384x12h-256 VS 8x384x12h-2rbk-256 unbalanced 60+1s' -engine name='8x384x12h' args='-w /home/raph/leela/networks/8x384x12h-256/8x384x12h-256-150000.pb.gz' cmd=./build/release/lc0 -engine name='8x384x12h-2rbk-256' args='-w /home/raph/leela/networks/8x384x12h-2rbk-256/8x384x12h-2rbk-256-150000.pb.gz' cmd=./build/release/lc0 -openings file='/home/raph/leela/books/book-6-ply-unbalanced.pgn' format=pgn order=random -each tc=60+0.1 -rounds 100 -concurrency 1 -pgnout file=/home/raph/leela/pgns/8x12-2rbk-60s -config file=/home/raph/leela/configs/8x12-2rbk-60s.json -recover

fastchess -event '8x384x12h-256 VS 8x384x12h-2rbk-256 unbalanced 60+1s' -engine name='8x384x12h' args='-w /home/raph/leela/networks/8x384x12h-256/8x384x12h-256-150000.pb.gz' cmd=./build/release/lc0 -engine name='8x384x12h-2rbk-256' args='-w /home/raph/leela/networks/8x384x12h-2rbk-256/8x384x12h-2rbk-256-150000.pb.gz' cmd=./build/release/lc0 -openings file='/home/raph/leela/books/book-6-ply-unbalanced.pgn' format=pgn order=random -each tc=60+0.1 -rounds 250 -concurrency 1 -pgnout file=/home/raph/leela/pgns/8x12-2rbk-60s -config outname=/home/raph/leela/configs/8x12-2rbk-60s.json
