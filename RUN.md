.\lc0.exe backendbench --config=C:\Users\RAPHAL~1\Documents\GitHub\lc0\lc0.config

# cuDNN mobile net
lc0.exe --weights=C:\Users\RAPHAL~1\Leela\Networks\mobile_net_brk-96x6\mobile_net_brk-96x6-100000.pb.gz -o fuse_DWPW=false

# Weights size 9
lc0.exe --weights=C:\Users\RAPHAL~1\Leela\Networks\FormatDWPW\mobile_net_brk-96x6-latest.pb.gz


# mobile net 96x6 !
lc0.exe --weights=C:\Users\RAPHAL~1\Leela\Networks\mobile_net-96x6\mobile_net-96x6-100000.pb.gz -o fuse_DWPW=false

# mobile net 96x18 !
lc0.exe --weights=C:\Users\RAPHAL~1\Leela\Networks\mobile_net-96x18\mobile_net-96x18-1000.pb.gz -o fuse_DWPW=false

# mobile net 192x6 !
lc0.exe --weights=C:\Users\RAPHAL~1\Leela\Networks\mobile_net-192x6\mobile_net-192x6-1000.pb.gz -o fuse_DWPW=false

# mobile net brk 96x6 !
lc0.exe --weights=C:\Users\RAPHAL~1\Leela\Networks\mobile_net_brk-96x6\mobile_net_brk-96x6-1000.pb.gz -o fuse_DWPW=false

# mobile net brk 96x18 ! 
lc0.exe --weights=C:\Users\RAPHAL~1\Leela\Networks\mobile_net_brk-96x18\mobile_net_brk-96x18-1000.pb.gz -o fuse_DWPW=false

# mobile net brk 192x6 ! 
lc0.exe --weights=C:\Users\RAPHAL~1\Leela\Networks\mobile_net_brk-192x6\mobile_net_brk-192x6-1000.pb.gz -o fuse_DWPW=false

# residual 96x6 !
lc0.exe --weights=C:\Users\RAPHAL~1\Leela\Networks\residual-3x3-96x6\residual_block-3x3-96x6-1000.pb.gz 

# residual 96x18 !
lc0.exe --weights=C:\Users\RAPHAL~1\Leela\Networks\residual-3x3-96x18\residual_block-3x3-96x18-1000.pb.gz 

# residual 192x6 !
lc0.exe --weights=C:\Users\RAPHAL~1\Leela\Networks\residual-3x3-192x6\residual_block-3x3-192x6-1000.pb.gz 

# residual 5x5 96x6 !
lc0.exe --weights=C:\Users\RAPHAL~1\Leela\Networks\residual-5x5-96x6\residual_block-5x5-96x6-100000.pb.gz 

# residual 5x5 96x18 ! 
lc0.exe --weights=C:\Users\RAPHAL~1\Leela\Networks\residual-5x5-96x18\residual_block-5x5-96x18-1000.pb.gz 

# residual 5x5 192x6 !
lc0.exe --weights=C:\Users\RAPHAL~1\Leela\Networks\residual-5x5-192x6\residual_block-5x5-192x6-1000.pb.gz

# residual br 96x6 !
lc0.exe --weights=C:\Users\RAPHAL~1\Leela\Networks\residual_br-5x5-96x6\residual_block_br-5x5-96x6-1000.pb.gz

# residual br 96x18 ! 
lc0.exe --weights=C:\Users\RAPHAL~1\Leela\Networks\residual_br-5x5-96x18\residual_block_br-5x5-96x18-1000.pb.gz

# residual br 192x6 ! 
lc0.exe --weights=C:\Users\RAPHAL~1\Leela\Networks\residual_br-5x5-192x6\residual_block_br-5x5-192x6-1000.pb.gz

# residual brk 96x6 ! 
lc0.exe --weights=C:\Users\RAPHAL~1\Leela\Networks\residual_brk-5x5-96x6\residual_block_brk-5x5-96x6-1000.pb.gz

# residual brk 96x18 ! 
lc0.exe --weights=C:\Users\RAPHAL~1\Leela\Networks\residual_brk-5x5-96x18\residual_block_brk-5x5-96x18-1000.pb.gz

# residual brk 192x6 ! 
lc0.exe --weights=C:\Users\RAPHAL~1\Leela\Networks\residual_brk-5x5-192x6\residual_block_brk-5x5-192x6-1000.pb.gz


lc0.exe --weights=C:\Users\RAPHAL~1\Leela\Networks\FormatDWPW\mobile_net_brk-96x6-latest.pb.gz -o fuse_DWPW=true
lc0.exe --weights=C:\Users\RAPHAL~1\Leela\Networks\mobile_net_brk-96x6\mobile_net_brk-96x6-100000.pb.gz 


Cublas output 3000 : 3.05273
Cublas output 3001 : 2.83594
Cublas output 3002 : 3.23047
Cublas output 3003 : 2.96094
Cublas output 3004 : 1.2168
Cublas output 3005 : 3.22852
Cublas output 3006 : 2.40039
Cublas output 3007 : 3.36328
Cublas output 3008 : -2.33984
Cublas output 3009 : -1.4541
Cublas output 3010 : -0.783203
Cublas output 3011 : -3.70898
Cublas output 3012 : -0.174438
Cublas output 3013 : -0.657715
Cublas output 3014 : -1.4541
Cublas output 3015 : -2.44336
Cublas output 3016 : 0.0337219
Cublas output 3017 : 0.104187
Cublas output 3018 : 0.0251923
Cublas output 3019 : 0.140503
Cublas output 3020 : 0.00909424
Cublas output 3021 : -0.144409
Cublas output 3022 : -0.0176392
Cublas output 3023 : 0.00675201
Cublas output 3024 : 0.244385


Cublas output 3000 : 3.05469
Cublas output 3001 : 2.82812
Cublas output 3002 : 3.22266
Cublas output 3003 : 2.96094
Cublas output 3004 : 1.21582
Cublas output 3005 : 3.21484
Cublas output 3006 : 2.39258
Cublas output 3007 : 3.35742
Cublas output 3008 : -2.33008
Cublas output 3009 : -1.4541
Cublas output 3010 : -0.782715
Cublas output 3011 : -3.70703
Cublas output 3012 : -0.174316
Cublas output 3013 : -0.658203
Cublas output 3014 : -1.45312
Cublas output 3015 : -2.44141
Cublas output 3016 : 0.0327148
Cublas output 3017 : 0.104553
Cublas output 3018 : 0.0248871
Cublas output 3019 : 0.140625
Cublas output 3020 : 0.00830078
Cublas output 3021 : -0.144043
Cublas output 3022 : -0.0175171
Cublas output 3023 : 0.00626755
Cublas output 3024 : 0.244507