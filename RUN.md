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


lc0.exe --weights=C:\Users\RAPHAL~1\Leela\Networks\FormatDWPW\mobile_net_brk-96x6-latest.pb.gz
lc0.exe --weights=C:\Users\RAPHAL~1\Leela\Networks\mobile_net_brk-96x6\mobile_net_brk-96x6-100000.pb.gz -o fuse_DWPW=false



