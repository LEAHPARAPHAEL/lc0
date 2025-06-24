--backend-opts


Depthwise convolution result at positions before converting (0, 340-341, 3, 1): (0.000000, 0.000000)
Depthwise convolution result at positions before converting (0, 560-561, 0, 3): (0.395479, 0.057922)
Depthwise convolution result at positions before converting (0, 140-141, 7, 7): (0.000000, 0.052323)
Depthwise convolution result at positions (0, 340-341, 3, 1): (0.000000, 0.000000)
Depthwise convolution result at positions (0, 140-141, 7, 7): (0.000000, 0.052338)
Depthwise convolution result at positions (0, 560-561, 0, 3): (0.395508, 0.057922)
Pointwise convolution result at positions converted (0, 24-25, 3, 1): (0.371826, 0.251709)
Pointwise convolution result at positions converted (0, 66-67, 3, 1): (0.059082, -0.169067)
Pointwise convolution result at positions (0, 24-25, 3, 1): (0.371803, 0.251790)
Pointwise convolution result at positions (0, 66-67, 3, 1): (0.059070, -0.169023)

0.052323 vs 0.0522461

Depthwise convolution result at position (0, 340, 3, 1): 0
Depthwise convolution result at position (0, 341, 3, 1): 0
Depthwise convolution result at position (0, 560, 0, 3): 0.395508
Depthwise convolution result at position (0, 561, 0, 3): 0.0579224
Depthwise convolution result at position (0, 140, 7, 7): 0
Depthwise convolution result at position (0, 141, 7, 7): 0.0522461
Pointwise convolution result at position (0, 66, 3, 1): 0.0589905
Pointwise convolution result at position (0, 67, 3, 1): -0.169067
Pointwise convolution result at position (0, 24, 3, 1): 0.371338
Pointwise convolution result at position (0, 25, 3, 1): 0.251709

Depthwise convolution result at positions before converting (0, 202-203, 3, 2): (0.000000, 0.218719)
Depthwise convolution result at positions before converting (0, 560-561, 0, 3): (0.395479, 0.057922)
Depthwise convolution result at positions before converting (0, 140-141, 7, 7): (0.000000, 0.052323)
Pointwise convolution result at positions converted (0, 24, 3, 1): 0.371803
Pointwise convolution result at positions converted (0, 66, 3, 1): 0.059070



# Mobile net not optimized
Benchmark batch size 50 with inference average time 1.95075ms - throughput 25631.1 nps.
Benchmark batch size 100 with inference average time 3.45351ms - throughput 28956 nps.
Benchmark batch size 150 with inference average time 5.01951ms - throughput 29883.4 nps.
Benchmark batch size 200 with inference average time 6.65646ms - throughput 30046 nps.


# Residual net 
Benchmark batch size 50 with inference average time 0.92922ms - throughput 53808.6 nps.
Benchmark batch size 100 with inference average time 1.23695ms - throughput 80843.8 nps.
Benchmark batch size 150 with inference average time 1.55236ms - throughput 96627.2 nps.
Benchmark batch size 200 with inference average time 1.86575ms - throughput 107196 nps.


