#!/bin/bash

# Define the 6 networks
NETS=("Tx8" "Mx2-Tx8" "Mx4-Tx6" "Mx6-Tx4" "Mx8-Tx2" "Mx10")

# Calculate the total number of networks
NUM_NETS=${#NETS[@]}

echo "Starting a 15-match tournament..."

# Nested loop to generate the 15 unique pairs
for (( i=0; i<NUM_NETS; i++ )); do
    for (( j=i+1; j<NUM_NETS; j++ )); do
        
        ENG1="${NETS[$i]}"
        ENG2="${NETS[$j]}"
        
        echo "=========================================================="
        echo "Now starting: ${ENG1} VS ${ENG2}"
        echo "=========================================================="

        fastchess \
          -event "${ENG1} VS ${ENG2}" \
          -engine name="${ENG1}" args="-w /home/raph/leela/networks/${ENG1}/${ENG1}-150000.pb.gz" \
          -engine name="${ENG2}" args="-w /home/raph/leela/networks/${ENG2}/${ENG2}-150000.pb.gz" cmd=./build/release/lc0 \
          -openings file='/home/raph/leela/books/book-6-ply-unbalanced.pgn' format=pgn order=sequential \
          -each cmd=./build/release/lc0 tc=30+0.1 \
          -rounds 50 -concurrency 1 \
          -pgnout file="/home/raph/leela/pgns/${ENG1}_vs_${ENG2}-30s" \
          -config outname="/home/raph/leela/configs/${ENG1}_vs_${ENG2}-30s.json"
          
        echo "Finished: ${ENG1} VS ${ENG2}"
        echo ""

    done
done

echo "All 15 matches are complete!"