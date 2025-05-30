#!/bin/bash

samples=$(ls inst_sofq_*_.dat | wc -l)

paste inst_sofq_*_.dat | awk '{acum=0; for(i=0;i<NF;i++){acum+=$i}; if(NF>0) print acum*1.0/NF; else print;}' \
> "sofq_"$samples"samples.dat"

file="sofq_"$samples"samples.dat"
echo $file

gnuplot -p -e "set logs; plot for[i=0:7] '$file' index i u 0:1 w lp t ''"
