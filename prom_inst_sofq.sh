#!/bin/bash

dir=$1

samples=$(ls $dir/inst_sofq.dat | wc -l)

echo "#"$samples

paste $dir/inst_sofq.dat | awk '{acum1=0.0; acum2=0.0; n=0;for(i=1;i<=NF;i+=2){acum1+=$i;acum2+=$(i+1);n++;}; if(NF>0) print acum1*1.0/n, acum2*1.0/n; else print;}' \

#file="sofq_"$samples"samples.dat"
#echo $file

#gnuplot -p -e "set term png; set out 'sofq.png';set logs; plot for[i=0:7] '$file' index i u 0:1 w lp t ''"
