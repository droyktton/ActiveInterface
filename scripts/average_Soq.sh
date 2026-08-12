samples=$(ls inst_sofq_*_.dat | wc -l)

paste inst_sofq_*_.dat | awk '{acum=0; for(i=0;i<NF;i++){acum+=$i}; if(NF>0) print acum*1.0/NF; else print;}' \
> "sofq_"$samples"samples.dat"

#paste cm_*_.dat | awk '{acum=0; for(i=0;i<NF;i++){acum[i%6]+=$i}; if(NF>0){for(i=0;i<6;i++) print acum[i]*6/NF}; else print;}' \
#> "cm_"$samples"samples.dat"
