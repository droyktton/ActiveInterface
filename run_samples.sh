samples=$1
L=$2
Nrun=$3
i0=$4

echo "samples="$samples

for((i=1;i<=$samples;i++))
do 
    semilla=$(awk -v i=$i -v i0=$i0 'BEGIN{print i+i0}')
    echo "semilla = "$semilla 

    ./activeinterface $L $Nrun $semilla 
    cp inst_sofq.dat "inst_sofq_"$i"_.dat" 
    cp cm.dat "cm_"$i"_.dat";
done

