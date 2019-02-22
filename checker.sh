in=./in
out=./out
ref=./ref

names=(lenna)
colors=(bw color)
filters=(smooth blur sharpen mean emboss)
format=(pgm pnm)

mpicc homework.c
N=8

for i in ${names[@]}; do
    for c in $(seq 0 $((${#colors[@]} - 1)));  do
        for f in ${filters[@]}; do
            fileName_in=${i}_${colors[$c]}.${format[$c]}
            fileName_out=${i}_${colors[$c]}_${f}_out.${format[$c]}
            fileName_ref=${i}_${colors[$c]}_${f}.${format[$c]}
            mpirun -np $N ./a.out $in/$fileName_in $out/$fileName_out ${f}
            diff $out/$fileName_out $ref/$fileName_ref #&>/dev/null
            if [ $? -eq 0 ]; then
                echo "Test $fileName_ref ........... [OK]"
            else
                echo "Test $fileName_ref ........... [FAIL]"
            fi
        done
    done 
done

# inorderFilters=(blur smooth sharpen emboss mean blur smooth sharpen emboss mean)

# cp ./in/lenna_color.pnm ./out/lenna_color.pnm
# mpirun -np $N ./a.out ./in/lenna_color.pnm ./out/lenna_color_f.pnm $(echo ${inorderFilters[*]})

# for f in ${inorderFilters[@]}; do
#     mpirun -np $N ./a.out ./out/lenna_color.pnm ./out/lenna_color.pnm ${f}
# done

# diff ./out/lenna_color_f.pnm ./out/lenna_color.pnm
# diff ./out/lenna_color_f.pnm ./ref/lenna_color_bssembssem.pnm
