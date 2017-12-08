#!/bin/bash
#
# clear the model output directory
#
for i in $(seq 0 1); do
   for j in $(seq 1 9); do
# echo bssd_300_vgg16_reduced_300-00$i$j.params 
    rm -f 'bssd_300_vgg16_reduced_300-00$i$j.params' 
   done
done

