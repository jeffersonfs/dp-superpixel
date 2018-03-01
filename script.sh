#/bin/bash

for entry in "/home/jefferson/Ferramentas/dataset/icgbench/groundtruth"/*
do
   #echo "$entry"
   filename=$(basename "$entry" .gt)
   #echo "$filename"
   /home/jefferson/Programas/gco-dp-build/app/img/img-gco -i /home/jefferson/Ferramentas/dataset/icgbench/images -o /home/jefferson/Ferramentas/dataset/icgbench-test/dp/ -g $entry -d /home/jefferson/Ferramentas/dataset/icgbench/result2/$filename
done
