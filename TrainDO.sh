#!/bin/bash

Nr=8
for ((i=1; i<=Nr; i++))
do
    echo $i is ok
    nohup python DO.py --i $i --s 0 --device 0 > l1.file 2>&1 &
    nohup python DO.py --i $i --s 1 --device 1 > l1.file 2>&1 &
done

#set other seed for training