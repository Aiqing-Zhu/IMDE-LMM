#!/bin/bash

Ns=4
for ((s=0; s<=Ns; s++))
do
    echo $s is ok
    nohup python GO.py --i 1 --device 0 --s $s > l1.file 2>&1 &
    nohup python GO.py --i 2 --device 0 --s $s > l1.file 2>&1 &
    nohup python GO.py --i 4 --device 1 --s $s > l1.file 2>&1 &
    nohup python GO.py --i 8 --device 1 --s $s > l1.file 2>&1 &
    nohup python GO.py --i 16 --device 0 --s $s > l1.file 2>&1 &
done
