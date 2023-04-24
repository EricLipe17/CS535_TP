#!/bin/bash

HOSTS=("venus" "jupiter" "mars")

CMD="kill $(ps aux | grep main_dist.py | grep -v grep | awk '{print $2}')"

$CMD

LENGTH=${#HOSTS[@]}
for (( j=1; j<${LENGTH}+1; j++ ));
do
  CMD="${CMD_TEMPLATE/RANK/"$j"}"
  echo ssh -n "${HOSTS[$j-1]}" "${CMD}"
done
