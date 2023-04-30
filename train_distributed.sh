#!/bin/bash

HOSTS=("uranus" "venus" "jupiter" "mars")

CMD_TEMPLATE="torchrun --nproc_per_node=1 --nnodes=4 --node_rank=RANK --master_addr=129.82.44.124 --master_port=38017 main.py"

# Start nodes
LENGTH=${#HOSTS[@]}
for (( j=0; j<${LENGTH}; j++ ));
do
  CMD="${CMD_TEMPLATE/RANK/"$j"}"
  ssh -n "${HOSTS[$j]}" "${CMD}"
done
