#!/bin/bash

HOSTS=("venus" "jupiter" "mars")

CMD_TEMPLATE="torchrun \
--nproc_per_node=1 --nnodes=4 --node_rank=RANK \
--master_addr=129.82.44.124 --master_port=38017 \
main_dist.py"

# Start master node
torchrun \
--nproc_per_node=1 --nnodes=4 --node_rank=0 \
--master_addr=129.82.44.124 --master_port=38017 \
main_dist.py

# Start worker nodes
LENGTH=${#HOSTS[@]}
for (( j=1; j<${LENGTH}+1; j++ ));
do
  CMD="${CMD_TEMPLATE/RANK/"$j"}"
  echo ssh -n "${HOSTS[$j-1]}" "${CMD}"
done
