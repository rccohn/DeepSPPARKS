#!/bin/bash

# quick test to verify container can start and execute code from several
# libraries without dependency issues

CNAME=rccohn/agg-nvdc-v3-base:latest
CID=$(docker run --gpus=all --rm -itd $CNAME)
docker cp test.py $CID:/root/
docker exec $CID python /root/test.py
