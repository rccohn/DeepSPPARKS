#!/bin/bash
# quick check to make sure container doesn't have fatal dependency issues
CNAME=rccohn/agg-utils-docker-base:latest
CID=$(docker run --rm -itd $CNAME)
docker cp test.py $CID:/root/
docker exec $CID python /root/test.py

