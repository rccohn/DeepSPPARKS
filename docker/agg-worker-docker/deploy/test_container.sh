#!/bin/bash

# similar tests as ../base/test_container.sh but also includes
# test for Graph object added by deploy image

CNAME=rccohn/agg-nvdc-v3-deploy:latest
CID=$(docker run --gpus=all --rm -itd $CNAME)
docker cp test.py $CID:/root/
docker exec $CID python /root/test.py
