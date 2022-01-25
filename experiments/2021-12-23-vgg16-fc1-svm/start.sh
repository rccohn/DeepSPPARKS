# !/bin/bash

echo "starting experiment now"
# -u: force stdout and stderr to be unbuffered
# allowing outputs to be seen when run in docker
python -u vgg16-fc1.py
