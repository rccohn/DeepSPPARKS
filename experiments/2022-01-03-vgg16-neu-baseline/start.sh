# !/bin/bash
echo "starting experiment now"
# -u: force stdout and stderr to be unbuffered
# allowing outputs to be seen when run in docker
echo "test" > /root/data/processed/hello.txt
python -u neu-baseline.py /root/inputs/params.yaml

