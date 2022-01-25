# !/bin/bash
echo "starting experiment now"
# -u: force stdout and stderr to be unbuffered
# hopefully this makes them show up when running?
python -u sgc_combined.py
