# code to start tracking server on vm running gcp Container Optimized OS

# NOTE: if running as a startup script, it is not run from your home directory
# it is probably easier to hard-code paths and docker-compose image run command
# into the script

# non-interactive shells (like ones from running scripts) don't use aliases
shopt -s expand_aliases  # turn alias on so it will work
source .env # get environment variables like ${MLFLOW_PORT} and ${PPORT}
source compose_as_container.sh # get "doco" command 
#     (run docker-compose through docker container 
#      with current directory mounted as volume)

# docker compose up 
doco up

# alter ip tables (linux sw firewall- machine level (ie cannot control through gcp))
# allow incoming traffic to mlflow server on port 5000
sudo iptables -A INPUT -p tcp --dport ${MLFLOW_PORT} -j ACCEPT

# not needed for mlflow tracking server, but needed to access
# db directly (and for debugging with netcat)
sudo iptables -A INPUT -p tcp --dport ${PPORT} -j ACCEPT


