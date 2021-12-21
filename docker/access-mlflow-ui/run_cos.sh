source compose_as_container.sh # get "doco" command 
#     (run docker-compose through docker container 
#      with current directory mounted as volume)

# docker compose up 
doco up

# alter ip tables (linux sw firewall- machine level (ie cannot control through gcp))
# allow incoming traffic to mlflow server on port 5000
sudo iptables -A input -p tcp --dport 5000 -j ACCEPT

# not needed for mlflow tracking server, but needed to access
# db directly (and for debugging with netcat)
sudo iptables -A input -p tcp --dport ${PPORT} -j ACCEPT


