# run on gcp container optimized-os, which doesn't have docker-compose
# instead, we run docker-compose itself AS a docker image 

# TODO: finish and test this

# alter iptables

COMPOSE_IMAGE=docker/compose:1.29.2

# echo $COMPOSE_IMAGE

alias doco='docker run -itd --rm --network="host" \
	-v /var/run/docker.sock:/var/run/docker.sock \
	-v "$PWD:$PWD" \
	-w="$PWD" \
	${COMPOSE_IMAGE}' 

