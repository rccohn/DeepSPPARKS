# run on gcp container optimized-os, which doesn't have docker-compose
# instead, we run docker-compose itself AS a docker image 

COMPOSE_IMAGE=docker/compose:1.29.2

# echo $COMPOSE_IMAGE

# note- don't use --name argument, otherwise you won't be able to run doco down after runnig
# doco up, as it will try to start a container with the same name
alias doco='docker run -itv --rm --network="host" \
	-v /var/run/docker.sock:/var/run/docker.sock \
	-v "$PWD:$PWD" \
	-w="$PWD" \
	${COMPOSE_IMAGE}'

