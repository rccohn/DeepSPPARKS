#!/bin/bash
# strict mode --> avoid annoying debugging
set -euo pipefail
IFS=$'\n\t' # set field separators to tabs and/or newlines, not spaces

HTPASSWD_FILE=/etc/apache2/.htpasswd

# error if htpasswd file does not exist --> no user authentication
if [ ! -d / ${HTPASSWD_FILE} ];
then
	printf "${HTPASSWD_FILE} not found! Configure user accounts with: "
	printf "$ htpasswd -dbc <file> <user1> <password1>\n"
	printf "or mount existing htpasswd file to container"
	exit 1
fi

# error if mlflow init command is not specified
if [ -z ${MLFLOW_INIT:+x} ];
then
	printf "MLFOW_INIT variable is unset. Set MLFLOW_INIT to desired 
	command to start tracking server"
	exit 1
fi

# TODO need to restart nginx (see https://stackoverflow.com/questions/21830644/non-privileged-non-root-user-to-start-or-restart-webserver-server-such-as-ngin/22014769#22014769)
