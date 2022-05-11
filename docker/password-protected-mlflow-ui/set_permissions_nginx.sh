#!/bin/bash
# set permissions so nginx can be run as non-root user in docker
set -eou pipefail
IFS=$'\n'


files=(
	"/var/log/nginx/"
	"/var/run/nginx.pid"
	"/var/lib/nginx/"
)

touch "/var/run/nginx.pid"

for file in ${files[@]}
do
	chown -R ${USER_UID}:${USER_GID} $file;
done
