# use container to generate htpasswd file with info name:user1 password:pass1 and save it to current directory / htpasswd
docker run --rm  -v ${PWD}:/home/mlflow test-mlflow-htpasswd htpasswd -dbc /home/mlflow/htpasswd user1 pass1
Adding password for user user1
