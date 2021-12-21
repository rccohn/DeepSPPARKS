source .env
psql "host=${PHOST} port=${PPORT} sslmode=disable dbname=${PNAME} user=${PUSER} password=${PPWD}"
