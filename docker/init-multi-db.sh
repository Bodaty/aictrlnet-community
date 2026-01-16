#!/bin/bash
set -e

# Parse comma-separated database names
IFS=',' read -ra DBS <<< "$POSTGRES_MULTIPLE_DATABASES"

# Create each database
for db in "${DBS[@]}"; do
    echo "Creating database: $db"
    psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" <<-EOSQL
        CREATE DATABASE $db;
        GRANT ALL PRIVILEGES ON DATABASE $db TO $POSTGRES_USER;
EOSQL
done