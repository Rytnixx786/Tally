#!/bin/sh
# Runs once, on first start of an empty Postgres volume. Creates roles, schemas and raw tables.
set -eu
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" \
  -v writer_pw="$TALLY_WRITER_PASSWORD" -v lexicon_pw="$LEXICON_RO_PASSWORD" \
  -f /docker-entrypoint-initdb.d/sql/schema.sql
