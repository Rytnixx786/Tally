-- Tally warehouse bootstrap (CONTRACTS.md §3). Idempotent: safe to run again via `tally db init`.

DO $$ BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'tally_writer') THEN
    CREATE ROLE tally_writer LOGIN;
  END IF;
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'lexicon_ro') THEN
    CREATE ROLE lexicon_ro LOGIN;
  END IF;
END $$;

ALTER ROLE tally_writer PASSWORD :'writer_pw';
ALTER ROLE lexicon_ro PASSWORD :'lexicon_pw';
-- Lexicon's agent gets a read-only role with a hard statement timeout (Lexicon ADR-003).
ALTER ROLE lexicon_ro SET default_transaction_read_only = on;
ALTER ROLE lexicon_ro SET statement_timeout = '5s';

CREATE SCHEMA IF NOT EXISTS raw AUTHORIZATION tally_writer;
CREATE SCHEMA IF NOT EXISTS staging AUTHORIZATION tally_writer;
CREATE SCHEMA IF NOT EXISTS intermediate AUTHORIZATION tally_writer;
CREATE SCHEMA IF NOT EXISTS marts AUTHORIZATION tally_writer;
GRANT CREATE ON DATABASE tally TO tally_writer;

SET ROLE tally_writer;

CREATE TABLE IF NOT EXISTS raw.otel_spans (
  trace_id        text        NOT NULL,
  span_id         text        NOT NULL,
  parent_span_id  text,
  name            text        NOT NULL,
  kind            smallint,
  start_time      timestamptz NOT NULL,
  end_time        timestamptz NOT NULL,
  status_code     smallint,
  service_name    text,
  scope_name      text,
  attributes      jsonb       NOT NULL,
  resource        jsonb,
  ingested_at     timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (trace_id, span_id)
);
CREATE INDEX IF NOT EXISTS otel_spans_request_id ON raw.otel_spans ((attributes->>'tally.request_id'));
CREATE INDEX IF NOT EXISTS otel_spans_cinch_request_id ON raw.otel_spans ((attributes->>'cinch.request_id'));
CREATE INDEX IF NOT EXISTS otel_spans_decision_id ON raw.otel_spans ((attributes->>'tally.decision_id'));

CREATE TABLE IF NOT EXISTS raw.eval_outcomes (
  run_id            text        NOT NULL,
  decision_id       uuid        PRIMARY KEY,
  ticket_id         text        NOT NULL,
  split             text        NOT NULL,
  arm               text        NOT NULL,
  tenant_id         text        NOT NULL,
  setup             text        NOT NULL,
  prompt_version    text        NOT NULL,
  gold_intent       text        NOT NULL,
  gold_category     text        NOT NULL,
  predicted_intent  text,
  predicted_category text,
  valid             boolean     NOT NULL,
  confidence        double precision,
  calibrated_confidence double precision,
  escalated         boolean     NOT NULL,
  final_request_id  uuid        NOT NULL,
  flags             text,
  decided_at        timestamptz NOT NULL,
  loaded_at         timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS raw.call_costs (
  request_id   uuid PRIMARY KEY,
  run_id       text NOT NULL,
  cost_usd     numeric(14, 10) NOT NULL,
  cost_method  text NOT NULL,
  loaded_at    timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS raw.prompt_registry (
  prompt_version text PRIMARY KEY,
  feature        text NOT NULL,
  prompt_hash    text NOT NULL,
  template_path  text NOT NULL,
  n_fewshot      int  NOT NULL,
  description    text NOT NULL,
  created_at     timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS raw.price_book (
  provider        text NOT NULL,
  model           text NOT NULL,
  input_per_mtok  numeric NOT NULL,
  output_per_mtok numeric NOT NULL,
  cached_input_per_mtok numeric,
  checked_on      date NOT NULL,
  source          text NOT NULL,
  PRIMARY KEY (provider, model, checked_on)
);

ALTER TABLE raw.price_book ADD COLUMN IF NOT EXISTS cached_input_per_mtok numeric;

CREATE TABLE IF NOT EXISTS raw.tenants (
  tenant_id          text PRIMARY KEY,
  display_name       text NOT NULL,
  owner_project      text NOT NULL,
  monthly_budget_usd numeric NOT NULL,
  traffic_mix        jsonb NOT NULL
);

CREATE TABLE IF NOT EXISTS raw.run_manifest (
  run_id     text PRIMARY KEY,
  manifest   jsonb NOT NULL,
  loaded_at  timestamptz NOT NULL DEFAULT now()
);

RESET ROLE;

GRANT USAGE ON SCHEMA marts TO lexicon_ro;
ALTER DEFAULT PRIVILEGES FOR ROLE tally_writer IN SCHEMA marts GRANT SELECT ON TABLES TO lexicon_ro;
