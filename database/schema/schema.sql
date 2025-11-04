-- Extensions
CREATE EXTENSION IF NOT EXISTS citext;
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- Campaigns (one per immediate subfolder, e.g. "Auto repair")
CREATE TABLE IF NOT EXISTS campaigns (
  id           BIGSERIAL PRIMARY KEY,
  name         TEXT NOT NULL UNIQUE,
  slug         TEXT UNIQUE,
  notes        TEXT,
  created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Cities/Timezones (seed below for your listed cities)
CREATE TABLE IF NOT EXISTS cities (
  id        BIGSERIAL PRIMARY KEY,
  city      TEXT NOT NULL,
  state     TEXT NOT NULL,
  tz_name   TEXT NOT NULL,
  UNIQUE (city, state)
);

-- Master leads (dedup across ALL campaigns)
CREATE TABLE IF NOT EXISTS leads (
  id              BIGSERIAL PRIMARY KEY,
  company         CITEXT,
  contact         CITEXT,
  email           CITEXT,
  phone_raw       TEXT,
  phone_e164      TEXT,              -- normalized US number, e.g. +15125551234
  website         TEXT,
  address         TEXT,
  city            TEXT,
  state           TEXT,
  postal_code     TEXT,
  country_code    CHAR(2) NOT NULL DEFAULT 'US',
  tz_name         TEXT,              -- looked up from cities or inferred
  source_first_seen TIMESTAMPTZ NOT NULL DEFAULT now(),
  last_seen       TIMESTAMPTZ,
  source_file     TEXT,              -- last file we saw it in
  source_row      JSONB              -- original row snapshot (flexible)
);

-- Dedup constraints (skip if null)
CREATE UNIQUE INDEX IF NOT EXISTS leads_phone_unique
  ON leads (phone_e164) WHERE phone_e164 IS NOT NULL;

CREATE UNIQUE INDEX IF NOT EXISTS leads_email_unique
  ON leads (email) WHERE email IS NOT NULL;

-- Fast lookups
CREATE INDEX IF NOT EXISTS leads_city_state_idx ON leads (state, city);
CREATE INDEX IF NOT EXISTS leads_source_row_gin ON leads USING GIN (source_row);

-- A lead can belong to many campaigns (join table)
CREATE TABLE IF NOT EXISTS campaign_leads (
  campaign_id   BIGINT NOT NULL REFERENCES campaigns(id) ON DELETE CASCADE,
  lead_id       BIGINT NOT NULL REFERENCES leads(id) ON DELETE CASCADE,
  source_city   TEXT,
  source_state  TEXT,
  source_file   TEXT,
  created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
  PRIMARY KEY (campaign_id, lead_id)
);

-- Do-Not-Call
CREATE TABLE IF NOT EXISTS dnc (
  phone_e164 TEXT PRIMARY KEY,
  reason     TEXT,
  added_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Import runs (for idempotency + audit)
CREATE TABLE IF NOT EXISTS imports (
  id            BIGSERIAL PRIMARY KEY,
  campaign_id   BIGINT REFERENCES campaigns(id) ON DELETE SET NULL,
  list_path     TEXT,
  file_name     TEXT,
  file_sha1     TEXT,
  rows_total    INT,
  rows_loaded   INT,
  rows_skipped  INT,
  status        TEXT,  -- queued|running|done|skipped|error
  log           TEXT,
  started_at    TIMESTAMPTZ DEFAULT now(),
  finished_at   TIMESTAMPTZ
);

-- (Optional, already future-proof for dialer)
CREATE TABLE IF NOT EXISTS call_attempts (
  id            BIGSERIAL PRIMARY KEY,
  lead_id       BIGINT NOT NULL REFERENCES leads(id) ON DELETE CASCADE,
  campaign_id   BIGINT REFERENCES campaigns(id) ON DELETE SET NULL,
  started_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
  disposition   TEXT,            -- ANSWERED/BUSY/NOANSWER/FAILED/etc.
  duration_sec  INT,
  recording_url TEXT,
  notes         TEXT
);
