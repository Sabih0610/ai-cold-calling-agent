-- ============================================================
-- Dialer DB bootstrap (run via psql on Windows)
-- Creates database (if missing), schemas, enums, tables, funcs
-- File: F:\ai cold calling agent\database\schema\dialer_schema.sql
-- ============================================================

-- 0) Create database if it doesn't exist, then connect
SELECT 'CREATE DATABASE dialerdb'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'dialerdb')\gexec

\connect dialerdb

-- 1) Extensions
CREATE EXTENSION IF NOT EXISTS pgcrypto;   -- for gen_random_uuid()

-- 2) Schemas
CREATE SCHEMA IF NOT EXISTS dialer;
CREATE SCHEMA IF NOT EXISTS stage;

-- 3) Enums
DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'lead_status') THEN
    CREATE TYPE dialer.lead_status AS ENUM (
      'NEW','IN_PROGRESS','CALLBACK_DUE','NOANSWER','BUSY','VOICEMAIL',
      'BAD_NUMBER','DNC','COMPLETED','FAILED'
    );
  END IF;

  IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'campaign_status') THEN
    CREATE TYPE dialer.campaign_status AS ENUM ('ACTIVE','PAUSED','ARCHIVED');
  END IF;

  IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'disposition') THEN
    CREATE TYPE dialer.disposition AS ENUM (
      'NEW','IN_PROGRESS','CALLBACK_DUE','NOANSWER','BUSY','VOICEMAIL',
      'BAD_NUMBER','DNC','COMPLETED','FAILED'
    );
  END IF;

  IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'role') THEN
    CREATE TYPE dialer.role AS ENUM ('ADMIN','MANAGER','AGENT','VIEWER');
  END IF;
END$$;

-- 4) Helper table: config (K/V store)
CREATE TABLE IF NOT EXISTS dialer.config (
  key   text PRIMARY KEY,
  value text NOT NULL,
  scope text NOT NULL DEFAULT 'global'
);

-- tiny helper to fetch config with default
CREATE OR REPLACE FUNCTION dialer.cfg(p_key text, p_default text)
RETURNS text LANGUAGE sql STABLE AS $$
  SELECT COALESCE((SELECT value FROM dialer.config WHERE key = p_key LIMIT 1), p_default)
$$;

-- 5) Users (for admin UI / assignments)
CREATE TABLE IF NOT EXISTS dialer.users (
  id             uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  email          text UNIQUE NOT NULL,
  password_hash  text NOT NULL,
  role           dialer.role NOT NULL DEFAULT 'VIEWER',
  created_at     timestamptz NOT NULL DEFAULT now(),
  last_login_at  timestamptz
);

-- 6) Campaigns
CREATE TABLE IF NOT EXISTS dialer.campaigns (
  id                  uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  name                text UNIQUE NOT NULL,
  status              dialer.campaign_status NOT NULL DEFAULT 'ACTIVE',

  -- Defaults (nullable = use global config if null)
  call_window_start   time,
  call_window_end     time,
  max_attempts        int,
  retry_map           jsonb,              -- e.g. {"NOANSWER":"24 hours","BUSY":"2 hours"}
  dial_mode           text,               -- TEST | LIVE
  dial_string_test    text,               -- e.g. Local/7000@default
  dial_string_live    text,               -- e.g. PJSIP/${PHONE_E164}@carrier_us_1
  caller_id_e164      text,
  notes               text,

  created_at          timestamptz NOT NULL DEFAULT now(),
  updated_at          timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_campaigns_status ON dialer.campaigns(status);

-- 7) Cities → Timezone mapping
CREATE TABLE IF NOT EXISTS dialer.cities (
  city  text NOT NULL,
  state text NOT NULL,
  tz    text NOT NULL,      -- IANA tz, e.g. America/Chicago
  PRIMARY KEY (city, state)
);
CREATE INDEX IF NOT EXISTS idx_cities_state ON dialer.cities(state);

-- 8) Leads
CREATE TABLE IF NOT EXISTS dialer.leads (
  id             uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  campaign_id    uuid NOT NULL REFERENCES dialer.campaigns(id) ON DELETE CASCADE,

  -- Identity
  business_name  text,
  first_name     text,
  last_name      text,

  -- Contact
  phone_e164     text NOT NULL,       -- normalized later to +1XXXXXXXXXX for NANPA
  email          text,
  website        text,

  -- Geo / time
  city           text,
  state          text,
  country        text,
  tz             text,                -- IANA tz

  -- State
  ready          boolean NOT NULL DEFAULT true,
  status         dialer.lead_status NOT NULL DEFAULT 'NEW',

  -- Counters / scheduling
  attempts       int NOT NULL DEFAULT 0,
  last_called_at timestamptz,
  next_action_at timestamptz,

  -- Source provenance
  source_file    text,
  source_path    text,
  source_hash    text,

  -- Assignment
  assigned_to    uuid REFERENCES dialer.users(id),

  created_at     timestamptz NOT NULL DEFAULT now(),
  updated_at     timestamptz NOT NULL DEFAULT now()
);

-- dedupe: (campaign, phone, city, normalized business_name)
-- Treat empty business_name as '~'
CREATE UNIQUE INDEX IF NOT EXISTS ux_leads_dedupe
  ON dialer.leads (
    campaign_id,
    phone_e164,
    city,
    COALESCE(NULLIF(business_name, ''), '~')
  );

CREATE INDEX IF NOT EXISTS idx_leads_status_ready ON dialer.leads(status, ready);
CREATE INDEX IF NOT EXISTS idx_leads_next_action   ON dialer.leads(next_action_at);
CREATE INDEX IF NOT EXISTS idx_leads_tz            ON dialer.leads(tz);
CREATE INDEX IF NOT EXISTS idx_leads_phone         ON dialer.leads(phone_e164);

-- 9) DNC (global)
CREATE TABLE IF NOT EXISTS dialer.dnc (
  phone_e164 text PRIMARY KEY,
  reason     text,
  added_by   uuid REFERENCES dialer.users(id),
  added_at   timestamptz NOT NULL DEFAULT now()
);

-- 10) Call attempts / CDR-light
CREATE TABLE IF NOT EXISTS dialer.call_attempts (
  id            uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  lead_id       uuid NOT NULL REFERENCES dialer.leads(id) ON DELETE CASCADE,
  campaign_id   uuid NOT NULL REFERENCES dialer.campaigns(id) ON DELETE CASCADE,

  started_at    timestamptz NOT NULL,
  ended_at      timestamptz,
  duration_sec  int,

  sip_call_id   text,
  channel       text,
  trunk         text,

  disposition   dialer.disposition,
  recording_url text,
  agent_notes   text
);
CREATE INDEX IF NOT EXISTS idx_call_attempts_lead_time
  ON dialer.call_attempts(lead_id, started_at DESC);
CREATE INDEX IF NOT EXISTS idx_call_attempts_campaign_time
  ON dialer.call_attempts(campaign_id, started_at DESC);

-- 11) Imports tracker
CREATE TABLE IF NOT EXISTS dialer.imports (
  id           uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  source_path  text,
  file_count   int,
  row_count    bigint,
  errors       jsonb,
  started_at   timestamptz NOT NULL DEFAULT now(),
  finished_at  timestamptz,
  status       text
);

-- 12) Caller ID pools (for LIVE later)
CREATE TABLE IF NOT EXISTS dialer.caller_id_pools (
  id          uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  label       text,
  country     text,
  state       text,
  number_e164 text UNIQUE,
  verified    boolean NOT NULL DEFAULT false,
  active      boolean NOT NULL DEFAULT true
);

-- 13) Stage table for raw ingest (do not index heavily)
CREATE TABLE IF NOT EXISTS stage.raw (
  id            bigserial PRIMARY KEY,
  campaign_hint text,
  business_name text,
  first_name    text,
  last_name     text,
  phone         text,
  email         text,
  website       text,
  city          text,
  state         text,
  country       text,
  source_file   text,
  source_path   text,
  source_hash   text,
  loaded_at     timestamptz NOT NULL DEFAULT now()
);

-- 14) Utility: normalize NANPA phone to +1XXXXXXXXXX; return NULL if invalid
CREATE OR REPLACE FUNCTION dialer.normalize_nanpa_phone(p_in text)
RETURNS text LANGUAGE plpgsql IMMUTABLE AS $$
DECLARE
  digits text;
BEGIN
  IF p_in IS NULL THEN
    RETURN NULL;
  END IF;

  -- keep only digits
  digits := regexp_replace(p_in, '\D', '', 'g');

  IF length(digits) = 10 THEN
    RETURN '+1' || digits;
  ELSIF length(digits) = 11 AND left(digits,1) = '1' THEN
    RETURN '+1' || right(digits,10);
  ELSE
    RETURN NULL; -- not a valid NANPA 10/11-digit number
  END IF;
END$$;

-- 15) Generic updated_at trigger
CREATE OR REPLACE FUNCTION dialer.set_updated_at()
RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
  NEW.updated_at := now();
  RETURN NEW;
END$$;

-- 16) Leads BEFORE trigger: normalize phone; empty strings → NULLs
CREATE OR REPLACE FUNCTION dialer.leads_before_ins_upd()
RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
  -- normalize phone
  NEW.phone_e164 := dialer.normalize_nanpa_phone(NEW.phone_e164);

  -- coerce empty strings to NULL
  IF NEW.business_name = '' THEN NEW.business_name := NULL; END IF;
  IF NEW.first_name    = '' THEN NEW.first_name    := NULL; END IF;
  IF NEW.last_name     = '' THEN NEW.last_name     := NULL; END IF;
  IF NEW.email         = '' THEN NEW.email         := NULL; END IF;
  IF NEW.website       = '' THEN NEW.website       := NULL; END IF;
  IF NEW.city          = '' THEN NEW.city          := NULL; END IF;
  IF NEW.state         = '' THEN NEW.state         := NULL; END IF;
  IF NEW.country       = '' THEN NEW.country       := NULL; END IF;
  IF NEW.tz            = '' THEN NEW.tz            := NULL; END IF;

  RETURN NEW;
END$$;

-- 17) DNC AFTER INSERT: immediately mark matching leads as DNC
CREATE OR REPLACE FUNCTION dialer.dnc_after_insert()
RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
  UPDATE dialer.leads
     SET status = 'DNC',
         ready = false,
         next_action_at = NULL,
         updated_at = now()
   WHERE phone_e164 = NEW.phone_e164;

  RETURN NEW;
END$$;

-- 18) Attach triggers
DROP TRIGGER IF EXISTS trg_campaigns_set_updated_at ON dialer.campaigns;
CREATE TRIGGER trg_campaigns_set_updated_at
BEFORE UPDATE ON dialer.campaigns
FOR EACH ROW EXECUTE FUNCTION dialer.set_updated_at();

DROP TRIGGER IF EXISTS trg_leads_set_updated_at ON dialer.leads;
CREATE TRIGGER trg_leads_set_updated_at
BEFORE UPDATE ON dialer.leads
FOR EACH ROW EXECUTE FUNCTION dialer.set_updated_at();

DROP TRIGGER IF EXISTS trg_users_set_updated_at ON dialer.users;
CREATE TRIGGER trg_users_set_updated_at
BEFORE UPDATE ON dialer.users
FOR EACH ROW EXECUTE FUNCTION dialer.set_updated_at();

DROP TRIGGER IF EXISTS trg_leads_before_ins_upd ON dialer.leads;
CREATE TRIGGER trg_leads_before_ins_upd
BEFORE INSERT OR UPDATE ON dialer.leads
FOR EACH ROW EXECUTE FUNCTION dialer.leads_before_ins_upd();

DROP TRIGGER IF EXISTS trg_dnc_after_insert ON dialer.dnc;
CREATE TRIGGER trg_dnc_after_insert
AFTER INSERT ON dialer.dnc
FOR EACH ROW EXECUTE FUNCTION dialer.dnc_after_insert();

-- 19) Eligibility view (who we can call right now)
CREATE OR REPLACE VIEW dialer.leads_eligible_now AS
WITH cfg AS (
  SELECT
    (dialer.cfg('CALL_WINDOW_DEFAULT_START','09:00'))::time AS win_start,
    (dialer.cfg('CALL_WINDOW_DEFAULT_END','18:00'))::time   AS win_end,
    (dialer.cfg('MAX_ATTEMPTS','3'))::int                   AS max_attempts_default
),
eff AS (
  SELECT
    l.*,
    c.name AS campaign_name,
    COALESCE(c.max_attempts, (SELECT max_attempts_default FROM cfg)) AS eff_max_attempts,
    COALESCE(c.call_window_start, (SELECT win_start FROM cfg)) AS eff_win_start,
    COALESCE(c.call_window_end,   (SELECT win_end   FROM cfg)) AS eff_win_end
  FROM dialer.leads l
  JOIN dialer.campaigns c ON c.id = l.campaign_id
  LEFT JOIN dialer.dnc d  ON d.phone_e164 = l.phone_e164
  WHERE d.phone_e164 IS NULL
)
SELECT *
FROM (
  SELECT
    e.*,
    -- Local time in lead's tz; fallback to DEFAULT_TZ config if tz missing
    (now() AT TIME ZONE COALESCE(e.tz, dialer.cfg('DEFAULT_TZ','America/Chicago')))::time AS local_time
  FROM eff e
) z
WHERE
  z.ready = true
  AND z.status IN ('NEW','CALLBACK_DUE','IN_PROGRESS')
  AND z.attempts < z.eff_max_attempts
  AND COALESCE(z.next_action_at, now()) <= now()
  AND (
        -- simple window (start < end)
        (z.local_time >= z.eff_win_start AND z.local_time <= z.eff_win_end)
      OR
        -- handle overnight window (start > end) e.g., 20:00..08:00
        (z.eff_win_start > z.eff_win_end AND (z.local_time >= z.eff_win_start OR z.local_time <= z.eff_win_end))
      );

-- 20) Seed global config (safe upserts)
INSERT INTO dialer.config(key,value) VALUES
 ('DIALER_ENABLED','true')
ON CONFLICT (key) DO UPDATE SET value=EXCLUDED.value;

INSERT INTO dialer.config(key,value) VALUES
 ('CONCURRENCY','2'),
 ('DEFAULT_TZ','America/Chicago'),
 ('CALL_WINDOW_DEFAULT_START','09:00'),
 ('CALL_WINDOW_DEFAULT_END','18:00'),
 ('MAX_ATTEMPTS','3'),
 ('QUIET_HOURS_LOCAL','21:00-09:00'),
 ('DIAL_STRING_TEST','Local/7000@default'),
 ('DIAL_STRING_LIVE','PJSIP/${PHONE_E164}@carrier_us_1')
ON CONFLICT (key) DO NOTHING;

-- 21) (Optional) Quick seed cities for testing
INSERT INTO dialer.cities(city,state,tz) VALUES
 ('Austin','TX','America/Chicago'),
 ('Dallas','TX','America/Chicago'),
 ('Houston','TX','America/Chicago'),
 ('Miami','FL','America/New_York'),
 ('New York','NY','America/New_York'),
 ('Los Angeles','CA','America/Los_Angeles')
ON CONFLICT DO NOTHING;

-- ===================== Done =====================
