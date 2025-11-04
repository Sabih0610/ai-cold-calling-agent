-- === CONFIG (simple K/V) ===
CREATE TABLE IF NOT EXISTS config (
  key   text PRIMARY KEY,
  value text NOT NULL
);

-- sane defaults
INSERT INTO config(key,value) VALUES
 ('DIALER_ENABLED','true'),
 ('DIAL_MODE','TEST'),
 ('CONCURRENCY','1'),
 ('CALL_WINDOW_DEFAULT_START','09:00'),
 ('CALL_WINDOW_DEFAULT_END','18:00'),
 ('DEFAULT_TZ','America/Chicago'),
 ('MAX_ATTEMPTS','3'),
 ('DIAL_STRING_TEST','Local/7000@default'),
 ('DIAL_STRING_LIVE','PJSIP/${PHONE_E164}@carrier_us_1'),
 ('CALLER_ID_E164','+15005550006')        -- change after you verify a real caller ID
ON CONFLICT (key) DO NOTHING;

-- === CAMPAIGN-LEAD STATE (lives per-campaign) ===
ALTER TABLE campaign_leads
  ADD COLUMN IF NOT EXISTS status        text        DEFAULT 'NEW',
  ADD COLUMN IF NOT EXISTS ready         boolean     DEFAULT true,
  ADD COLUMN IF NOT EXISTS attempts      int         DEFAULT 0,
  ADD COLUMN IF NOT EXISTS last_called_at timestamptz,
  ADD COLUMN IF NOT EXISTS next_action_at timestamptz,
  ADD COLUMN IF NOT EXISTS agent_notes   text;

-- helpful indexes
CREATE INDEX IF NOT EXISTS ix_cl_state
  ON campaign_leads (campaign_id, status, ready, next_action_at);

-- === UNIQUE CONSTRAINTS that import_all.py expects ===
-- remove partial uniques if they exist (safe if missing)
DROP INDEX IF EXISTS leads_phone_unique;
DROP INDEX IF EXISTS leads_email_unique;

-- dedupe by phone (keep the most recent)
WITH d AS (
  SELECT id, phone_e164,
         row_number() OVER (PARTITION BY phone_e164
                            ORDER BY last_seen DESC NULLS LAST, source_first_seen DESC) AS rn
  FROM leads WHERE phone_e164 IS NOT NULL
)
DELETE FROM leads l USING d
WHERE l.id = d.id AND d.rn > 1;

-- dedupe by email (keep the most recent)
WITH d AS (
  SELECT id, email,
         row_number() OVER (PARTITION BY email
                            ORDER BY last_seen DESC NULLS LAST, source_first_seen DESC) AS rn
  FROM leads WHERE email IS NOT NULL
)
DELETE FROM leads l USING d
WHERE l.id = d.id AND d.rn > 1;

-- now install whole-table unique constraints (names match your Python)
ALTER TABLE leads
  ADD CONSTRAINT IF NOT EXISTS leads_phone_unique_constraint UNIQUE (phone_e164);
ALTER TABLE leads
  ADD CONSTRAINT IF NOT EXISTS leads_email_unique_constraint UNIQUE (email);

-- === BACKFILL tz_name from your cities table (once) ===
UPDATE leads l
SET tz_name = c.tz_name
FROM cities c
WHERE l.tz_name IS NULL AND l.city = c.city AND l.state = c.state;

-- === ELIGIBILITY VIEW ===
DROP VIEW IF EXISTS leads_eligible_now;
CREATE VIEW leads_eligible_now AS
WITH cfg AS (
  SELECT
    (SELECT value::time FROM config WHERE key='CALL_WINDOW_DEFAULT_START') AS win_start,
    (SELECT value::time FROM config WHERE key='CALL_WINDOW_DEFAULT_END')   AS win_end,
    (SELECT value::int  FROM config WHERE key='MAX_ATTEMPTS')              AS max_attempts,
    (SELECT value       FROM config WHERE key='DEFAULT_TZ')                AS def_tz
),
joined AS (
  SELECT
    cl.campaign_id, cl.lead_id, cl.status, cl.ready, cl.attempts, cl.last_called_at, cl.next_action_at,
    l.company, l.email, l.phone_raw, l.phone_e164, l.website, l.address, l.city, l.state, l.postal_code, l.tz_name
  FROM campaign_leads cl
  JOIN leads l ON l.id = cl.lead_id
  LEFT JOIN dnc d ON d.phone_e164 = l.phone_e164
  WHERE d.phone_e164 IS NULL
)
SELECT j.*,
       ((now() AT TIME ZONE COALESCE(j.tz_name, (SELECT def_tz FROM cfg)))::time) AS local_time,
       (SELECT max_attempts FROM cfg) AS eff_max_attempts,
       (SELECT win_start FROM cfg)    AS eff_win_start,
       (SELECT win_end   FROM cfg)    AS eff_win_end
FROM joined j
WHERE j.ready = true
  AND j.status IN ('NEW','CALLBACK_DUE','IN_PROGRESS')
  AND j.attempts < (SELECT max_attempts FROM cfg)
  AND COALESCE(j.next_action_at, now()) <= now()
  AND (
        ( (now() AT TIME ZONE COALESCE(j.tz_name,(SELECT def_tz FROM cfg)))::time
            BETWEEN (SELECT win_start FROM cfg) AND (SELECT win_end FROM cfg) )
        OR
        -- optional overnight window support:
        ( (SELECT win_start FROM cfg) > (SELECT win_end FROM cfg)
          AND ( (now() AT TIME ZONE COALESCE(j.tz_name,(SELECT def_tz FROM cfg)))::time >= (SELECT win_start FROM cfg)
                OR  (now() AT TIME ZONE COALESCE(j.tz_name,(SELECT def_tz FROM cfg)))::time <= (SELECT win_end FROM cfg) ) )
      );

-- === CALL ATTEMPTS table (you already had one; ensure columns) ===
ALTER TABLE call_attempts
  ADD COLUMN IF NOT EXISTS trunk         text,
  ADD COLUMN IF NOT EXISTS sip_call_id   text;
