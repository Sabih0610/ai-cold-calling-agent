-- Create UNIQUE(phone_e164) named exactly as import_all.py expects
-- database\schema\patch_mvp_fix_constraints.sql
DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_constraint
    WHERE conname = 'leads_phone_unique_constraint'
  ) THEN
    ALTER TABLE leads
      ADD CONSTRAINT leads_phone_unique_constraint UNIQUE (phone_e164);
  END IF;
END$$;

-- Create UNIQUE(email) named exactly as import_all.py expects
DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_constraint
    WHERE conname = 'leads_email_unique_constraint'
  ) THEN
    ALTER TABLE leads
      ADD CONSTRAINT leads_email_unique_constraint UNIQUE (email);
  END IF;
END$$;
