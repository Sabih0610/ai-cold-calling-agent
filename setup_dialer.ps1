# setup_dialer.ps1
$ErrorActionPreference = 'Stop'

# --- config ---
$DbName = 'dialer'
$DbUser = 'dialer'
$DbPass = 'Letmein$100'    # your chosen password (used for the 'dialer' role)
$PgSuper = 'postgres'      # superuser name

# Path to psql.exe
$PgBin  = 'E:\Program Files\PostgreSQL\17\bin'
$Psql   = Join-Path $PgBin 'psql.exe'
if (-not (Test-Path $Psql)) { throw "psql not found at: $Psql" }

# Repo paths
$RepoRoot  = Split-Path -Parent $MyInvocation.MyCommand.Path
$SchemaDir = Join-Path $RepoRoot 'database\schema'
$SchemaSql = Join-Path $SchemaDir 'schema.sql'
$SeedSql   = Join-Path $SchemaDir 'seed_cities.sql'
if (-not (Test-Path $SchemaSql)) { throw "Schema file not found: $SchemaSql" }
if (-not (Test-Path $SeedSql))   { throw "Seed file not found: $SeedSql" }

# Superuser password for this run (single quotes keep the $ literal)
$env:PGPASSWORD = 'Letmein$100'

function RunPsql([string[]]$args) {
  & $Psql @args
  if ($LASTEXITCODE -ne 0) { throw "psql failed: $($args -join ' ')" }
}

# Ensure DB
RunPsql @('-U', $PgSuper, '-d', 'postgres', '-v','ON_ERROR_STOP=1',
  '-c', "SELECT 'CREATE DATABASE $DbName' WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = '$DbName')\gexec")

# Ensure role
RunPsql @('-U', $PgSuper, '-d', 'postgres', '-v','ON_ERROR_STOP=1',
  '-c', "DO $$ BEGIN IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname='$DbUser')
          THEN CREATE ROLE $DbUser LOGIN PASSWORD '$DbPass'; END IF; END $$;")

# Grants/ownership
RunPsql @('-U', $PgSuper, '-d', 'postgres', '-v','ON_ERROR_STOP=1',
  '-c', "GRANT ALL PRIVILEGES ON DATABASE $DbName TO $DbUser;")
RunPsql @('-U', $PgSuper, '-d', 'postgres', '-v','ON_ERROR_STOP=1',
  '-c', "ALTER DATABASE $DbName OWNER TO $DbUser;")
RunPsql @('-U', $PgSuper, '-d', $DbName,  '-v','ON_ERROR_STOP=1',
  '-c', "ALTER SCHEMA public OWNER TO $DbUser;")

# Run schema as superuser (for CREATE EXTENSION), then seed as dialer
RunPsql @('-U', $PgSuper, '-d', $DbName, '-v','ON_ERROR_STOP=1', '-f', $SchemaSql)
RunPsql @('-U', $DbUser,  '-d', $DbName, '-v','ON_ERROR_STOP=1', '-f', $SeedSql)

# Verify
RunPsql @('-U', $DbUser, '-d', $DbName, '-c', '\dt')
RunPsql @('-U', $DbUser, '-d', $DbName, '-c', 'SELECT COUNT(*) AS total_cities FROM cities;')

# Clean up the env var after run
Remove-Item Env:\PGPASSWORD -ErrorAction SilentlyContinue
Write-Host "`n✅ PostgreSQL Dialer DB setup complete!"
